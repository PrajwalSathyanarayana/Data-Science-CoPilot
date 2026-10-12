"""Tool — builds the report's charts from the other tools' findings, as compact Plotly figures counted by pandas."""
import json
import logging
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..agents.planner import parse_date_column
from ..config import (
    CHART_HIST_BINS,
    CHART_MAX_BARS,
    CHART_MAX_BOXES,
    CHART_MAX_DATE_COLUMNS,
    CHART_MAX_DISTRIBUTIONS,
    CHART_MAX_TIME_POINTS,
    CHART_MAX_TOTAL,
    CHART_SCATTER_SAMPLE,
    OUTLIER_FEW_VALUES,
    TOOL_SUMMARY_MAX_TOKENS,
)
from ..models import ColumnFlag, ToolResult
from .correlation_analyzer import report_pairs
from ..utils.column_rules import looks_like_id
from ..utils.formatting import format_number, format_pct
from ..utils.json_safe import to_json_safe
from ..utils.summary_builder import SummaryBuilder

logger = logging.getLogger(__name__)

TOOL_NAME = "visualization_analyzer"
CURVE_BANDS = 10
MISSING_TOGETHER_MAX_COLUMNS = 15
# Day to year: the first that gives few enough points is used
TIME_PERIODS = [("D", "day"), ("W", "week"), ("MS", "month"), ("QS", "quarter"), ("YS", "year")]


# --- reading columns ---

def _numbers(series: pd.Series) -> np.ndarray:
    """Finite floats with NaN for gaps (and for anything that isn't a plain number)."""
    if (
        not pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_complex_dtype(series)
    ):
        return np.full(len(series), np.nan)
    values = np.array(series.to_numpy(dtype=float, na_value=np.nan), dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _labels(series: pd.Series) -> pd.Series:
    """Values as text labels, NaN for gaps. Numbers are written the same way as in the summaries."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) and not pd.api.types.is_complex_dtype(series):
        values = pd.Series(_numbers(series), index=series.index)
        return values.map(lambda v: format_number(float(v)), na_action="ignore")
    has_value = series.notna()
    return series.astype(str).where(has_value)


def _top_with_other(labels: pd.Series) -> pd.Series:
    """Keep the CHART_MAX_BARS most common values; group the rest as 'other (N values)'."""
    counts = labels.value_counts()
    if counts.size <= CHART_MAX_BARS:
        return labels
    keep = set(counts.index[:CHART_MAX_BARS - 1])
    other = f"other ({counts.size - len(keep):,} values)"
    return labels.where(labels.isna() | labels.isin(keep), other)


def _figure(traces: List[Any], title: str, **layout: Any) -> Dict[str, Any]:
    """A Plotly figure as plain JSON. The frontend applies the styling, so no template is embedded."""
    fig = go.Figure(traces)
    fig.update_layout(template="none", title=title, **layout)
    return json.loads(fig.to_json())


def _chart(chart_id: str, kind: str, title: str, caption: str, columns: List[str], figure: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    return {"id": chart_id, "kind": kind, "title": title, "caption": caption, "columns": columns, "figure": figure, **extra}


# --- overview charts ---

def _missing_chart(missing: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    columns = [c for c in missing.get("columns", []) if c["missing_pct"] + c["placeholder_pct"] > 0]
    if not columns:
        return None
    columns = sorted(columns, key=lambda c: -(c["missing_pct"] + c["placeholder_pct"]))[:CHART_MAX_BARS]
    names = [c["name"] for c in columns]
    traces = [go.Bar(name="missing", x=names, y=[round(c["missing_pct"], 4) for c in columns])]
    if any(c["placeholder_pct"] for c in columns):
        traces.append(go.Bar(name="possible placeholders ('?', 'unknown'...)", x=names, y=[round(c["placeholder_pct"], 4) for c in columns]))
    top = columns[0]
    caption = (
        f"{len(columns):,} column(s) with gaps shown; most affected: {top['name']} "
        f"({format_pct(top['missing_pct'] + top['placeholder_pct'])})"
    )
    figure = _figure(traces, "Missing values per column", barmode="stack", yaxis_title="% of rows")
    return _chart("missing_values", "bar", "Missing values per column (%)", caption, names, figure)


def _missing_together_chart(missing: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    pairs = missing.get("missing_together", [])
    if not pairs:
        return None
    counts = Counter(name for pair in pairs for name in pair["columns"])
    names = [name for name, _ in counts.most_common(MISSING_TOGETHER_MAX_COLUMNS)]
    index = {name: i for i, name in enumerate(names)}
    grid = [[None] * len(names) for _ in names]
    for pair in pairs:
        a, b = pair["columns"]
        if a in index and b in index:
            grid[index[a]][index[b]] = grid[index[b]][index[a]] = round(pair["overlap"], 4)
    first = pairs[0]
    caption = f"strongest: {first['columns'][0]} & {first['columns'][1]}, {first['overlap']:.0%} of rows missing either are missing both"
    figure = _figure([go.Heatmap(z=grid, x=names, y=names, zmin=0, zmax=1, colorscale="Blues")], "Columns missing in the same rows")
    return _chart("missing_together", "heatmap", "Columns missing in the same rows (overlap 0 to 1)", caption, names, figure)


def _correlation_heatmap(correlation: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    grid = correlation.get("number_grid", {})
    names = grid.get("columns", [])
    if len(names) < 2:
        return None
    z = [[None if v is None else round(v, 4) for v in row] for row in grid["pearson"]]
    figure = _figure([go.Heatmap(z=z, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu")], "Correlation between number columns (Pearson)")
    caption = f"{len(names)} number columns; blue = rise together, red = one rises as the other falls; blank = too few shared rows"
    return _chart("heatmap", "heatmap", f"Correlation heatmap of {len(names)} number columns (Pearson)", caption, names, figure)


def _balance_charts(df: pd.DataFrame, target_column: Optional[str], flags: Sequence[ColumnFlag]) -> List[Dict[str, Any]]:
    """Class balance of the target, or of columns the planner flagged as imbalanced when no target is set."""
    if target_column is not None:
        positions = [i for i in range(df.shape[1]) if df.columns[i] == target_column][:1]
    else:
        flagged = [f.column for f in flags if f.kind == "class_imbalance"]
        positions = [i for i in range(df.shape[1]) if str(df.columns[i]) in flagged][:CHART_MAX_DATE_COLUMNS]
    charts = []
    for position in positions:
        name = str(df.columns[position])
        counts = _labels(df.iloc[:, position]).value_counts()
        if counts.size < 2 or counts.size > CHART_MAX_BARS:
            continue
        total = int(counts.sum())
        smallest = counts.index[-1]
        caption = f"smallest group '{smallest}': {format_pct(100 * counts.iloc[-1] / total)} ({int(counts.iloc[-1]):,} of {total:,} rows)"
        figure = _figure([go.Bar(x=[str(v) for v in counts.index], y=counts.tolist())], f"Class balance of {name}", yaxis_title="rows")
        charts.append(_chart(f"balance_{len(charts) + 1}", "bar", f"Class balance of {name}", caption, [name], figure))
    return charts


# --- relationship charts ---

def _flagged_rows(outliers: Mapping[str, Any], positions: Sequence[int]) -> np.ndarray:
    """0-based row positions the outlier tool flagged in any of these columns."""
    rows = [row - 1 for c in outliers.get("columns", []) if c["position"] in positions for row in c["rows"]]
    return np.unique(np.array(rows, dtype=int))


def _scatter(df: pd.DataFrame, pair: Dict[str, Any], outliers: Mapping[str, Any]) -> Dict[str, Any]:
    (a, b), (i, j) = pair["columns"], pair["positions"]
    x, y = _numbers(df.iloc[:, i]), _numbers(df.iloc[:, j])
    shared = np.flatnonzero(~np.isnan(x) & ~np.isnan(y))
    sampled = None
    shown = shared
    if shared.size > CHART_SCATTER_SAMPLE:
        shown = np.sort(np.random.default_rng(0).choice(shared, CHART_SCATTER_SAMPLE, replace=False))
        sampled = {"shown": int(CHART_SCATTER_SAMPLE), "of": int(shared.size)}
    traces = [go.Scattergl(x=x[shown].tolist(), y=y[shown].tolist(), mode="markers", name="rows", marker={"size": 4, "opacity": 0.5})]

    flagged = np.intersect1d(_flagged_rows(outliers, (i, j)), shared)
    if flagged.size:
        traces.append(go.Scattergl(x=x[flagged].tolist(), y=y[flagged].tolist(), mode="markers", name="flagged outliers", marker={"size": 8, "symbol": "x"}))

    # Curves: average of y in 10 equal-sized bands of x, so the U-shape is visible
    if pair.get("curve"):
        bands = pd.qcut(pd.Series(x[shared]), CURVE_BANDS, labels=False, duplicates="drop")
        means = pd.DataFrame({"x": x[shared], "y": y[shared], "band": bands}).groupby("band")[["x", "y"]].mean()
        traces.append(go.Scatter(x=means["x"].round(6).tolist(), y=means["y"].round(6).tolist(), mode="lines+markers", name="average per band"))

    caption = f"Pearson {pair['pearson']:.2f}, Spearman {pair['spearman']:.2f}"
    if pair.get("curve"):
        caption += f", banded {pair['banded']:.2f}: a curve"
    elif pair.get("note"):
        caption += f"; {pair['note']}"
    if sampled:
        caption += f"; random sample of {sampled['shown']:,} of {sampled['of']:,} rows"
    if flagged.size:
        caption += f"; {flagged.size:,} flagged outlier row(s) marked"
    figure = _figure(traces, f"{b} vs {a}", xaxis_title=a, yaxis_title=b)
    return _chart("", "scatter", f"{b} vs {a} (scatter)", caption, [a, b], figure, sampled=sampled)


def _group_averages(df: pd.DataFrame, pair: Dict[str, Any]) -> Dict[str, Any]:
    """Average of the number column per group (category or two-value column)."""
    (pos_a, pos_b), kinds = pair["positions"], pair["kinds"]
    number_pos, group_pos = (pos_a, pos_b) if kinds[0] == "number" else (pos_b, pos_a)
    number, group = str(df.columns[number_pos]), str(df.columns[group_pos])
    frame = pd.DataFrame({"value": _numbers(df.iloc[:, number_pos]), "group": _top_with_other(_labels(df.iloc[:, group_pos]))}).dropna()
    stats = frame.groupby("group")["value"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    figure = _figure(
        [go.Bar(x=stats.index.tolist(), y=stats["mean"].round(6).tolist(), customdata=stats["count"].tolist(),
                hovertemplate="%{x}: average %{y} (%{customdata} rows)<extra></extra>")],
        f"Average {number} per {group}", xaxis_title=group, yaxis_title=f"average {number}",
    )
    caption = f"{pair['value']:.2f} ({pair['measure'].replace('_', '-')}); {pair['detail']}"
    return _chart("", "bar", f"Average {number} per {group}", caption, [group, number], figure)


def _group_shares(df: pd.DataFrame, pair: Dict[str, Any]) -> Dict[str, Any]:
    """For each value of the first column, the % of rows with each value of the second (100% stacked bars)."""
    (a, b), (i, j) = pair["columns"], pair["positions"]
    frame = pd.DataFrame({"a": _top_with_other(_labels(df.iloc[:, i])), "b": _top_with_other(_labels(df.iloc[:, j]))}).dropna()
    shares = pd.crosstab(frame["a"], frame["b"], normalize="index") * 100
    traces = [go.Bar(name=str(value), x=shares.index.tolist(), y=shares[value].round(4).tolist()) for value in shares.columns]
    figure = _figure(traces, f"{b} by {a}", barmode="stack", xaxis_title=a, yaxis_title=f"% of rows ({b})")
    caption = f"{pair['value']:.2f} ({'phi' if pair['measure'] == 'phi' else 'Cramer' + chr(39) + 's V'})"
    if pair.get("detail"):
        caption += f"; {pair['detail']}"
    return _chart("", "bar", f"{b} by {a} (% of rows)", caption, [a, b], figure)


def _relationship_chart(df: pd.DataFrame, pair: Dict[str, Any], outliers: Mapping[str, Any]) -> Dict[str, Any]:
    if pair["measure"] == "pearson_spearman":
        chart = _scatter(df, pair, outliers)
    elif pair["measure"] in ("eta", "point_biserial"):
        chart = _group_averages(df, pair)
    else:
        chart = _group_shares(df, pair)
    if pair["near_duplicate"]:
        chart["caption"] += "; near-duplicate" + (" involving the target: possible leakage" if pair["involves_target"] else "")
    return chart


# --- outlier charts ---

def _box_chart(df: pd.DataFrame, column: Dict[str, Any]) -> Dict[str, Any]:
    """Box from the same values and normal range the outlier tool used, with its flagged values as points."""
    name = column["name"]
    values = _numbers(df.iloc[:, column["position"]])
    if column["dominant"] is not None:
        values[values == column["dominant"]["value"]] = np.nan
    checked = values[~np.isnan(values)]
    q1, median, q3 = np.percentile(checked, [25, 50, 75])
    inside = checked[(checked >= column["lower"]) & (checked <= column["upper"])]
    flagged = values[np.array(column["rows"], dtype=int) - 1]
    traces = [
        go.Box(name=name, q1=[q1], median=[median], q3=[q3], lowerfence=[inside.min()], upperfence=[inside.max()], boxpoints=False),
        go.Scatter(name="flagged", x=[name] * flagged.size, y=flagged.tolist(), mode="markers", marker={"symbol": "x"}),
    ]
    layout = {"yaxis_type": "log"} if column["log_scale"] else {}
    caption = (
        f"{column['outlier_count']:,} outlier(s) ({format_pct(column['outlier_pct'])}), normal range "
        f"{format_number(column['lower'])} to {format_number(column['upper'])}"
        + (", log scale" if column["log_scale"] else "")
        + (f"; {column['dominant']['share']:.0%} of rows are {format_number(column['dominant']['value'])} and not shown" if column["dominant"] else "")
    )
    return _chart("", "box", f"Outliers in {name} (box plot)", caption, [name], _figure(traces, f"Outliers in {name}", **layout))


# --- time charts ---

def _time_charts(df: pd.DataFrame, flags: Sequence[ColumnFlag], target_column: Optional[str], dayfirst: Optional[bool]) -> List[Dict[str, Any]]:
    flagged = {f.column for f in flags if f.kind == "date_column"}
    positions = [
        i for i in range(df.shape[1])
        if pd.api.types.is_datetime64_any_dtype(df.iloc[:, i]) or str(df.columns[i]) in flagged
    ][:CHART_MAX_DATE_COLUMNS]
    target = next((i for i in range(df.shape[1]) if target_column is not None and df.columns[i] == target_column), None)

    charts = []
    for position in positions:
        name = str(df.columns[position])
        dates = parse_date_column(df.iloc[:, position], dayfirst)
        if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_localize(None)
        present = dates.dropna()
        if present.nunique() < 2:
            continue
        span_days = (present.max() - present.min()).days
        code, period = next(
            ((c, p) for c, p in TIME_PERIODS if span_days / {"D": 1, "W": 7, "MS": 30.4, "QS": 91.3, "YS": 365.25}[c] <= CHART_MAX_TIME_POINTS),
            TIME_PERIODS[-1],
        )
        frame = pd.DataFrame({"date": dates}).dropna()
        counts = frame.groupby(pd.Grouper(key="date", freq=code)).size()
        x = [d.strftime("%Y-%m-%d") for d in counts.index]
        caption = f"{len(present):,} dated rows from {present.min().date()} to {present.max().date()}, counted per {period}"
        figure = _figure([go.Scatter(x=x, y=counts.tolist(), mode="lines+markers")], f"Rows per {period} ({name})", yaxis_title="rows")
        charts.append(_chart("", "line", f"Rows per {period} over {name}", caption, [name], figure))

        # The target over time: its average (numbers) or the share of its less common value (two values)
        if target is not None and target != position:
            target_name = str(df.columns[target])
            numbers = _numbers(df.iloc[:, target])
            labels = _labels(df.iloc[:, target])
            value_counts = labels.value_counts()
            if value_counts.size == 2:
                event = value_counts.index[-1]
                series = 100 * (labels == event).astype(float).where(labels.notna())
                what = f"% of rows where {target_name} is '{event}'"
            elif value_counts.size > 2 and not np.all(np.isnan(numbers)):
                series, what = pd.Series(numbers, index=df.index), f"average {target_name}"
            else:
                continue
            by_period = pd.DataFrame({"date": dates, "v": series}).dropna().groupby(pd.Grouper(key="date", freq=code))["v"].mean()
            figure = _figure([go.Scatter(x=[d.strftime("%Y-%m-%d") for d in by_period.index], y=by_period.round(6).tolist(), mode="lines+markers")],
                             f"{what} per {period}", yaxis_title=what)
            charts.append(_chart("", "line", f"{what} per {period} over {name}", f"target over time, per {period}", [name, target_name], figure))
    return charts


# --- distribution charts ---

def _distribution_chart(df: pd.DataFrame, position: int, log_scale: bool) -> Optional[Dict[str, Any]]:
    series, name = df.iloc[:, position], str(df.columns[position])
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        return None
    values = _numbers(series)
    present = values[~np.isnan(values)]
    if present.size and np.unique(present).size > OUTLIER_FEW_VALUES:
        if looks_like_id(np.unique(present), present.size):
            return None
        # Histogram counted by pandas/numpy: bin edges and counts only
        if log_scale and present.min() > 0:
            edges = np.geomspace(present.min(), present.max(), CHART_HIST_BINS + 1)
        else:
            edges = np.histogram_bin_edges(present, bins=CHART_HIST_BINS)
        counts, edges = np.histogram(present, bins=edges)
        centers = np.sqrt(edges[:-1] * edges[1:]) if log_scale and present.min() > 0 else (edges[:-1] + edges[1:]) / 2
        trace = go.Bar(x=centers.round(6).tolist(), y=counts.tolist(), width=np.diff(edges).round(6).tolist(),
                       customdata=np.column_stack([edges[:-1], edges[1:]]).round(6).tolist(),
                       hovertemplate="%{customdata[0]} to %{customdata[1]}: %{y} rows<extra></extra>")
        layout = {"xaxis_type": "log"} if log_scale and present.min() > 0 else {}
        caption = f"{present.size:,} values from {format_number(float(present.min()))} to {format_number(float(present.max()))}, median {format_number(float(np.median(present)))}"
        return _chart("", "histogram", f"Distribution of {name}", caption, [name], _figure([trace], f"Distribution of {name}", yaxis_title="rows", bargap=0, **layout))

    # Few values or text: how often each value appears
    labels = _labels(series)
    counts = labels.value_counts()
    if counts.empty or (counts.size > 1 and counts.iloc[0] == 1):
        return None  # empty, or every value different (IDs, free text)
    counts = _top_with_other(labels).value_counts()
    # The grouped leftovers go last, even when together they outnumber every single value
    is_other = counts.index.astype(str).str.startswith("other (")
    counts = pd.concat([counts[~is_other], counts[is_other]])
    caption = f"{labels.nunique():,} different values; most common '{counts.index[0]}' ({int(counts.iloc[0]):,} rows)"
    figure = _figure([go.Bar(x=[str(v) for v in counts.index], y=counts.tolist())], f"Most common values of {name}", yaxis_title="rows")
    return _chart("", "bar", f"Most common values of {name}", caption, [name], figure)


# --- summary ---

def _build_summary(charts: List[Dict[str, Any]], skipped: List[Dict[str, str]]) -> str:
    builder = SummaryBuilder(TOOL_SUMMARY_MAX_TOKENS)
    if not charts:
        builder.add("Charts: none could be made for this file")
        return builder.text()
    builder.add(f"Charts ({len(charts):,}) - refer to a chart by its id:")
    builder.add_items([f"- {c['id']}: {c['title']}" for c in charts], more=lambda n: f"... and {n:,} more charts (see full results)")
    if skipped:
        builder.add(f"{len(skipped):,} chart(s) could not be made (see full results)")
    return builder.text()


# --- entry point ---

def analyze_visualizations(
    df: pd.DataFrame,
    tool_payloads: Optional[Mapping[str, Mapping[str, Any]]] = None,
    target_column: Optional[str] = None,
    focus_columns: Optional[Sequence[str]] = None,
    column_flags: Optional[Sequence[ColumnFlag]] = None,
    date_dayfirst: Optional[bool] = None,
) -> ToolResult:
    """Build the report's charts. Runs after the analysis tools, whose payloads decide what is drawn.

    `tool_payloads` maps tool names to payloads of tools that succeeded. Charts, in order:
    overview (missing values, missing together, correlation heatmap), class balance, one chart per
    relationship in the correlation summary, box plots for outlier columns, time charts for date
    columns, then one distribution chart per column. All counting is done here, so each figure
    carries counts and averages instead of the raw rows.
    """
    try:
        payloads = tool_payloads or {}
        missing, correlation, outliers = (payloads.get(name, {}) for name in ("missing_value_analyzer", "correlation_analyzer", "outlier_analyzer"))
        flags = list(column_flags or [])
        charts: List[Dict[str, Any]] = []
        skipped: List[Dict[str, str]] = []

        def add(prefix: str, build: Callable[[], Optional[Dict[str, Any]]], what: str) -> None:
            if len(charts) >= CHART_MAX_TOTAL:
                return
            # One chart that can't be drawn must not stop the others
            try:
                chart = build()
            except Exception as exc:
                logger.warning("visualization_analyzer: %s failed: %s", what, exc)
                skipped.append({"chart": what, "reason": str(exc)})
                return
            if chart is not None:
                if not chart["id"]:
                    chart["id"] = f"{prefix}_{sum(1 for c in charts if c['id'].startswith(prefix + '_')) + 1}"
                charts.append(chart)

        add("missing", lambda: _missing_chart(missing), "missing values")
        add("missing", lambda: _missing_together_chart(missing), "missing together")
        add("heatmap", lambda: _correlation_heatmap(correlation), "correlation heatmap")
        for chart in _safe_list(lambda: _balance_charts(df, target_column, flags), skipped, "class balance"):
            add("balance", lambda chart=chart: chart, "class balance")
        for pair in report_pairs(correlation):
            add("rel", lambda pair=pair: _relationship_chart(df, pair, outliers), f"relationship {pair['columns'][0]} & {pair['columns'][1]}")
        box_columns = sorted((c for c in outliers.get("columns", []) if c["outlier_count"]), key=lambda c: -c["outlier_pct"])
        for column in box_columns[:CHART_MAX_BOXES]:
            add("box", lambda column=column: _box_chart(df, column), f"box plot {column['name']}")
        for chart in _safe_list(lambda: _time_charts(df, flags, target_column, date_dayfirst), skipped, "time charts"):
            add("time", lambda chart=chart: chart, "time chart")

        # Distributions: target, planner focus columns, then file order
        focus = {str(name) for name in focus_columns or []}
        order = sorted(range(df.shape[1]), key=lambda i: (
            0 if target_column is not None and df.columns[i] == target_column else 1 if str(df.columns[i]) in focus else 2, i,
        ))
        log_columns = {c["position"] for c in outliers.get("columns", []) if c["log_scale"]}
        made = 0
        for position in order:
            if made >= CHART_MAX_DISTRIBUTIONS:
                break
            before = len(charts)
            add("dist", lambda position=position: _distribution_chart(df, position, position in log_columns), f"distribution {df.columns[position]}")
            made += len(charts) - before

        payload = to_json_safe({"charts": charts, "skipped": skipped, "settings": {
            "max_total": CHART_MAX_TOTAL, "scatter_sample": CHART_SCATTER_SAMPLE, "hist_bins": CHART_HIST_BINS,
        }})
        return ToolResult(tool_name=TOOL_NAME, success=True, payload=payload, summary=_build_summary(charts, skipped))
    except Exception as exc:
        logger.exception("visualization_analyzer failed")
        return ToolResult(tool_name=TOOL_NAME, success=False, payload={}, summary="", error=str(exc))


def _safe_list(build: Callable[[], List[Dict[str, Any]]], skipped: List[Dict[str, str]], what: str) -> List[Dict[str, Any]]:
    try:
        return build()
    except Exception as exc:
        logger.warning("visualization_analyzer: %s failed: %s", what, exc)
        skipped.append({"chart": what, "reason": str(exc)})
        return []
