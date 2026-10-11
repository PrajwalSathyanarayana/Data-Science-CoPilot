"""Tool — finds unusual values in each number column (box-plot rule) and unusual combinations across columns (robust Mahalanobis)."""
import logging
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

from ..config import (
    OUTLIER_DOMINANT_SHARE,
    OUTLIER_EXTREMES_SHOWN,
    OUTLIER_FEW_VALUES,
    OUTLIER_IQR_K,
    OUTLIER_LOG_QUARTILE_SKEW,
    OUTLIER_MIN_VALUES,
    OUTLIER_RARE_PCT,
    OUTLIER_ROWS_KEPT,
    ROW_CHECK_CUTOFF,
    ROW_CHECK_FIT_ROWS,
    ROW_CHECK_MAX_COLUMNS,
    ROW_CHECK_MAX_MISSING_PCT,
    ROW_CHECK_MIN_ROWS,
    ROW_CHECK_WARN_PCT,
    TOOL_SUMMARY_MAX_TOKENS,
)
from ..models import ToolResult
from ..utils.column_rules import looks_like_id
from ..utils.formatting import format_number, format_pct
from ..utils.json_safe import to_json_safe
from ..utils.summary_builder import SummaryBuilder

logger = logging.getLogger(__name__)

TOOL_NAME = "outlier_analyzer"
MAX_SKIPPED_NAMES_SHOWN = 5
NEAR_COPY_CORRELATION = 0.99  # a column this correlated with one already used adds nothing to the row check
ROWS_PER_COLUMN = 10  # the row check needs at least this many complete rows per column used


# --- one column at a time ---

def _number_values(series: pd.Series) -> Union[np.ndarray, str]:
    """The column as floats (NaN for gaps), or the reason it isn't a number column."""
    if (
        pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_timedelta64_dtype(series)
        or isinstance(series.dtype, pd.PeriodDtype)
    ):
        return "date or time"
    if (
        not pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_complex_dtype(series)
    ):
        return "not a number column"
    return np.array(series.to_numpy(dtype=float, na_value=np.nan), dtype=float, copy=True)


def _quartile_skew(values: np.ndarray) -> float:
    """(Q3 + Q1 - 2 x median) / (Q3 - Q1): above 0 for a long right tail. Extreme values can't move it."""
    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    return float((q3 + q1 - 2 * q2) / (q3 - q1)) if q3 > q1 else 0.0


def _fences(values: np.ndarray, log_scale: bool) -> Tuple[float, float]:
    """Box-plot fences, in the column's own units even when computed on the log scale."""
    scaled = np.log(values) if log_scale else values
    q1, q3 = np.percentile(scaled, [25, 75])
    spread = OUTLIER_IQR_K * (q3 - q1)
    low, high = q1 - spread, q3 + spread
    return (float(np.exp(low)), float(np.exp(high))) if log_scale else (float(low), float(high))


def _check_column(position: int, name: str, series: pd.Series) -> Union[Dict[str, Any], str]:
    """Outlier report for one column, or the reason it isn't checked."""
    values = _number_values(series)
    if isinstance(values, str):
        return values

    infinite = int(np.isinf(values).sum())
    values[np.isinf(values)] = np.nan  # counted separately; they would break the fences
    present = ~np.isnan(values)
    finite = values[present]
    distinct = np.unique(finite)
    if distinct.size <= 2:
        return "two values or fewer"
    if looks_like_id(distinct, finite.size):
        return "looks like a row number, ID or evenly spaced sequence"

    # One value filling most of the column (e.g. mostly 0): check the other values among themselves
    dominant = None
    checked = present.copy()
    common_values, common_counts = np.unique(finite, return_counts=True)
    top = int(np.argmax(common_counts))
    share = common_counts[top] / finite.size
    if share >= OUTLIER_DOMINANT_SHARE:
        dominant = {"value": float(common_values[top]), "share": float(share)}
        checked &= values != common_values[top]
    rest = values[checked]
    if rest.size < OUTLIER_MIN_VALUES:
        return f"fewer than {OUTLIER_MIN_VALUES} values to check"

    # Few different values (ratings, small counts): the ends of the scale are real groups, not errors,
    # so only values that are outside the normal range AND rare are flagged. These stay on the normal scale.
    rest_values, rest_counts = np.unique(rest, return_counts=True)
    few_values = rest_values.size <= OUTLIER_FEW_VALUES
    # Long right tail and always positive: measure distance in multiples (log scale)
    log_scale = bool(not few_values and rest.min() > 0 and _quartile_skew(rest) > OUTLIER_LOG_QUARTILE_SKEW)
    low, high = _fences(rest, log_scale)
    if low == high and not few_values:
        return "values too concentrated to judge"

    is_low = checked & (values < low)
    is_high = checked & (values > high)
    if few_values:
        rare = rest_values[100 * rest_counts / rest.size < OUTLIER_RARE_PCT]
        is_rare = np.isin(values, rare)
        is_low &= is_rare
        is_high &= is_rare
    flagged = np.flatnonzero(is_low | is_high)
    # Most unusual first, measured as how far outside the fence (on the scale used)
    scale = np.log if log_scale else (lambda x: x)
    distance = np.where(values[flagged] > high, scale(values[flagged]) - scale(high), scale(low) - scale(values[flagged]))
    flagged = flagged[np.argsort(-distance, kind="stable")]

    return {
        "name": name,
        "position": position,
        "checked_values": int(rest.size),
        "missing": int((~present).sum()) - infinite,
        "infinite": infinite,
        "dominant": dominant,
        "few_values": few_values,
        "log_scale": log_scale,
        "lower": low,
        "upper": high,
        "low_count": int(is_low.sum()),
        "high_count": int(is_high.sum()),
        "outlier_count": int(flagged.size),
        "outlier_pct": 100 * flagged.size / rest.size,
        "lowest": np.sort(values[is_low])[:OUTLIER_EXTREMES_SHOWN].tolist(),
        "highest": np.sort(values[is_high])[::-1][:OUTLIER_EXTREMES_SHOWN].tolist(),
        "rows": (flagged[:OUTLIER_ROWS_KEPT] + 1).tolist(),  # 1 = first data row of the file
    }


# --- unusual combinations across columns ---

def _missing_pct(column: Dict[str, Any]) -> float:
    total = column["checked_values"] + column["missing"] + column["infinite"]
    return 100 * column["missing"] / total if total else 0.0


def _choose_row_columns(
    checked: List[Dict[str, Any]], target_position: Optional[int], focus: Sequence[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Columns for the row check: checked columns without a dominant value, in the usual order of preference."""
    focus_set = {str(name) for name in focus}
    candidates = [
        c for c in checked
        if c["dominant"] is None and not c["few_values"] and _missing_pct(c) <= ROW_CHECK_MAX_MISSING_PCT
    ]
    candidates.sort(key=lambda c: (
        0 if c["position"] == target_position else 1 if c["name"] in focus_set else 2, c["missing"], c["position"],
    ))
    left_out = [
        {"name": c["name"], "reason": f"one value fills {c['dominant']['share']:.0%} of the column"}
        for c in checked if c["dominant"] is not None
    ]
    left_out += [
        {"name": c["name"], "reason": f"only {OUTLIER_FEW_VALUES} or fewer different values"}
        for c in checked if c["dominant"] is None and c["few_values"]
    ]
    left_out += [
        {"name": c["name"], "reason": f"{_missing_pct(c):.0f}% missing"}
        for c in checked
        if c["dominant"] is None and not c["few_values"] and _missing_pct(c) > ROW_CHECK_MAX_MISSING_PCT
    ]
    chosen = candidates[:ROW_CHECK_MAX_COLUMNS]
    left_out += [{"name": c["name"], "reason": f"over the {ROW_CHECK_MAX_COLUMNS}-column limit"} for c in candidates[ROW_CHECK_MAX_COLUMNS:]]
    return sorted(chosen, key=lambda c: c["position"]), left_out


def _row_check(df: pd.DataFrame, chosen: List[Dict[str, Any]], left_out: List[Dict[str, str]]) -> Dict[str, Any]:
    """Robust Mahalanobis distance: how far each complete row is from the normal shape of the columns together."""
    result: Dict[str, Any] = {"status": "skipped", "reason": None, "columns": [], "left_out": left_out}
    if len(chosen) < 2:
        result["reason"] = "fewer than 2 number columns can be used"
        return result

    # Long-tailed columns go on the log scale, as in the per-column check (their values are all positive)
    prepared = []
    for c in chosen:
        values = _number_values(df.iloc[:, c["position"]])
        values[np.isinf(values)] = np.nan
        prepared.append(np.log(values) if c["log_scale"] else values)
    data = pd.DataFrame(np.column_stack(prepared))  # numbered columns, so duplicate names can't collide
    names = [c["name"] for c in chosen]

    # A near-copy of a column already used (e.g. the same price in two currencies) makes the shape impossible to learn
    keep: List[int] = []
    correlations = data.corr().to_numpy()
    for i in range(len(chosen)):
        copy_of = next((j for j in keep if abs(correlations[i, j]) >= NEAR_COPY_CORRELATION), None)
        if copy_of is None:
            keep.append(i)
        else:
            result["left_out"].append({"name": names[i], "reason": f"near-copy of {names[copy_of]}"})
    if len(keep) < 2:
        result["reason"] = "fewer than 2 number columns can be used"
        return result

    complete = data[keep].notna().all(axis=1).to_numpy()
    rows = int(complete.sum())
    needed = max(ROW_CHECK_MIN_ROWS, ROWS_PER_COLUMN * len(keep))
    result.update(columns=[names[i] for i in keep], rows_checked=rows, rows_with_gaps=int(len(data) - rows))
    if rows < needed:
        result["reason"] = f"only {rows:,} complete rows (at least {needed:,} needed for {len(keep)} columns)"
        return result

    # Robust scaling so very different units (age vs salary) don't cause numerical trouble
    X = data[keep].to_numpy()[complete]
    center = np.median(X, axis=0)
    spread = np.subtract(*np.percentile(X, [75, 25], axis=0))
    spread[spread == 0] = np.std(X, axis=0)[spread == 0]
    spread[spread == 0] = 1.0
    X = (X - center) / spread

    # Learn the normal shape from a fixed-seed sample (documented exception: every row is still checked)
    rng = np.random.default_rng(0)
    fit_rows = X if len(X) <= ROW_CHECK_FIT_ROWS else X[rng.choice(len(X), ROW_CHECK_FIT_ROWS, replace=False)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MinCovDet(random_state=0).fit(fit_rows)
            distances = model.mahalanobis(X)
            inverse = np.linalg.pinv(model.covariance_)
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("outlier_analyzer: row check could not learn the normal shape: %s", exc)
        result["reason"] = "the columns are too closely related to learn a normal shape"
        return result

    cutoff = float(chi2.ppf(ROW_CHECK_CUTOFF, len(keep)))
    unusual = np.flatnonzero(distances > cutoff)
    unusual = unusual[np.argsort(-distances[unusual], kind="stable")]
    # Which column added most to each unusual row's distance
    diff = X[unusual] - model.location_
    main = np.argmax(diff * (diff @ inverse), axis=1) if unusual.size else np.array([], dtype=int)
    row_numbers = np.flatnonzero(complete)[unusual] + 1  # 1 = first data row of the file
    used = [names[i] for i in keep]

    unusual_pct = 100 * unusual.size / rows
    result.update(
        status="done",
        warning=unusual_pct > ROW_CHECK_WARN_PCT,
        fit_rows=int(len(fit_rows)),
        cutoff=cutoff,
        unusual_count=int(unusual.size),
        unusual_pct=unusual_pct,
        main_columns=dict(Counter(used[i] for i in main).most_common()),
        rows=[
            {"row": int(r), "distance": float(np.sqrt(distances[u])), "main_column": used[m]}
            for r, u, m in list(zip(row_numbers, unusual, main))[:OUTLIER_ROWS_KEPT]
        ],
    )
    return result


# --- summary ---

def _column_line(column: Dict[str, Any]) -> str:
    parts = []
    if column["dominant"] is not None:
        d = column["dominant"]
        parts.append(f"{d['share']:.0%} are {format_number(d['value'])}; checked among the other {column['checked_values']:,} values")
    counts = []
    if column["high_count"]:
        counts.append(f"{column['high_count']:,} high")
    if column["low_count"]:
        counts.append(f"{column['low_count']:,} low")
    if counts:
        scale = ", log scale" if column["log_scale"] else ""
        if column["few_values"]:
            scale += f", few different values: only values under {OUTLIER_RARE_PCT:g}% of rows flagged"
        parts.append(
            f"{' and '.join(counts)} ({format_pct(column['outlier_pct'])}), "
            f"normal range {format_number(column['lower'])} to {format_number(column['upper'])}{scale}"
        )
    if column["highest"]:
        parts.append("highest: " + ", ".join(format_number(v) for v in column["highest"]))
    if column["lowest"]:
        parts.append("lowest: " + ", ".join(format_number(v) for v in column["lowest"]))
    if column["infinite"]:
        parts.append(f"{column['infinite']:,} infinite values")
    return f"- {column['name']}: " + ", ".join(parts)


def _row_lines(row_check: Dict[str, Any]) -> List[str]:
    if row_check["status"] != "done":
        return [f"Unusual combinations: not checked - {row_check['reason']}"]
    columns = row_check["columns"]
    shown = ", ".join(columns[:MAX_SKIPPED_NAMES_SHOWN]) + (f", and {len(columns) - MAX_SKIPPED_NAMES_SHOWN} more" if len(columns) > MAX_SKIPPED_NAMES_SHOWN else "")
    line = (
        f"Unusual combinations (robust Mahalanobis on {len(columns)} number columns: {shown}): "
        f"{row_check['unusual_count']:,} of {row_check['rows_checked']:,} complete rows ({format_pct(row_check['unusual_pct'])})"
    )
    if row_check["main_columns"]:
        drivers = ", ".join(f"{name} ({count:,})" for name, count in list(row_check["main_columns"].items())[:3])
        line += f"; column adding most to each: {drivers}"
    lines = [line]
    if row_check["warning"]:
        lines.append(
            "Caution: far more unusual rows than the ~0.1% expected; these columns probably follow a curve or "
            "separate groups, which this check can't model, so treat these rows as uncertain"
        )
    if row_check["rows_with_gaps"]:
        lines.append(f"{row_check['rows_with_gaps']:,} rows with gaps in these columns were not checked for unusual combinations")
    return lines


def _build_summary(payload: Dict[str, Any]) -> str:
    builder = SummaryBuilder(TOOL_SUMMARY_MAX_TOKENS)
    checked = payload["columns"]
    with_outliers = [c for c in checked if c["outlier_count"] or c["infinite"]]
    builder.add(
        f"Outliers (box-plot rule, {OUTLIER_IQR_K:g} x IQR; long-tailed columns on the log scale): "
        f"{len(with_outliers):,} of {len(checked):,} checked columns have outliers"
    )
    for line in _row_lines(payload["row_check"]):
        builder.add(line)

    with_outliers.sort(key=lambda c: (-c["outlier_pct"], -c["infinite"]))
    builder.add_items(
        [_column_line(c) for c in with_outliers],
        more=lambda n: f"... and {n:,} more columns with outliers (see full results)",
    )
    mostly_one = [c for c in checked if c["dominant"] is not None and not c["outlier_count"]]
    if mostly_one:
        builder.add_items(
            [_column_line(c) + ", no outliers" for c in mostly_one],
            more=lambda n: f"... and {n:,} more columns mostly filled by one value (see full results)",
        )

    by_reason: Dict[str, List[str]] = {}
    for column in payload["skipped_columns"]:
        by_reason.setdefault(column["reason"], []).append(column["name"])
    lines = []
    for reason, names in by_reason.items():
        extra = f", and {len(names) - MAX_SKIPPED_NAMES_SHOWN:,} more" if len(names) > MAX_SKIPPED_NAMES_SHOWN else ""
        lines.append(f"- {reason}: {', '.join(names[:MAX_SKIPPED_NAMES_SHOWN])}{extra}")
    builder.add_items(lines, more=lambda n: f"... and {n:,} more reasons (see full results)", header="Columns not checked:")
    return builder.text()


# --- entry point ---

def analyze_outliers(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    focus_columns: Optional[Sequence[str]] = None,
) -> ToolResult:
    """Find unusual values per number column, and unusual combinations of values across columns.

    Per column: the box-plot rule (outside Q1 - 1.5 x IQR or Q3 + 1.5 x IQR), on the log scale for
    long-tailed positive columns. Columns mostly filled by one value are checked among their other
    values; columns with few different values, IDs, two-value, text and date columns are not checked.
    Across columns: robust Mahalanobis distance, learned from up to ROW_CHECK_FIT_ROWS random rows
    and applied to every complete row. `target_column` and `focus_columns` are preferred when the
    row check has to choose columns.
    """
    try:
        target_position = None
        if target_column is not None:
            matches = [i for i in range(df.shape[1]) if df.columns[i] == target_column]
            if not matches:
                raise ValueError(f"target_column '{target_column}' is not a column in the dataset")
            target_position = matches[0]

        checked, skipped = [], []
        for position in range(df.shape[1]):
            # By position, so duplicate column names can't confuse lookups
            name, series = str(df.columns[position]), df.iloc[:, position]
            try:
                report = _check_column(position, name, series)
            except Exception as exc:
                logger.warning("outlier_analyzer: could not check column '%s': %s", name, exc)
                report = "values could not be checked"
            if isinstance(report, str):
                skipped.append({"name": name, "position": position, "reason": report})
            else:
                checked.append(report)

        chosen, left_out = _choose_row_columns(checked, target_position, focus_columns or [])
        try:
            row_check = _row_check(df, chosen, left_out)
        except Exception as exc:
            logger.exception("outlier_analyzer: row check failed")
            row_check = {"status": "skipped", "reason": f"could not be computed ({exc})", "columns": [], "left_out": left_out}

        payload = to_json_safe({
            "total_rows": len(df),
            "total_columns": df.shape[1],
            "columns": checked,
            "skipped_columns": skipped,
            "row_check": row_check,
            "settings": {
                "iqr_k": OUTLIER_IQR_K, "few_values": OUTLIER_FEW_VALUES, "dominant_share": OUTLIER_DOMINANT_SHARE,
                "row_check_max_columns": ROW_CHECK_MAX_COLUMNS, "row_check_fit_rows": ROW_CHECK_FIT_ROWS,
                "row_check_cutoff": ROW_CHECK_CUTOFF,
            },
        })
        return ToolResult(tool_name=TOOL_NAME, success=True, payload=payload, summary=_build_summary(payload))
    except Exception as exc:
        logger.exception("outlier_analyzer failed")
        return ToolResult(tool_name=TOOL_NAME, success=False, payload={}, summary="", error=str(exc))
