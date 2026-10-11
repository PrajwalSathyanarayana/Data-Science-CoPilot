"""Tool — measures how strongly pairs of columns are related, using the right measure for each pair of column kinds."""
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..config import (
    CORR_BANDS,
    CORR_CURVE_MARGIN,
    CORR_CURVE_MIN,
    CORR_MAX_CATEGORIES,
    CORR_METHOD_GAP,
    CORR_MIN_ROWS,
    CORR_MIN_STRENGTH,
    CORR_NEAR_PERFECT,
    CORR_SUMMARY_TOP,
    PAIR_CHECK_MAX_COLUMNS,
    TOOL_SUMMARY_MAX_TOKENS,
)
from ..models import ToolResult
from ..utils.column_rules import looks_like_id
from ..utils.formatting import format_number
from ..utils.json_safe import to_json_safe
from ..utils.summary_builder import SummaryBuilder

logger = logging.getLogger(__name__)

TOOL_NAME = "correlation_analyzer"
NUMBER, TWO_VALUE, CATEGORY = "number", "two-value", "category"
MAX_SKIPPED_NAMES_SHOWN = 5
# Pairs with gaps get an exact Spearman when they're strong enough to be reported (see _PairMeasurer)
EXACT_SPEARMAN_FROM = CORR_MIN_STRENGTH - 0.1
EXACT_SPEARMAN_MAX_PAIRS = 200

# Which measure each pair of column kinds gets (order of the two kinds doesn't matter)
MEASURES = {
    frozenset([NUMBER]): "pearson_spearman",
    frozenset([NUMBER, TWO_VALUE]): "point_biserial",
    frozenset([TWO_VALUE]): "phi",
    frozenset([NUMBER, CATEGORY]): "eta",
    frozenset([CATEGORY]): "cramers_v",
    frozenset([TWO_VALUE, CATEGORY]): "cramers_v",
}
MEASURE_LABELS = {"point_biserial": "point-biserial", "phi": "phi", "eta": "eta", "cramers_v": "Cramer's V"}


@dataclass
class _Column:
    """A column prepared for measuring.

    values: numbers -> floats (NaN for gaps); two-value -> 1.0 / 0.0 (NaN for gaps); category -> int codes (-1 for gaps)
    labels: two-value -> [value coded 0, value coded 1]; category -> the value for each code
    """

    position: int
    name: str
    kind: str
    values: np.ndarray
    missing: int
    labels: List[str] = field(default_factory=list)

    def codes(self) -> np.ndarray:
        """Group codes (-1 for gaps), so two-value columns can be compared as categories too."""
        if self.kind == CATEGORY:
            return self.values
        return np.where(np.isnan(self.values), -1, self.values).astype(int)

    def present(self) -> np.ndarray:
        return self.values >= 0 if self.kind == CATEGORY else ~np.isnan(self.values)


# --- preparing columns ---

def _label_column(position: int, name: str, labels: pd.Series) -> Union[_Column, str]:
    """Two-value and category columns, from text labels (missing values stay missing)."""
    codes, uniques = pd.factorize(labels, sort=True)  # gaps get -1
    missing = int(np.count_nonzero(codes == -1))
    if len(uniques) == 0:
        return "no values"
    if len(uniques) == 1:
        return "only one value"
    if len(uniques) == 2:
        # The less common value is coded 1 (e.g. 'yes' in 'returned'), so a positive score reads naturally
        counts = np.bincount(codes[codes >= 0], minlength=2)
        event = 1 if counts[1] <= counts[0] else 0
        coded = np.where(codes == -1, np.nan, (codes == event).astype(float))
        return _Column(position, name, TWO_VALUE, coded, missing, [str(uniques[1 - event]), str(uniques[event])])
    if len(uniques) > CORR_MAX_CATEGORIES:
        return (
            f"too many different values to compare as categories ({len(uniques):,}): "
            "e.g. IDs, free text, dates stored as text"
        )
    return _Column(position, name, CATEGORY, codes, missing, [str(u) for u in uniques])


def _prepare_column(position: int, name: str, series: pd.Series) -> Union[_Column, str]:
    """Decide how a column can be measured. Returns the prepared column, or the reason it can't be measured."""
    if (
        pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_timedelta64_dtype(series)
        or isinstance(series.dtype, pd.PeriodDtype)
    ):
        return "date or time (trends over time are a separate analysis)"

    is_number = (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
        and not pd.api.types.is_complex_dtype(series)
    )
    if is_number:
        # A copy: pandas may hand back read-only data, and infinities are overwritten below
        values = np.array(series.to_numpy(dtype=float, na_value=np.nan), dtype=float, copy=True)
        values[~np.isfinite(values)] = np.nan  # infinity can't be measured, so it's treated as a gap
        present = values[~np.isnan(values)]
        distinct = np.unique(present)
        if distinct.size == 2:
            # A 0/1 (or any two-number) column is compared as two groups, which keeps the direction readable
            names = [format_number(float(d)) for d in distinct]
            if names[0] == names[1]:
                names = [repr(float(d)) for d in distinct]
            labels = np.where(np.isnan(values), None, np.where(values == distinct[1], names[1], names[0]))
            return _label_column(position, name, pd.Series(labels, dtype=object))
        if distinct.size == 0:
            return "no values"
        if distinct.size == 1:
            return "only one value"
        if looks_like_id(distinct, present.size):
            return "looks like a row number, ID or evenly spaced sequence"
        return _Column(position, name, NUMBER, values, int(np.isnan(values).sum()))

    # Booleans, text, and anything else: compared by their values as text
    has_value = series.notna().to_numpy()
    labels = np.where(has_value, series.astype(str).to_numpy(), None)
    return _label_column(position, name, pd.Series(labels, dtype=object))


def _choose_columns(columns: List[_Column], target_position: Optional[int], focus: Sequence[str]) -> List[_Column]:
    """Order of preference when there are too many columns: target, planner's focus columns, fewest gaps, file order."""
    focus_set = {str(name) for name in focus}

    def preference(column: _Column):
        rank = 0 if column.position == target_position else 1 if column.name in focus_set else 2
        return rank, column.missing, column.position

    return sorted(columns, key=preference)


# --- measures ---

def _banded_score(bands: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Eta, adjusted for chance: how much of y's variation is explained by which group/band each row is in (0 to 1).

    Raw eta always comes out above 0 by chance, more so with many groups and few rows, so it's adjusted the same
    way as adjusted R-squared. Returns NaN when y doesn't vary or there's only one group.
    """
    groups, values = bands[mask], y[mask]
    n = values.size
    counts = np.bincount(groups)
    used = counts > 0
    k = int(used.sum())
    mean = values.mean()
    total = float(((values - mean) ** 2).sum())
    if total == 0 or k < 2 or n <= k:
        return math.nan
    group_means = np.bincount(groups, values)[used] / counts[used]
    eta_squared = float((counts[used] * (group_means - mean) ** 2).sum()) / total
    adjusted = 1 - (1 - eta_squared) * (n - 1) / (n - k)
    return math.sqrt(min(1.0, max(0.0, adjusted)))


def _cramers_v(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Cramer's V with the small-sample correction (Bergsma 2013), 0 to 1. NaN if either column has one value."""
    a, b = a[mask], b[mask]
    n = a.size
    table = np.bincount(a * (b.max() + 1) + b, minlength=(a.max() + 1) * (b.max() + 1)).reshape(a.max() + 1, b.max() + 1)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    rows, cols = table.shape
    if rows < 2 or cols < 2 or n < 2:
        return math.nan
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    phi2 = max(0.0, chi2 / n - (rows - 1) * (cols - 1) / (n - 1))
    rows_c = rows - (rows - 1) ** 2 / (n - 1)
    cols_c = cols - (cols - 1) ** 2 / (n - 1)
    denominator = min(rows_c - 1, cols_c - 1)
    if denominator <= 0:
        return math.nan
    return math.sqrt(min(1.0, phi2 / denominator))


def _bands(column: _Column) -> np.ndarray:
    """Split a number column into up to CORR_BANDS equal-sized bands (-1 for gaps)."""
    bands = pd.qcut(pd.Series(column.values), CORR_BANDS, labels=False, duplicates="drop")
    return bands.fillna(-1).to_numpy().astype(int)


def _sign_text(value: float) -> str:
    return "rise together" if value > 0 else "as one rises, the other falls"


def _describe_number_pair(pair: Dict[str, Any]) -> None:
    """Fill in the strength, direction, shape note and curve flag of a number-vs-number pair."""
    p, s, banded = pair["pearson"], pair["spearman"], pair["banded"]
    strongest = p if abs(p) >= abs(s) else s
    pair.update(value=strongest, strength=abs(strongest), detail=_sign_text(strongest), note=_shape_note(p, s))
    pair["curve"] = bool(
        banded is not None and banded >= CORR_CURVE_MIN and banded - max(abs(p), abs(s)) >= CORR_CURVE_MARGIN
    )


def _shape_note(pearson: float, spearman: float) -> Optional[str]:
    if abs(spearman) - abs(pearson) >= CORR_METHOD_GAP:
        return "consistent order but not a straight line"
    if abs(pearson) - abs(spearman) >= CORR_METHOD_GAP:
        return "may be driven by a few extreme values"
    return None


# --- measuring every pair ---

class _PairMeasurer:
    """Computes whole-matrix results once (Pearson, ranks, shared rows), then measures each pair."""

    def __init__(self, columns: List[_Column]) -> None:
        self.columns = columns
        self.index = {id(c): i for i, c in enumerate(columns)}
        present = np.column_stack([c.present() for c in columns]).astype(np.float32)
        self.shared = np.rint(present.T @ present).astype(int)  # rows where both columns have values

        # Pearson over numbers and 0/1 columns covers Pearson, point-biserial and phi in one pass
        self.linear = [c for c in columns if c.kind in (NUMBER, TWO_VALUE)]
        self.linear_index = {id(c): i for i, c in enumerate(self.linear)}
        if self.linear:
            frame = pd.DataFrame(np.column_stack([c.values for c in self.linear]))
            self.pearson = frame.corr().to_numpy()
        # Spearman = Pearson on ranks. For speed every column is ranked once, over all its values: pandas'
        # pairwise version re-ranks each pair's shared rows and took 49.7 s instead of 2.8 s (75 columns x
        # 75,000 rows, 10% gaps). This is exact when neither column has gaps; with gaps it can be off by ~0.01,
        # so pairs strong enough to be reported are recalculated exactly (see _measure_pairs).
        numbers = [c for c in columns if c.kind == NUMBER]
        self.number_index = {id(c): i for i, c in enumerate(numbers)}
        if numbers:
            ranks = pd.DataFrame(np.column_stack([c.values for c in numbers])).rank()
            self.spearman = ranks.corr().to_numpy()
            self.bands = {id(c): _bands(c) for c in numbers}
        self.numbers = numbers
        self.present_count = {id(c): int(c.present().sum()) for c in columns}

    def shared_rows(self, a: _Column, b: _Column) -> int:
        return int(self.shared[self.index[id(a)], self.index[id(b)]])

    def exact_spearman(self, a: _Column, b: _Column) -> float:
        """Spearman with ranks recalculated on just the rows where both columns have values."""
        mask = a.present() & b.present()
        return float(pd.Series(a.values[mask]).rank().corr(pd.Series(b.values[mask]).rank()))

    def measure(self, a: _Column, b: _Column) -> Optional[Dict[str, Any]]:
        """Measure one pair. Returns None when a column has no variation within the shared rows."""
        measure = MEASURES[frozenset([a.kind, b.kind])]
        pair: Dict[str, Any] = {"measure": measure}

        if measure == "pearson_spearman":
            i, j = self.number_index[id(a)], self.number_index[id(b)]
            p = float(self.pearson[self.linear_index[id(a)], self.linear_index[id(b)]])
            s = float(self.spearman[i, j])
            if math.isnan(p) or math.isnan(s):
                return None
            # Curve check in both directions (bands of a explaining b, and bands of b explaining a)
            mask = a.present() & b.present()
            scores = [
                _banded_score(self.bands[id(x)], y.values, mask)
                for x, y in ((a, b), (b, a))
            ]
            scores = [score for score in scores if not math.isnan(score)]
            # Shared rows = all of both columns' values means the per-column ranks are already exact
            rows = self.shared_rows(a, b)
            exact = rows == self.present_count[id(a)] == self.present_count[id(b)]
            pair.update(pearson=p, spearman=s, spearman_exact=exact, banded=max(scores) if scores else None)
            _describe_number_pair(pair)
            return pair

        if measure in ("point_biserial", "phi"):
            r = float(self.pearson[self.linear_index[id(a)], self.linear_index[id(b)]])
            if math.isnan(r):
                return None
            if measure == "point_biserial":
                number, group = (a, b) if a.kind == NUMBER else (b, a)
                detail = f"{number.name} is higher when {group.name} is '{group.labels[1 if r > 0 else 0]}'"
            else:
                detail = f"'{b.labels[1]}' in {b.name} is more common when {a.name} is '{a.labels[1 if r > 0 else 0]}'"
            pair.update(value=r, strength=abs(r), detail=detail)
            return pair

        mask = a.present() & b.present()
        if measure == "eta":
            number, group = (a, b) if a.kind == NUMBER else (b, a)
            score = _banded_score(group.values, number.values, mask)
            if math.isnan(score):
                return None
            groups, values = group.values[mask], number.values[mask]
            counts = np.bincount(groups)
            used = np.flatnonzero(counts)
            means = np.bincount(groups, values)[used] / counts[used]
            high, low = used[np.argmax(means)], used[np.argmin(means)]
            detail = (
                f"average {number.name} highest for '{group.labels[high]}' ({format_number(float(means.max()))}), "
                f"lowest for '{group.labels[low]}' ({format_number(float(means.min()))})"
            )
            pair.update(value=score, strength=score, detail=detail)
            return pair

        score = _cramers_v(a.codes(), b.codes(), mask)
        if math.isnan(score):
            return None
        pair.update(value=score, strength=score, detail=None)
        return pair


def _measure_pairs(columns: List[_Column], target_position: Optional[int]) -> Dict[str, Any]:
    measurer = _PairMeasurer(columns)
    pairs, too_few, no_variation = [], [], 0
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            rows = measurer.shared_rows(a, b)
            if rows < CORR_MIN_ROWS:
                too_few.append({"columns": [a.name, b.name], "rows": rows})
                continue
            pair = measurer.measure(a, b)
            if pair is None:
                no_variation += 1
                continue
            pairs.append(({
                "columns": [a.name, b.name],
                "positions": [a.position, b.position],
                "kinds": [a.kind, b.kind],
                "rows": rows,
                **pair,
                "involves_target": target_position in (a.position, b.position),
            }, a, b))

    # Exact Spearman for the strongest pairs with gaps: the ones the summary can show
    approximate = [
        (pair, a, b) for pair, a, b in pairs
        if pair["measure"] == "pearson_spearman" and not pair["spearman_exact"] and pair["strength"] >= EXACT_SPEARMAN_FROM
    ]
    approximate.sort(key=lambda item: -item[0]["strength"])
    for pair, a, b in approximate[:EXACT_SPEARMAN_MAX_PAIRS]:
        pair.update(spearman=measurer.exact_spearman(a, b), spearman_exact=True)
        _describe_number_pair(pair)

    for pair, _, _ in pairs:
        if pair["measure"] == "pearson_spearman":
            # Both measures must agree for a near-duplicate: one extreme row can push Pearson alone close to 1
            pair["near_duplicate"] = min(abs(pair["pearson"]), abs(pair["spearman"])) >= CORR_NEAR_PERFECT
        else:
            pair["near_duplicate"] = pair["strength"] >= CORR_NEAR_PERFECT
    pairs = sorted((pair for pair, _, _ in pairs), key=lambda p: -p["strength"])

    # Number-vs-number grid for the heatmap (Pearson and Spearman), None where too few rows are shared
    numbers = measurer.numbers
    grid: Dict[str, Any] = {"columns": [c.name for c in numbers], "pearson": [], "spearman": []}
    for a in numbers:
        grid["pearson"].append([
            float(measurer.pearson[measurer.linear_index[id(a)], measurer.linear_index[id(b)]])
            if measurer.shared_rows(a, b) >= CORR_MIN_ROWS else None
            for b in numbers
        ])
        grid["spearman"].append([
            float(measurer.spearman[measurer.number_index[id(a)], measurer.number_index[id(b)]])
            if measurer.shared_rows(a, b) >= CORR_MIN_ROWS else None
            for b in numbers
        ])
    return {"pairs": pairs, "too_few_rows": too_few, "no_variation": no_variation, "number_grid": grid}


# --- summary ---

def _value_text(pair: Dict[str, Any]) -> str:
    if pair["measure"] == "pearson_spearman":
        return f"Pearson {pair['pearson']:.2f}, Spearman {pair['spearman']:.2f}"
    return f"{pair['value']:.2f} ({MEASURE_LABELS[pair['measure']]})"


def _pair_line(pair: Dict[str, Any]) -> str:
    a, b = pair["columns"]
    details = [d for d in (pair.get("detail"), pair.get("note")) if d]
    return f"- {a} & {b}: {_value_text(pair)}" + (f"; {'; '.join(details)}" if details else "")


def _add_section(builder: SummaryBuilder, header: str, lines: List[str], what: str) -> None:
    """At most CORR_SUMMARY_TOP lines; the closing line counts both the lines cut here and any that didn't fit."""
    shown = lines[:CORR_SUMMARY_TOP]
    hidden = len(lines) - len(shown)
    added = builder.add_items(shown, more=lambda n: f"... and {n + hidden:,} more {what} (see full results)", header=header)
    if hidden and shown and added == len(shown):
        builder.add(f"... and {hidden:,} more {what} (see full results)")


def _build_summary(payload: Dict[str, Any]) -> str:
    builder = SummaryBuilder(TOOL_SUMMARY_MAX_TOKENS)
    measured, pairs = payload["measured_columns"], payload["pairs"]
    target = payload["target_column"]

    if len(measured) < 2:
        builder.add(
            f"Relationships: not measured - only {len(measured)} of {payload['total_columns']:,} columns "
            "can be compared (at least 2 are needed)"
        )
    else:
        kinds = Counter(c["kind"] for c in measured)
        kind_text = ", ".join(f"{kinds[k]} {k}" for k in (NUMBER, TWO_VALUE, CATEGORY) if kinds[k])
        builder.add(
            f"Relationships: measured {len(measured):,} of {payload['total_columns']:,} columns ({kind_text}); "
            f"{len(pairs):,} pairs compared, each using every row where both values are present"
        )
    if target is not None and payload["target_skipped_reason"]:
        builder.add(f"Target '{target}' could not be measured: {payload['target_skipped_reason']}")

    near = [p for p in pairs if p["near_duplicate"]]
    near_lines = [
        _pair_line(p) + (
            " - involves the target: possible leakage, check this column is known before the outcome"
            if p["involves_target"]
            else " - near-duplicate: one almost fully determines the other (a copy, a calculation, or the same fact recorded twice)"
        )
        for p in near
    ]
    _add_section(builder, f"Near-duplicate or possible leakage ({CORR_NEAR_PERFECT:g} or more):", near_lines, "near-duplicate pairs")

    rest = [p for p in pairs if not p["near_duplicate"]]
    if target is not None and not payload["target_skipped_reason"]:
        target_pairs = [p for p in rest if p["involves_target"]]
        strong_target = [p for p in target_pairs if p["strength"] >= CORR_MIN_STRENGTH]
        _add_section(builder, f"Strongest with target '{target}':", [_pair_line(p) for p in strong_target], "target pairs")
        weaker = len(target_pairs) - len(strong_target)
        if weaker:
            builder.add(f"{weaker:,} other column(s) relate to target '{target}' more weakly (below {CORR_MIN_STRENGTH:g})")
        rest = [p for p in rest if not p["involves_target"]]

    strong = [p for p in rest if p["strength"] >= CORR_MIN_STRENGTH and not p.get("curve")]
    if strong:
        _add_section(builder, f"Strongest relationships ({CORR_MIN_STRENGTH:g} or more):", [_pair_line(p) for p in strong], "pairs")
    elif len(measured) >= 2:
        builder.add(f"No other relationships of {CORR_MIN_STRENGTH:g} or more between measured columns")

    curves = [p for p in pairs if p.get("curve")]
    curve_lines = [
        f"- {p['columns'][0]} & {p['columns'][1]}: banded {p['banded']:.2f} vs Pearson {p['pearson']:.2f}, "
        f"Spearman {p['spearman']:.2f} (a curve such as a U-shape)"
        for p in curves
    ]
    _add_section(builder, "Related, but not in a straight line (see the scatter chart):", curve_lines, "curved pairs")

    by_reason: Dict[str, List[str]] = {}
    for column in payload["skipped_columns"]:
        by_reason.setdefault(column["reason"], []).append(column["name"])
    reason_lines = []
    for reason, names in by_reason.items():
        shown = ", ".join(names[:MAX_SKIPPED_NAMES_SHOWN])
        extra = f", and {len(names) - MAX_SKIPPED_NAMES_SHOWN:,} more" if len(names) > MAX_SKIPPED_NAMES_SHOWN else ""
        reason_lines.append(f"- {reason}: {shown}{extra}")
    builder.add_items(reason_lines, more=lambda n: f"... and {n:,} more reasons (see full results)", header="Columns not measured:")

    if payload["too_few_rows"]:
        builder.add(f"{len(payload['too_few_rows']):,} pair(s) not measured: fewer than {CORR_MIN_ROWS} rows have both values")
    if payload["no_variation"]:
        builder.add(f"{payload['no_variation']:,} pair(s) not measured: a column has only one value in the shared rows")
    return builder.text()


# --- entry point ---

def analyze_correlations(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    focus_columns: Optional[Sequence[str]] = None,
) -> ToolResult:
    """Measure how strongly pairs of columns are related.

    Numbers vs numbers get Pearson and Spearman plus a banded check for curves; two-value columns get
    point-biserial or phi; categories get eta or Cramer's V. IDs, free text, dates and constant columns
    are skipped with a reason. At most PAIR_CHECK_MAX_COLUMNS columns are compared, chosen in this
    order: `target_column`, `focus_columns` (from the planner), fewest missing values, file order.
    """
    try:
        target_position = None
        if target_column is not None:
            matches = [i for i in range(df.shape[1]) if df.columns[i] == target_column]
            if not matches:
                raise ValueError(f"target_column '{target_column}' is not a column in the dataset")
            target_position = matches[0]

        prepared, skipped = [], []
        for position in range(df.shape[1]):
            # By position, so duplicate column names can't confuse lookups
            name, series = str(df.columns[position]), df.iloc[:, position]
            try:
                column = _prepare_column(position, name, series)
            except Exception as exc:
                logger.warning("correlation_analyzer: could not prepare column '%s': %s", name, exc)
                column = "values could not be read as numbers or categories"
            if isinstance(column, str):
                skipped.append({"name": name, "position": position, "reason": column})
            else:
                prepared.append(column)

        ordered = _choose_columns(prepared, target_position, focus_columns or [])
        kept = sorted(ordered[:PAIR_CHECK_MAX_COLUMNS], key=lambda c: c.position)
        for column in ordered[PAIR_CHECK_MAX_COLUMNS:]:
            skipped.append({
                "name": column.name, "position": column.position,
                "reason": f"over the {PAIR_CHECK_MAX_COLUMNS}-column limit for pair checks",
            })
        skipped.sort(key=lambda c: c["position"])

        results = _measure_pairs(kept, target_position) if len(kept) >= 2 else {
            "pairs": [], "too_few_rows": [], "no_variation": 0,
            "number_grid": {"columns": [], "pearson": [], "spearman": []},
        }
        target_reason = next((c["reason"] for c in skipped if c["position"] == target_position), None)

        payload = to_json_safe({
            "total_rows": len(df),
            "total_columns": df.shape[1],
            "target_column": None if target_column is None else str(target_column),
            "target_skipped_reason": target_reason,
            "measured_columns": [
                {"name": c.name, "position": c.position, "kind": c.kind, "missing": c.missing} for c in kept
            ],
            "skipped_columns": skipped,
            **results,
            "settings": {
                "min_rows": CORR_MIN_ROWS, "min_strength": CORR_MIN_STRENGTH, "near_perfect": CORR_NEAR_PERFECT,
                "max_columns": PAIR_CHECK_MAX_COLUMNS, "max_categories": CORR_MAX_CATEGORIES, "bands": CORR_BANDS,
            },
        })
        return ToolResult(tool_name=TOOL_NAME, success=True, payload=payload, summary=_build_summary(payload))
    except Exception as exc:
        logger.exception("correlation_analyzer failed")
        return ToolResult(tool_name=TOOL_NAME, success=False, payload={}, summary="", error=str(exc))
