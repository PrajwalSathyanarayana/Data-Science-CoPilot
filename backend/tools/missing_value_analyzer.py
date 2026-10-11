"""Tool — measures missing data per column and per row, counts placeholder values, and finds columns missing together."""
import logging
from itertools import combinations
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import (
    CO_MISSING_MIN_OVERLAP,
    CO_MISSING_MIN_ROWS,
    MISSING_HIGHLIGHT_PCT,
    MISSING_PLACEHOLDERS,
    PAIR_CHECK_MAX_COLUMNS,
    TOOL_SUMMARY_MAX_TOKENS,
)
from ..models import ToolResult
from ..utils.json_safe import to_json_safe
from ..utils.summary_builder import SummaryBuilder

logger = logging.getLogger(__name__)

TOOL_NAME = "missing_value_analyzer"
PLACEHOLDER_SET = {value.lower() for value in MISSING_PLACEHOLDERS}
MAX_PLACEHOLDER_VALUES_SHOWN = 3


def _share(part: int, whole: int) -> float:
    return 100 * part / whole if whole else 0.0


def _format_pct(pct: float) -> str:
    # "24.1%", but small shares keep a meaningful digit: "0.02%" rather than "0.0%"
    return f"{pct:.1f}%" if pct >= 1 or pct == 0 else f"{pct:.2g}%"


def _placeholders(series: pd.Series) -> Dict[str, int]:
    """Count text values that look like "no data" ("?", "unknown", whitespace-only). The data is not changed."""
    if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
        return {}
    text = series.dropna().astype(str).str.strip()
    is_placeholder = (text == "") | text.str.lower().isin(PLACEHOLDER_SET)
    if not is_placeholder.any():
        return {}
    # Show whitespace-only cells as a readable label instead of an invisible string
    found = text[is_placeholder].replace("", "(blank)")
    return {str(value): int(count) for value, count in found.value_counts().items()}


def _column_report(name: str, series: pd.Series, total_rows: int) -> Dict[str, Any]:
    missing = int(series.isna().sum())
    report = {
        "name": name,
        "missing": missing,
        "missing_pct": _share(missing, total_rows),
        "placeholders": 0,
        "placeholder_pct": 0.0,
        "placeholder_values": {},
    }
    # An odd column (e.g. lists in cells) must not fail the whole tool
    try:
        found = _placeholders(series)
    except Exception as exc:
        logger.warning("missing_value_analyzer: placeholder check failed on column '%s': %s", name, exc)
        found = {}
    if found:
        count = sum(found.values())
        report.update(placeholders=count, placeholder_pct=_share(count, total_rows), placeholder_values=found)
    return report


def _row_report(is_missing: np.ndarray) -> Dict[str, Any]:
    """Complete and incomplete rows, ignoring fully empty columns.

    A column with no values at all would make every row look incomplete, hiding the real
    picture; the obvious fix for it is dropping the column, which the summary already says.
    """
    rows = is_missing.shape[0]
    fully_empty = is_missing.all(axis=0) if rows else np.zeros(is_missing.shape[1], dtype=bool)
    used = is_missing[:, ~fully_empty]
    with_missing = int(used.any(axis=1).sum()) if used.shape[1] else 0
    return {
        "complete": rows - with_missing,
        "complete_pct": _share(rows - with_missing, rows),
        "with_missing": with_missing,
        "with_missing_pct": _share(with_missing, rows),
        "entirely_empty": int(used.all(axis=1).sum()) if used.shape[1] else 0,
        "ignored_empty_columns": int(fully_empty.sum()),
    }


def _missing_together(is_missing: np.ndarray, columns: List[Dict[str, Any]], total_rows: int) -> List[Dict[str, Any]]:
    """Pairs of columns whose gaps fall in the same rows (often one cause, e.g. a skipped form section)."""
    # Candidates: partly missing columns (fully empty ones trivially "overlap"), most-missing first, capped
    candidates = [
        i for i, col in enumerate(columns)
        if col["missing_pct"] > MISSING_HIGHLIGHT_PCT and col["missing"] < total_rows
    ]
    candidates = sorted(candidates, key=lambda i: -columns[i]["missing"])[:PAIR_CHECK_MAX_COLUMNS]

    pairs = []
    for a, b in combinations(candidates, 2):
        both = int(np.count_nonzero(is_missing[:, a] & is_missing[:, b]))
        if both < CO_MISSING_MIN_ROWS:
            continue
        either = columns[a]["missing"] + columns[b]["missing"] - both
        overlap = both / either
        if overlap >= CO_MISSING_MIN_OVERLAP:
            pairs.append({"columns": [columns[a]["name"], columns[b]["name"]], "both_missing": both, "overlap": overlap})
    # Strongest and biggest first
    return sorted(pairs, key=lambda pair: (-pair["overlap"], -pair["both_missing"]))


def _column_line(column: Dict[str, Any]) -> str:
    parts = []
    if column["missing"]:
        parts.append(f"{column['missing']:,} missing ({_format_pct(column['missing_pct'])})")
    else:
        parts.append("no blanks")
    if column["placeholders"]:
        shown = list(column["placeholder_values"].items())[:MAX_PLACEHOLDER_VALUES_SHOWN]
        values = ", ".join(f"'{value}' x{count:,}" for value, count in shown)
        parts.append(f"{_format_pct(column['placeholder_pct'])} possible placeholders ({values})")
    return f"- {column['name']}: " + ", ".join(parts)


def _build_summary(payload: Dict[str, Any]) -> str:
    rows, columns = payload["total_rows"], payload["columns"]
    builder = SummaryBuilder(TOOL_SUMMARY_MAX_TOKENS)

    with_gaps = [c for c in columns if c["missing"]]
    with_placeholders = [c for c in columns if c["placeholders"]]
    if not with_gaps and not with_placeholders:
        builder.add(f"No missing values in any of the {len(columns):,} columns ({rows:,} rows).")
        return builder.text()

    builder.add(
        f"Missing values: {len(with_gaps):,} of {len(columns):,} columns have gaps "
        f"({payload['total_missing_cells']:,} cells, {_format_pct(payload['missing_cell_pct'])} of all cells)"
    )
    r = payload["rows"]
    ignoring = f" (ignoring {r['ignored_empty_columns']:,} fully empty column(s))" if r["ignored_empty_columns"] else ""
    builder.add(
        f"Rows{ignoring}: {r['complete']:,} of {rows:,} ({_format_pct(r['complete_pct'])}) complete; "
        f"{r['with_missing']:,} have at least one gap; {r['entirely_empty']:,} are entirely empty"
    )

    # Fully empty columns go on one line instead of one line each
    empty = [c["name"] for c in columns if rows and c["missing"] == rows]
    if empty:
        names = ", ".join(empty[:10]) + (f", and {len(empty) - 10} more" if len(empty) > 10 else "")
        builder.add(f"{len(empty):,} column(s) are 100% missing: {names}")

    # Columns above the highlight threshold (gaps + placeholders), most affected first
    highlighted = [
        c for c in columns
        if c["missing"] < rows and c["missing_pct"] + c["placeholder_pct"] > MISSING_HIGHLIGHT_PCT
    ]
    highlighted.sort(key=lambda c: -(c["missing_pct"] + c["placeholder_pct"]))
    builder.add_items(
        [_column_line(c) for c in highlighted],
        more=lambda n: f"... and {n:,} more columns with over {MISSING_HIGHLIGHT_PCT:g}% missing (see full results)",
    )

    small = [c for c in columns if 0 < c["missing_pct"] + c["placeholder_pct"] <= MISSING_HIGHLIGHT_PCT]
    if small:
        builder.add(f"{len(small):,} more column(s) have a little missing data ({MISSING_HIGHLIGHT_PCT:g}% or less)")

    pairs = payload["missing_together"]
    builder.add_items(
        [
            f"- {a} & {b}: {pair['overlap']:.0%} of rows missing either are missing both ({pair['both_missing']:,} rows)"
            for pair in pairs
            for a, b in [pair["columns"]]
        ],
        more=lambda n: f"... and {n:,} more pairs of columns missing together (see full results)",
        header="Columns missing together:",
    )
    return builder.text()


def analyze_missing_values(df: pd.DataFrame) -> ToolResult:
    """Report missing data per column and per row, possible placeholder values, and columns missing together."""
    try:
        total_rows = len(df)
        # By position, so duplicate column names can't confuse lookups
        columns = [_column_report(str(df.columns[i]), df.iloc[:, i], total_rows) for i in range(df.shape[1])]
        is_missing = df.isna().to_numpy()
        total_missing = int(is_missing.sum())

        payload = to_json_safe({
            "total_rows": total_rows,
            "total_columns": df.shape[1],
            "total_missing_cells": total_missing,
            "missing_cell_pct": _share(total_missing, total_rows * df.shape[1]),
            "highlight_threshold_pct": MISSING_HIGHLIGHT_PCT,
            "rows": _row_report(is_missing),
            "columns": columns,  # every column, in file order (the report decides what to show)
            "missing_together": _missing_together(is_missing, columns, total_rows),
        })
        return ToolResult(tool_name=TOOL_NAME, success=True, payload=payload, summary=_build_summary(payload))
    except Exception as exc:
        logger.exception("missing_value_analyzer failed")
        return ToolResult(tool_name=TOOL_NAME, success=False, payload={}, summary="", error=str(exc))
