"""Tool — describes a dataset's shape, column types, per-column statistics, and a few example rows."""
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import INSPECTOR_SAMPLE_ROWS, TOOL_SUMMARY_MAX_TOKENS
from ..models import ToolResult
from ..utils.formatting import format_number
from ..utils.json_safe import to_json_safe
from ..utils.summary_builder import SummaryBuilder

logger = logging.getLogger(__name__)

TOOL_NAME = "dataset_inspector"
MAX_TEXT_VALUE_CHARS = 40  # long text values are shortened in the summary


def _format_date(iso_text: str) -> str:
    """'2024-01-05T00:00:00' -> '2024-01-05'; keeps the time only when there is one."""
    if iso_text.endswith("T00:00:00"):
        return iso_text[:10]
    return iso_text.replace("T", " ")


def _column_kind(series: pd.Series) -> str:
    # bool first: pandas also counts True/False columns as numeric
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    # Complex numbers can't be ranked or averaged like ordinary numbers, so they're treated as text
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_complex_dtype(series):
        return "numeric"
    return "text"


def _column_stats(series: pd.Series, kind: str) -> Dict[str, Any]:
    """Statistics that make sense for this kind of column, computed on its non-missing values."""
    values = series.dropna()
    if values.empty:
        return {}
    if kind == "numeric":
        # Infinity would turn mean and std into nonsense, so stats use finite values and infinities are counted
        finite = values[np.isfinite(values.astype(float))]
        stats = {"infinite": int(len(values) - len(finite))}
        if not finite.empty:
            stats.update({
                "mean": finite.mean(), "median": finite.median(), "std": finite.std(),
                "min": finite.min(), "max": finite.max(),
            })
        return stats
    if kind == "date":
        return {"min": values.min(), "max": values.max()}
    if kind == "boolean":
        return {"true_share": values.astype(bool).mean()}

    counts = values.astype(str).value_counts()
    return {"unique": int(counts.size), "top_value": counts.index[0], "top_count": int(counts.iloc[0])}


def _describe_column(name: str, series: pd.Series) -> Dict[str, Any]:
    kind = _column_kind(series)
    info = {
        "name": name,
        "dtype": str(series.dtype),
        "kind": kind,
        "missing": int(series.isna().sum()),
    }
    # One odd column (e.g. complex numbers, lists in cells) must not fail the whole tool
    try:
        info["stats"] = _column_stats(series, kind)
    except Exception as exc:
        logger.warning("dataset_inspector: could not compute stats for column '%s': %s", name, exc)
        info["stats"] = {}
        info["stats_error"] = str(exc)
    return info


def _summary_line(column: Dict[str, Any], total_rows: int) -> str:
    """One line per column, e.g. "- temp_c (numeric): mean 0.06, median 0, min -20, max 20"."""
    stats, kind = column["stats"], column["kind"]
    if not stats:
        detail = "no usable values"
    elif kind == "numeric":
        if "mean" in stats:
            detail = ", ".join(f"{key} {format_number(stats[key])}" for key in ("mean", "median", "min", "max"))
        else:
            detail = "no finite values"
        if stats["infinite"]:
            detail += f", {stats['infinite']:,} infinite values"
    elif kind == "date":
        detail = f"{_format_date(stats['min'])} to {_format_date(stats['max'])}"
    elif kind == "boolean":
        detail = f"{stats['true_share']:.0%} true"
    elif stats["top_count"] == 1:
        detail = f"{stats['unique']:,} unique, all different"
    else:
        top = str(stats["top_value"])
        if len(top) > MAX_TEXT_VALUE_CHARS:
            top = top[:MAX_TEXT_VALUE_CHARS] + "..."
        detail = f"{stats['unique']:,} unique, most common '{top}' ({stats['top_count']:,} rows)"

    if column["missing"] and total_rows:
        detail += f", {column['missing'] / total_rows:.0%} missing"
    return f"- {column['name']} ({kind}): {detail}"


def _build_summary(rows: int, columns: List[Dict[str, Any]], kind_counts: Dict[str, int]) -> str:
    """Overview line first, then column lines until the token budget is reached."""
    kinds = ", ".join(f"{count} {kind}" for kind, count in kind_counts.items())
    builder = SummaryBuilder(TOOL_SUMMARY_MAX_TOKENS)
    builder.add(f"Dataset: {rows:,} rows x {len(columns):,} columns ({kinds})")
    builder.add_items(
        [_summary_line(column, rows) for column in columns],
        more=lambda n: f"... and {n:,} more columns (all columns are in the full results)",
    )
    return builder.text()


def inspect_dataset(df: pd.DataFrame) -> ToolResult:
    """Describe the dataset. The summary (for the LLM) stays under TOOL_SUMMARY_MAX_TOKENS; the payload has everything."""
    try:
        # By position, so duplicate column names can't confuse lookups
        columns = [_describe_column(str(df.columns[i]), df.iloc[:, i]) for i in range(df.shape[1])]

        kind_counts: Dict[str, int] = {}
        for column in columns:
            kind_counts[column["kind"]] = kind_counts.get(column["kind"], 0) + 1

        sample = df.sample(n=min(INSPECTOR_SAMPLE_ROWS, len(df)), random_state=42) if len(df) else df
        payload = to_json_safe({
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "kind_counts": kind_counts,
            "columns": columns,
            # Columns + row lists (not one dict per row), so duplicate column names don't drop values
            "sample_rows": {"columns": [str(c) for c in sample.columns], "rows": sample.values.tolist()},
        })
        summary = _build_summary(df.shape[0], payload["columns"], kind_counts)

        return ToolResult(tool_name=TOOL_NAME, success=True, payload=payload, summary=summary)
    except Exception as exc:
        logger.exception("dataset_inspector failed")
        return ToolResult(tool_name=TOOL_NAME, success=False, payload={}, summary="", error=str(exc))
