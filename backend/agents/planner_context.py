"""Planner helpers — problem-type rule, flag ranking, and the planner's prompt context (pure Python, no LLM)."""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..config import (
    PLANNER_FLAGS_SHOWN,
    PROBLEM_UNCLEAR_MAX_VALUES,
    PROMPT_CELL_MAX_CHARS,
    PROMPT_COLUMN_NAME_MAX_CHARS,
    TARGET_MAX_CLASSES,
)
from ..models import ColumnFlag, TargetCheck, ToolResult
from ..models.analysis_plan import cut_text
from ..utils.formatting import format_number
from ..utils.token_manager import estimate_tokens

logger = logging.getLogger(__name__)

# Upload text can't open or close our tags: < and > become look-alike characters that mean nothing to us
TAG_LOOKALIKES = str.maketrans({"<": "‹", ">": "›"})
CONTEXT_TAG = "dataset_context"

FLAG_KIND_ORDER = {"class_imbalance": 0, "date_column": 1, "skewed_distribution": 2}
FLAG_KIND_LABELS = {"class_imbalance": "class imbalance", "date_column": "dates", "skewed_distribution": "long tail"}


# ---------------------------------------------------------------------------
# Prompt-injection defence: upload text is data, shown inside one tagged block
# ---------------------------------------------------------------------------

def neutralise_tags(text: str) -> str:
    """Swap < and > for look-alikes, so text from the upload can't close or open a prompt tag."""
    return text.translate(TAG_LOOKALIKES)


def build_name_lookup(columns: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Names as shown in the prompt (cut, tags neutralised), and a lookup back to the real names.

    Returns (shown name per real name, real name per accepted spelling). The lookup accepts the
    shown name, the real name and the real name with tags neutralised, since the inspector summary
    lists full names and the LLM may copy either.
    """
    shown: Dict[str, str] = {}
    lookup: Dict[str, str] = {}
    for column in map(str, columns):
        name = neutralise_tags(cut_text(column, PROMPT_COLUMN_NAME_MAX_CHARS))
        # Two long names can share their first 60 characters; number the repeats
        base, n = name, 2
        while name in lookup and lookup[name] != column:
            name, n = f"{base} #{n}", n + 1
        shown[column] = name
        for spelling in (name, column, neutralise_tags(column)):
            lookup.setdefault(spelling, column)
    return shown, lookup


# ---------------------------------------------------------------------------
# Problem type: Python's rule settles the clear cases (D-024 decision 1)
# ---------------------------------------------------------------------------

def check_target(series: pd.Series) -> TargetCheck:
    """What a column would be as a target: classification, regression, unclear, or invalid (with why).

    "unclear" is whole numbers with few values, e.g. stay_days 1/2/5/50: counts of days (regression)
    or levels (classes)? Only the column's meaning can tell, so the planner LLM picks for these.
    """
    try:
        values = series.dropna()
        if values.empty:
            return TargetCheck(verdict="invalid", reason="has no values")
        if pd.api.types.is_datetime64_any_dtype(values):
            return TargetCheck(verdict="invalid", reason="holds dates")

        distinct = values.nunique()
        if distinct < 2:
            return TargetCheck(verdict="invalid", reason="has only one value")
        if pd.api.types.is_bool_dtype(values) or distinct == 2:
            return TargetCheck(verdict="classification", reason="has 2 values")

        if pd.api.types.is_numeric_dtype(values):
            whole = bool((values % 1 == 0).all())
            if whole and distinct <= PROBLEM_UNCLEAR_MAX_VALUES:
                shown = ", ".join(format_number(v) for v in sorted(values.unique()))
                return TargetCheck(verdict="unclear", reason=f"whole numbers with only {distinct} values ({shown})")
            kind = "whole numbers" if whole else "decimal numbers"
            return TargetCheck(verdict="regression", reason=f"{kind} with {distinct:,} different values")

        if distinct <= TARGET_MAX_CLASSES:
            return TargetCheck(verdict="classification", reason=f"text with {distinct} categories")
        return TargetCheck(verdict="invalid", reason=f"text with {distinct:,} different values (an ID or free text)")
    except (TypeError, ValueError) as exc:
        # Odd cell types (lists, mixed objects) can't be compared; such a column can't be a target either
        logger.warning("Target check failed on column '%s': %s", series.name, exc)
        return TargetCheck(verdict="invalid", reason="values can't be compared")


def target_checks(df: pd.DataFrame, target_column: Optional[str]) -> Dict[str, TargetCheck]:
    """Checks for the user's target only, or for every column when the planner may suggest one."""
    columns = [target_column] if target_column is not None else [str(c) for c in df.columns]
    return {column: check_target(df[column]) for column in columns}


# ---------------------------------------------------------------------------
# Flags: Python ranks them; the LLM reorders only the numbered ones (decision 4b)
# ---------------------------------------------------------------------------

def rank_flags(flags: Sequence[ColumnFlag]) -> List[ColumnFlag]:
    """Python's order: class imbalance (smallest minority first), then dates, then long tails (largest skew first)."""
    def key(flag: ColumnFlag) -> Tuple[int, float]:
        value = flag.value or 0.0
        strength = value if flag.kind == "class_imbalance" else -value  # small minority / big skew = more notable
        return FLAG_KIND_ORDER.get(flag.kind, len(FLAG_KIND_ORDER)), strength
    return sorted(flags, key=key)  # stable: dates keep column order


def apply_flag_order(ranked: Sequence[ColumnFlag], flag_order: Sequence[int], shown: int) -> List[ColumnFlag]:
    """The LLM's order for the numbered flags (1-based), unnamed numbered ones after, then the grouped rest."""
    numbered = list(ranked[:shown])
    chosen = [numbered[n - 1] for n in flag_order if 1 <= n <= len(numbered)]
    rest = [flag for flag in numbered if flag not in chosen]
    return chosen + rest + list(ranked[shown:])


# ---------------------------------------------------------------------------
# The prompt context, kept inside the planner's token budget
# ---------------------------------------------------------------------------

@dataclass
class _Section:
    """One block of the context. Droppable blocks go whole; the others lose lines from the bottom."""

    name: str
    lines: List[str]
    droppable: bool = False


@dataclass
class PlannerContext:
    """The user message for the planner, plus what's needed to check its answer."""

    text: str
    tokens: int
    flag_count: int                       # flags numbered in the prompt (flag_order refers to these)
    name_lookup: Dict[str, str]
    trimmed: List[str] = field(default_factory=list)  # sections dropped or cut to fit the budget


def _inspector_section(inspector: Optional[ToolResult]) -> _Section:
    if inspector is None or not inspector.success:
        error = inspector.error if inspector is not None else "not run"
        return _Section("overview", ["Dataset overview: not available (" + (error or "unknown error") + ")"])
    return _Section("overview", ["Dataset overview:"] + inspector.summary.strip().splitlines())


def _target_section(
    target_column: Optional[str], checks: Mapping[str, TargetCheck], shown: Mapping[str, str]
) -> _Section:
    if target_column is not None:
        check = checks[target_column]
        line = f"Target: {shown[target_column]!r}, chosen by the user. Python's check: {check.verdict} ({check.reason})."
        if check.verdict == "unclear":
            line += " Choose problem_type classification or regression."
        return _Section("target", [line])

    # No target: group the columns by Python's verdict, so the LLM knows what it may suggest
    groups: Dict[str, List[str]] = {}
    for column, check in checks.items():
        groups.setdefault(check.verdict, []).append(shown[column])
    labels = [
        ("classification", "classification"),
        ("regression", "regression"),
        ("unclear", "unclear (if suggested, also choose classification or regression)"),
        ("invalid", "can't be a target"),
    ]
    lines = ["Target: none chosen by the user. Python's check of each column as a target:"]
    lines += [f"- {label}: {', '.join(groups[verdict])}" for verdict, label in labels if groups.get(verdict)]
    return _Section("target", lines)


def _flag_sections(ranked: Sequence[ColumnFlag], shown: Mapping[str, str]) -> Tuple[_Section, _Section, int]:
    """Numbered flags with full detail, and the rest grouped one line per kind."""
    numbered = ranked[:PLANNER_FLAGS_SHOWN]
    lines = ["Column flags (found by Python, numbered for flag_order):"]
    lines += [
        f"{i}. {shown.get(f.column, f.column)} [{FLAG_KIND_LABELS.get(f.kind, f.kind)}]: {f.detail}"
        for i, f in enumerate(numbered, start=1)
    ]
    if not numbered:
        lines = ["Column flags: none"]

    grouped: Dict[str, List[str]] = {}
    for flag in ranked[PLANNER_FLAGS_SHOWN:]:
        name = shown.get(flag.column, flag.column)
        if flag.kind == "skewed_distribution" and flag.value is not None:
            name += f" ({flag.value:.1f})"
        elif flag.kind == "class_imbalance" and flag.value is not None:
            name += f" ({flag.value:.1%})"
        grouped.setdefault(flag.kind, []).append(name)
    group_labels = {
        "class_imbalance": "Also imbalanced two-class columns (minority share)",
        "date_column": "Also dates",
        "skewed_distribution": "Also long-tailed (abs skew)",
    }
    rest = [f"{group_labels.get(kind, kind)}: {', '.join(names)}" for kind, names in grouped.items()]
    return _Section("flags", lines), _Section("more_flags", rest, droppable=True), len(numbered)


def _units_section(mixed_units: Mapping[str, Mapping[str, int]], shown: Mapping[str, str]) -> _Section:
    if not mixed_units:
        return _Section("units", [])
    lines = ["Mixed units (name each unit for pint, or leave the column out):"]
    for column, counts in mixed_units.items():
        units = ", ".join(f"{label or '(no unit)'} x{count:,}" for label, count in counts.items())
        lines.append(f"- {shown.get(column, column)}: {units}")
    return _Section("units", lines)


def _rows_section(inspector: Optional[ToolResult], shown: Mapping[str, str]) -> _Section:
    sample = (inspector.payload.get("sample_rows") if inspector is not None and inspector.success else None) or {}
    columns, rows = sample.get("columns") or [], sample.get("rows") or []
    if not columns or not rows:
        return _Section("rows", [], droppable=True)

    def cell(value: Any) -> str:
        return "" if value is None else cut_text(str(value).replace("\n", " "), PROMPT_CELL_MAX_CHARS)

    lines = [f"Example rows ({len(rows)}, cells cut to {PROMPT_CELL_MAX_CHARS} characters):"]
    lines.append(" | ".join(shown.get(str(c), str(c)) for c in columns))
    lines += [" | ".join(cell(v) for v in row) for row in rows]
    return _Section("rows", lines, droppable=True)


def _render(sections: Sequence[_Section]) -> str:
    inner = "\n\n".join("\n".join(s.lines) for s in sections if s.lines)
    return f"[CONTEXT]\n<{CONTEXT_TAG}>\n{neutralise_tags(inner)}\n</{CONTEXT_TAG}>"


def build_planner_context(
    inspector: Optional[ToolResult],
    ranked_flags: Sequence[ColumnFlag],
    checks: Mapping[str, TargetCheck],
    target_column: Optional[str],
    mixed_units: Mapping[str, Mapping[str, int]],
    loading_notes: Sequence[str],
    columns: Sequence[str],
    budget: int,
) -> PlannerContext:
    """Build the planner's user message within `budget` tokens.

    Over budget, the example rows go first, then the grouped flags, then lines from the bottom of the
    loading notes and the dataset overview. The target and the numbered flags are never cut.
    """
    shown, lookup = build_name_lookup(columns)
    flags, more_flags, flag_count = _flag_sections(ranked_flags, shown)
    notes = _Section("notes", ["Loading notes:"] + list(loading_notes) if loading_notes else [])

    # Reading order in the prompt
    sections = [
        _inspector_section(inspector),
        _target_section(target_column, checks, shown),
        flags,
        more_flags,
        _units_section(mixed_units, shown),
        notes,
        _rows_section(inspector, shown),
    ]
    by_name = {s.name: s for s in sections}

    trimmed: List[str] = []
    text = _render(sections)
    for name in ("rows", "more_flags", "notes", "overview"):
        section = by_name[name]
        while estimate_tokens(text) > budget and section.lines:
            # Droppable sections go whole; the others keep their heading line as long as possible
            if section.droppable or len(section.lines) <= 1:
                section.lines = []
            else:
                section.lines.pop()
            if name not in trimmed:
                trimmed.append(name)
            text = _render(sections)

    tokens = estimate_tokens(text)
    if trimmed:
        logger.info("Planner context trimmed to %d tokens (budget %d): %s", tokens, budget, trimmed)
    if tokens > budget:
        logger.warning("Planner context is %d tokens, over its budget of %d, after all trimming", tokens, budget)
    return PlannerContext(text=text, tokens=tokens, flag_count=flag_count, name_lookup=lookup, trimmed=trimmed)
