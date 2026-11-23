"""Pydantic models for the planner: PlannerAnswer (what the LLM returns) and AnalysisPlan (the final plan Python builds)."""
import logging
from typing import Any, Dict, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from ..config import PLANNER_FOCUS_MAX, PLANNER_RATIONALE_MAX_CHARS, PLANNER_REASON_MAX_CHARS
from ..utils.text_safety import to_plain_markdown
from ..utils.units import UnitError, conversion_factors, most_common_unit
from .column_flag import ColumnFlag
from .llm_response import LLMResponse

logger = logging.getLogger(__name__)

ProblemType = Literal["classification", "regression", "clustering", "unknown"]
# What Python's rule says about a column as a target. "unclear": whole numbers with few values (classes or amounts?)
TargetVerdict = Literal["classification", "regression", "unclear", "invalid"]
TargetSource = Literal["user", "planner", "none"]
ProblemTypeSource = Literal["rule", "planner"]
PlanSource = Literal["planner", "fallback"]


def cut_text(text: str, max_chars: int) -> str:
    """Shorten text to at most max_chars, at a word boundary where possible, ending with '...'."""
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 3]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "..."


class TargetCheck(BaseModel):
    """Python's verdict on one column as a target, with the reason in plain English."""

    verdict: TargetVerdict
    reason: str


class PlannerAnswer(BaseModel):
    """The planner LLM's answer. Every field is checked against the dataset; a failed check triggers one retry.

    Validate with the dataset's facts so names, flags and units can be checked:
        PlannerAnswer.model_validate_json(text, context={
            "name_lookup": {name as shown in the prompt: real column name},
            "target_column": the user's target or None,
            "target_checks": {column: TargetCheck},
            "flag_count": how many flags were numbered in the prompt,
            "mixed_units": {column: {unit as written: count}},
        })
    Without a context only the shape is checked. Extra keys are ignored: they can't reach the plan,
    and rejecting them would spend the only retry on something harmless.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    rationale: str = Field(min_length=1)
    focus_columns: List[str] = Field(default_factory=list)
    suggested_target: Optional[str] = None
    problem_type: Optional[Literal["classification", "regression"]] = None
    problem_type_reason: Optional[str] = None
    unit_meanings: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    flag_order: List[int] = Field(default_factory=list)

    # --- checks that need no dataset ---

    @field_validator("rationale")
    @classmethod
    def clean_rationale(cls, text: str) -> str:
        # Shown in the report: links and HTML are made harmless, and long text is cut instead of retried
        return cut_text(to_plain_markdown(text), PLANNER_RATIONALE_MAX_CHARS)

    @field_validator("problem_type_reason")
    @classmethod
    def clean_reason(cls, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        return cut_text(to_plain_markdown(text), PLANNER_REASON_MAX_CHARS)

    @field_validator("suggested_target", mode="before")
    @classmethod
    def empty_target_is_none(cls, value: Any) -> Any:
        # LLMs write "no target" as "", "none" or null; all mean the same
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return value

    # --- checks against the dataset (only with a context) ---

    @model_validator(mode="after")
    def check_against_dataset(self, info: ValidationInfo) -> "PlannerAnswer":
        context = info.context or {}
        lookup: Optional[Mapping[str, str]] = context.get("name_lookup")
        if lookup is None:
            return self

        errors: List[str] = []
        self.focus_columns = self._check_focus_columns(lookup, errors)
        self._check_target(lookup, context, errors)
        self._check_units(lookup, context.get("mixed_units", {}), errors)
        self._check_flag_order(context.get("flag_count", 0), errors)

        # All problems at once, so the single retry can fix them together
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def _check_focus_columns(self, lookup: Mapping[str, str], errors: List[str]) -> List[str]:
        unknown = [name for name in self.focus_columns if name not in lookup]
        if unknown:
            errors.append(f"focus_columns not in the dataset: {unknown}")
            return self.focus_columns

        columns = list(dict.fromkeys(lookup[name] for name in self.focus_columns))
        if len(columns) > PLANNER_FOCUS_MAX:
            logger.info("Planner named %d focus columns; keeping the first %d", len(columns), PLANNER_FOCUS_MAX)
        return columns[:PLANNER_FOCUS_MAX]

    def _check_target(self, lookup: Mapping[str, str], context: Mapping[str, Any], errors: List[str]) -> None:
        user_target = context.get("target_column")
        checks: Mapping[str, TargetCheck] = context.get("target_checks", {})

        # The user's choice always wins; a suggestion is only read when there is none
        if user_target is not None:
            self.suggested_target = None
        elif self.suggested_target is not None:
            if self.suggested_target not in lookup:
                errors.append(f"suggested_target {self.suggested_target!r} is not in the dataset")
                return
            self.suggested_target = lookup[self.suggested_target]
            check = checks.get(self.suggested_target)
            if check is not None and check.verdict == "invalid":
                errors.append(f"suggested_target {self.suggested_target!r} can't be a target: {check.reason}")
                return

        # The LLM's problem type is only needed (and then required) when the rule can't decide
        target = user_target or self.suggested_target
        check = checks.get(target) if target is not None else None
        if check is not None and check.verdict == "unclear":
            if self.problem_type is None or not self.problem_type_reason:
                errors.append(
                    f"target {target!r} is unclear ({check.reason}): problem_type must be "
                    "'classification' or 'regression', with problem_type_reason"
                )

    def _check_units(self, lookup: Mapping[str, str], mixed_units: Mapping[str, Mapping[str, int]], errors: List[str]) -> None:
        resolved: Dict[str, Dict[str, str]] = {}
        for shown_name, meanings in self.unit_meanings.items():
            column = lookup.get(shown_name, shown_name)
            counts = mixed_units.get(column)
            if counts is None:
                errors.append(f"unit_meanings: {shown_name!r} is not a column with mixed units")
                continue
            if set(meanings) != set(counts):
                errors.append(f"unit_meanings for {column!r} must name exactly these units: {sorted(counts)}")
                continue
            # pint checks the names exist and measure the same thing; Python does the arithmetic later
            try:
                conversion_factors(meanings, most_common_unit(counts))
            except UnitError as exc:
                errors.append(f"unit_meanings for {column!r}: {exc} (fix the names or leave the column out)")
                continue
            resolved[column] = dict(meanings)
        self.unit_meanings = resolved

    def _check_flag_order(self, flag_count: int, errors: List[str]) -> None:
        out_of_range = [n for n in self.flag_order if not 1 <= n <= flag_count]
        if out_of_range:
            errors.append(f"flag_order has numbers that are not flags: {out_of_range} (flags are 1 to {flag_count})")
            return
        self.flag_order = list(dict.fromkeys(self.flag_order))


class AnalysisPlan(BaseModel):
    """The final plan, built by Python from the LLM's checked answer (or without it, as a fallback).

    Saved in the report. All 3 analysis tools always run (D-024), so the plan holds no tool steps.
    """

    plan_source: PlanSource
    fallback_reason: Optional[str] = None
    rationale: str
    focus_columns: List[str] = Field(default_factory=list, max_length=PLANNER_FOCUS_MAX)
    target_column: Optional[str] = None
    target_source: TargetSource
    problem_type: ProblemType
    problem_type_source: ProblemTypeSource
    problem_type_reason: str
    unit_meanings: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    # Every flag from detection, most important first: the LLM's order for the numbered ones, then Python's
    flags: List[ColumnFlag] = Field(default_factory=list)


class PlanResult(BaseModel):
    """What create_plan() returns: the plan, plus every LLM call made for it (for cost and latency metrics)."""

    plan: AnalysisPlan
    llm_calls: List[LLMResponse] = Field(default_factory=list)
    unit_notes: List[str] = Field(default_factory=list)  # unit names that passed checks but couldn't be applied
