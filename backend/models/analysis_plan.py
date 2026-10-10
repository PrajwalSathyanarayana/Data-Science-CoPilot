"""Pydantic model — AnalysisPlan: the planner's ordered tool steps, rationale, focus columns, and problem type."""
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from ..config import AGENT_STEPS_MAX, PLANNER_TOOLS

ProblemType = Literal["classification", "regression", "clustering", "unknown"]


class AnalysisPlan(BaseModel):
    """The planner LLM's output. Validation here is the guard against hallucinated tools and columns.

    To check focus_columns against the real dataset, validate with a context:
        AnalysisPlan.model_validate(data, context={"columns": list(df.columns)})
    Without a context, column names are not checked.
    """

    # Strip stray spaces the LLM may add around names. Extra keys from the LLM are ignored.
    model_config = ConfigDict(str_strip_whitespace=True)

    steps: List[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    focus_columns: List[str] = Field(default_factory=list)
    problem_type: ProblemType

    @field_validator("steps")
    @classmethod
    def check_steps(cls, steps: List[str]) -> List[str]:
        # Made-up tool names are an error so the planner retries with a correction prompt
        unknown = [step for step in steps if step not in PLANNER_TOOLS]
        if unknown:
            raise ValueError(f"Unknown tool(s): {unknown}. Allowed tools: {PLANNER_TOOLS}")

        # Repeats are harmless, so drop them quietly (keeping the first) instead of forcing a retry
        unique_steps = list(dict.fromkeys(steps))
        if len(unique_steps) > AGENT_STEPS_MAX:
            raise ValueError(f"Plan has {len(unique_steps)} steps; the maximum is {AGENT_STEPS_MAX}")
        return unique_steps

    @field_validator("focus_columns")
    @classmethod
    def check_focus_columns(cls, columns: List[str], info: ValidationInfo) -> List[str]:
        unique_columns = list(dict.fromkeys(columns))

        # Only check against the dataset when the caller passed its column list
        known_columns = (info.context or {}).get("columns")
        if known_columns is None:
            return unique_columns

        # The error lists only the bad names, not every column, to keep the retry prompt short
        known = set(known_columns)
        missing = [col for col in unique_columns if col not in known]
        if missing:
            raise ValueError(f"Column(s) not in the dataset: {missing}")
        return unique_columns
