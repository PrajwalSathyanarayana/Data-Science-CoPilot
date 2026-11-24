"""Pydantic model — AnalysisRun: everything the orchestrator produced for one upload, before the report is built."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from .analysis_plan import AnalysisPlan
from .llm_response import LLMResponse
from .tool_result import ToolResult


class AnalysisRun(BaseModel):
    """The plan, every tool result (failed ones included) and how the run went.

    The ranked flags live in `plan.flags`. The agents (insights, critic, hypotheses) add their output
    later; the final report is built from this.
    """

    plan: AnalysisPlan
    # Most important first (TOOL_PRIORITY); a failed or timed-out tool is kept with its error
    tool_results: List[ToolResult] = Field(default_factory=list)
    llm_calls: List[LLMResponse] = Field(default_factory=list)
    loading_notes: List[str] = Field(default_factory=list)  # what the loader and the planner's unit names changed
    unit_notes: List[str] = Field(default_factory=list)  # unit names that passed the checks but couldn't be applied
    # Set when a safety limit (steps or memory) stopped the run before every tool ran
    finished_early: bool = False
    finish_reason: Optional[str] = None
    # Wall time per step in milliseconds, plus one entry per tool (for the speed metrics)
    step_timings_ms: Dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reason_matches_flag(self) -> "AnalysisRun":
        if self.finished_early != (self.finish_reason is not None):
            raise ValueError("finish_reason must be set exactly when finished_early is True")
        return self
