"""Pydantic models — AnalyzeConfig (the upload's settings) and JobRecord (one analysis job and how it is going)."""
from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal["queued", "running", "done", "failed"]


class AnalyzeConfig(BaseModel):
    """The `config` field sent with an upload. Unknown keys are rejected, so a typo never goes unnoticed."""

    model_config = ConfigDict(extra="forbid")

    target_column: Optional[str] = Field(default=None, min_length=1, max_length=500)
    analysis_depth: Literal["quick", "full"] = "full"
    hypothesis_count: int = Field(default=3, ge=1, le=10)
    model_rec_count: int = Field(default=3, ge=1, le=10)
    # True = dates are dd/mm/yyyy, False = mm/dd/yyyy, missing = work it out from the data
    date_dayfirst: Optional[bool] = None
    # Which unit to keep for a column written in several units, as written in the file: {"temperature": "C"}
    units: Dict[str, str] = Field(default_factory=dict, max_length=200)


class JobRecord(BaseModel):
    """One analysis job. Saved after every change, so its status survives a server restart."""

    report_id: str
    status: JobStatus
    step: str  # what is happening now, in plain words, for the progress display
    dataset_name: str
    created_at: datetime
    finished_at: Optional[datetime] = None
    error: Optional[str] = None  # a message safe to show the user; never internal details
    owner: Optional[str] = None  # the signed-in user, once accounts are added
