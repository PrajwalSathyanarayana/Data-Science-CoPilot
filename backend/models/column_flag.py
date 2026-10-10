"""Pydantic model — ColumnFlag: something worth noticing about a column, found by Python before the planner runs."""
from typing import Literal

from pydantic import BaseModel

FlagKind = Literal["date_column", "skewed_distribution", "class_imbalance"]


class ColumnFlag(BaseModel):
    """One finding from autonomous column detection.

    Shown to the planner as context and to the user as a "Copilot Noticed:" card.
    All numbers in `detail` are computed by pandas, never by the LLM.
    """

    kind: FlagKind
    column: str
    detail: str
