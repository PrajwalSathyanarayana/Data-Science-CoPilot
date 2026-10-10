"""Pydantic model — CompressedContext: prompt-ready tool summaries plus stats on what was cut to fit the budget."""
from typing import List

from pydantic import BaseModel, Field, computed_field


class CompressedContext(BaseModel):
    text: str
    budget: int
    tokens_before: int
    tokens_after: int
    included_tools: List[str] = Field(default_factory=list)
    truncated_tools: List[str] = Field(default_factory=list)
    dropped_tools: List[str] = Field(default_factory=list)

    # Share of tokens removed by compression, e.g. 0.42 means 42% fewer tokens sent to the LLM
    @computed_field
    @property
    def reduction_ratio(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return round(1 - self.tokens_after / self.tokens_before, 4)
