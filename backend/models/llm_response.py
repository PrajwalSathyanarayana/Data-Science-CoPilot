"""Pydantic model — LLMResponse: the answer text from one Claude call, plus the numbers needed for metrics."""
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """What call_llm() returns. Agents use `.text`; the rest feeds cost and latency metrics."""

    text: str
    agent: str
    model: str            # the model that actually answered (differs from the requested one after a fallback)
    stop_reason: str
    input_tokens: int
    output_tokens: int    # includes thinking tokens, which are billed as output
    latency_ms: int
    fallback_used: bool = False
