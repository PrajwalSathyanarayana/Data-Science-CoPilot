"""Pydantic model — ToolResult schema with full payload and ≤500-token summary for LLM prompts."""
import json
from typing import Dict, Optional

from pydantic import BaseModel, field_validator


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    payload: Dict
    summary: str
    error: Optional[str] = None

    @field_validator("payload")
    @classmethod
    def payload_must_be_json(cls, payload: Dict) -> Dict:
        # Payloads are saved to analysis.json and sent by the API, so they must be strict JSON:
        # no numpy types, no dates, no NaN. Failing here points at the tool that produced bad data.
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"payload is not JSON-safe ({exc}); convert it with utils.json_safe.to_json_safe") from exc
        return payload
