"""Pydantic model — ToolResult schema with full payload and ≤500-token summary for LLM prompts."""
from pydantic import BaseModel
from typing import Optional, Dict

class ToolResult(BaseModel):
    tool_name: str
    success: bool
    payload: Dict
    summary: str
    error: Optional[str] = None

