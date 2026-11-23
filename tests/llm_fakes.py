"""Test helpers — a fake Claude client that returns scripted answers in order, so agent tests cost nothing."""
import json
from types import SimpleNamespace
from typing import Any, List, Sequence, Union

from backend.config import ANALYSIS_MODEL


def make_response(text: str = "{}", stop_reason: str = "end_turn", model: str = ANALYSIS_MODEL) -> SimpleNamespace:
    """A response shaped like the SDK's: a thinking block first, then one text block."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        model=model,
        usage=SimpleNamespace(input_tokens=900, output_tokens=400, iterations=None),
    )


def answer(**fields: Any) -> SimpleNamespace:
    """A normal response whose text is the given fields as JSON."""
    return make_response(json.dumps(fields))


class ScriptedClient:
    """Returns (or raises) the scripted results one per call, and records every request."""

    def __init__(self, results: Sequence[Union[SimpleNamespace, Exception]]) -> None:
        self.results: List[Union[SimpleNamespace, Exception]] = list(results)
        self.requests: List[dict] = []
        create = self._create
        self.messages = SimpleNamespace(create=create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        if not self.results:
            raise AssertionError("the fake client was called more times than scripted")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def user_message(self, call: int) -> str:
        """The user message sent on call number `call` (0-based)."""
        return self.requests[call]["messages"][0]["content"]
