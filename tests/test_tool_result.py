"""Tests for backend/models/tool_result.py — payloads must be strict JSON."""
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from backend.models import ToolResult


def make(payload):
    return ToolResult(tool_name="t", success=True, payload=payload, summary="s")


def test_plain_json_payload_is_accepted():
    assert make({"a": [1, 2.5, None, "x", True, {"b": []}]}).payload["a"][0] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": pd.Timestamp("2024-01-01")},
        {"x": np.dtype("int64")},
        {"x": pd.Series([1, 2])},
    ],
    ids=["nan", "inf", "timestamp", "dtype", "series"],
)
def test_unsafe_payload_is_rejected_with_a_hint(payload):
    with pytest.raises(ValidationError, match="payload is not JSON-safe .*to_json_safe"):
        make(payload)


def test_numpy_numbers_must_be_converted_first():
    # The standard json module can't write numpy ints, so tools must run payloads through to_json_safe
    with pytest.raises(ValidationError):
        make({"x": np.int64(1)})
