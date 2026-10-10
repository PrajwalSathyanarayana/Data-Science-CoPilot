"""Tests for backend/utils/token_manager.py — token estimation and fitting tool summaries into a budget."""
import pytest

from backend.config import SAFETY_MARGIN
from backend.models import ToolResult
from backend.utils.token_manager import estimate_tokens, fit_summaries_to_budget


def make_result(name: str, n_lines: int = 5, success: bool = True) -> ToolResult:
    """Build a fake tool result with a summary of `n_lines` similar lines."""
    summary = "\n".join(f"{name} finding {i}: value={i * 1.5}" for i in range(n_lines))
    return ToolResult(
        tool_name=name,
        success=success,
        payload={},
        summary=summary if success else "",
        error=None if success else "something broke",
    )


# --- estimate_tokens ---

@pytest.mark.parametrize(
    "text, expected",
    [
        ("", 0),
        ("abcd", 1),
        ("abcde", 2),  # rounds up, never under-counts
        ("a" * 400, 100),
    ],
    ids=["empty", "exact_multiple", "rounds_up", "long_text"],
)
def test_estimate_tokens(text, expected):
    assert estimate_tokens(text) == expected


# --- fit_summaries_to_budget: normal behaviour ---

def test_everything_fits_when_budget_is_large():
    results = [make_result("dataset_inspector"), make_result("correlation_analyzer")]
    context = fit_summaries_to_budget(results, max_tokens=5000)

    assert context.included_tools == ["dataset_inspector", "correlation_analyzer"]
    assert context.truncated_tools == []
    assert context.dropped_tools == []
    assert context.tokens_after == context.tokens_before
    assert context.reduction_ratio == 0.0


def test_budget_applies_safety_margin():
    context = fit_summaries_to_budget([], max_tokens=1000)
    assert context.budget == int(1000 * (1 - SAFETY_MARGIN))


def test_tools_are_ordered_by_priority_with_unknown_tools_last():
    results = [
        make_result("custom_tool"),
        make_result("correlation_analyzer"),
        make_result("dataset_inspector"),
    ]
    context = fit_summaries_to_budget(results, max_tokens=5000)

    assert context.included_tools == ["dataset_inspector", "correlation_analyzer", "custom_tool"]
    assert context.text.index("## dataset_inspector") < context.text.index("## correlation_analyzer")


# --- fit_summaries_to_budget: when the budget is tight ---

@pytest.mark.parametrize("max_tokens", [50, 120, 300, 800])
def test_output_never_exceeds_budget(max_tokens):
    results = [make_result(name, 40) for name in ("dataset_inspector", "missing_value_analyzer", "outlier_analyzer")]
    context = fit_summaries_to_budget(results, max_tokens=max_tokens)

    assert context.tokens_after <= context.budget


def test_overflowing_tool_is_trimmed_at_line_boundaries_and_rest_dropped():
    results = [
        make_result("dataset_inspector", 5),
        make_result("missing_value_analyzer", 40),
        make_result("outlier_analyzer", 10),
    ]
    context = fit_summaries_to_budget(results, max_tokens=300)

    assert context.included_tools == ["dataset_inspector"]
    assert context.truncated_tools == ["missing_value_analyzer"]
    assert context.dropped_tools == ["outlier_analyzer"]

    # Every kept line must be a complete line from the original summaries
    original_lines = {line for r in results for line in (f"## {r.tool_name}\n" + r.summary).splitlines()}
    for line in context.text.splitlines():
        assert line == "" or line in original_lines


def test_tool_is_dropped_when_only_its_title_would_fit():
    # Budget is 240 after the safety margin. The first tool costs 231 tokens and the
    # separator 1, leaving 8: enough for the next title (7) but not its first line (11).
    first = ToolResult(tool_name="dataset_inspector", success=True, payload={}, summary="x" * 900)
    second = make_result("missing_value_analyzer", 10)
    context = fit_summaries_to_budget([first, second], max_tokens=300)

    assert context.included_tools == ["dataset_inspector"]
    assert context.truncated_tools == []
    assert context.dropped_tools == ["missing_value_analyzer"]
    assert "## missing_value_analyzer" not in context.text


def test_reduction_ratio_matches_token_counts():
    results = [make_result(name, 40) for name in ("dataset_inspector", "missing_value_analyzer")]
    context = fit_summaries_to_budget(results, max_tokens=200)

    expected = round(1 - context.tokens_after / context.tokens_before, 4)
    assert context.reduction_ratio == expected
    assert 0 < context.reduction_ratio < 1


# --- fit_summaries_to_budget: edge cases ---

def test_failed_tool_gets_one_line_note():
    context = fit_summaries_to_budget([make_result("outlier_analyzer", success=False)], max_tokens=500)

    assert "## outlier_analyzer\nTool failed: something broke" in context.text
    assert context.included_tools == ["outlier_analyzer"]


def test_empty_input_returns_empty_context():
    context = fit_summaries_to_budget([], max_tokens=800)

    assert context.text == ""
    assert context.tokens_before == 0
    assert context.tokens_after == 0
    assert context.reduction_ratio == 0.0


@pytest.mark.parametrize("bad_budget", [0, -100])
def test_invalid_budget_raises(bad_budget):
    with pytest.raises(ValueError):
        fit_summaries_to_budget([make_result("dataset_inspector")], max_tokens=bad_budget)
