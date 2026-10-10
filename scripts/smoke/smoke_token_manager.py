"""Smoke test for backend/utils/token_manager.py.

Run from the project root:
    python -m scripts.smoke.smoke_token_manager
"""
from backend.models import ToolResult
from backend.utils.token_manager import fit_summaries_to_budget
from scripts.smoke.common import SmokeRunner


def make_result(name: str, n_lines: int, success: bool = True) -> ToolResult:
    lines = "\n".join(f"{name} finding {i}: value={i * 1.5}" for i in range(n_lines))
    return ToolResult(
        tool_name=name,
        success=success,
        payload={},
        summary=lines if success else "",
        error=None if success else "boom",
    )


def main() -> None:
    runner = SmokeRunner("token_manager")

    # Deliberately shuffled input order, one failed tool, and summaries of different sizes
    results = [
        make_result("correlation_analyzer", 30),
        make_result("dataset_inspector", 10),
        make_result("outlier_analyzer", 0, success=False),
        make_result("missing_value_analyzer", 40),
    ]

    # Large budget: nothing should be cut, only reordered by priority
    runner.section("Large budget (max_tokens=5000)")
    large = fit_summaries_to_budget(results, max_tokens=5000)
    runner.check("budget after 20% safety margin", 4000, large.budget)
    runner.check("tokens before compression", 908, large.tokens_before)
    runner.check("tokens after compression", 908, large.tokens_after)
    runner.check("reduction ratio", 0.0, large.reduction_ratio)
    runner.check(
        "included tools, in priority order",
        ["dataset_inspector", "missing_value_analyzer", "outlier_analyzer", "correlation_analyzer"],
        large.included_tools,
    )
    runner.check("truncated tools", [], large.truncated_tools)
    runner.check("dropped tools", [], large.dropped_tools)
    runner.check("failed tool shows a note", True, "Tool failed: boom" in large.text)

    # Tight budget: keep the most important tools, trim one, drop the rest
    runner.section("Tight budget (max_tokens=300)")
    tight = fit_summaries_to_budget(results, max_tokens=300)
    runner.check("budget after 20% safety margin", 240, tight.budget)
    runner.check("tokens after compression", 233, tight.tokens_after)
    runner.check("stays within budget", True, tight.tokens_after <= tight.budget)
    runner.check("reduction ratio", 0.7434, tight.reduction_ratio)
    runner.check("included tools", ["dataset_inspector"], tight.included_tools)
    runner.check("truncated tools", ["missing_value_analyzer"], tight.truncated_tools)
    runner.check("dropped tools", ["outlier_analyzer", "correlation_analyzer"], tight.dropped_tools)

    runner.finish()


if __name__ == "__main__":
    main()
