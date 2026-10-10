"""Smoke test for backend/models/analysis_plan.py.

Run from the project root:
    python -m scripts.smoke.smoke_analysis_plan
"""
import json

from pydantic import ValidationError

from backend.models import AnalysisPlan
from scripts.smoke.common import SmokeRunner

DATASET_COLUMNS = ["age", "income", "signup_date", "churned"]


def try_parse(raw_json: str) -> str:
    """Parse LLM-style JSON and return 'valid' or the first validation error message."""
    try:
        AnalysisPlan.model_validate_json(raw_json, context={"columns": DATASET_COLUMNS})
        return "valid"
    except ValidationError as error:
        return error.errors()[0]["msg"]


def main() -> None:
    runner = SmokeRunner("analysis_plan")

    # A realistic planner response, with the kind of noise LLMs add
    runner.section("Realistic LLM response")
    llm_response = json.dumps({
        "steps": [" missing_value_analyzer", "outlier_analyzer", "outlier_analyzer", "correlation_analyzer"],
        "rationale": "Income is likely skewed, so check outliers before correlations.",
        "focus_columns": ["income", "churned"],
        "problem_type": "classification",
        "confidence": "high",
    })
    plan = AnalysisPlan.model_validate_json(llm_response, context={"columns": DATASET_COLUMNS})
    runner.check(
        "steps: whitespace stripped, duplicate removed",
        ["missing_value_analyzer", "outlier_analyzer", "correlation_analyzer"],
        plan.steps,
    )
    runner.check("focus columns kept", ["income", "churned"], plan.focus_columns)
    runner.check("problem type", "classification", plan.problem_type)
    runner.check("extra 'confidence' key ignored", False, "confidence" in plan.model_dump())

    # Each of these should be rejected, which is what triggers the planner's retry
    runner.section("Hallucinations and bad output are rejected")
    base = json.loads(llm_response)
    allowed = "['missing_value_analyzer', 'outlier_analyzer', 'correlation_analyzer', 'visualization_analyzer']"
    cases = [
        ("made-up tool", {"steps": ["sentiment_analyzer"]},
         f"Value error, Unknown tool(s): ['sentiment_analyzer']. Allowed tools: {allowed}"),
        ("tool that runs outside the planner", {"steps": ["dataset_inspector"]},
         f"Value error, Unknown tool(s): ['dataset_inspector']. Allowed tools: {allowed}"),
        ("made-up column", {"focus_columns": ["income", "revenue"]},
         "Value error, Column(s) not in the dataset: ['revenue']"),
        ("wrong problem type", {"problem_type": "binary"},
         "Input should be 'classification', 'regression', 'clustering' or 'unknown'"),
        ("empty rationale", {"rationale": "  "},
         "String should have at least 1 character"),
        ("no steps", {"steps": []},
         "List should have at least 1 item after validation, not 0"),
    ]
    for label, override, expected_error in cases:
        runner.check(label, expected_error, try_parse(json.dumps({**base, **override})))

    runner.finish()


if __name__ == "__main__":
    main()
