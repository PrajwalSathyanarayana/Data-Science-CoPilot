"""Smoke test for backend/models/analysis_plan.py — the checks on the planner LLM's answer.

Run from the project root:
    python -m scripts.smoke.smoke_analysis_plan
"""
import json

from pydantic import ValidationError

from backend.models import PlannerAnswer, TargetCheck
from scripts.smoke.common import SmokeRunner

# The facts of weather_ddmm.csv plus one mixed-units column, as create_plan passes them
COLUMNS = ["observed_on", "station", "temp_c", "rainfall_mm", "storm_warning", "temperature_reading"]
CONTEXT = {
    "name_lookup": {name: name for name in COLUMNS},
    "target_column": None,
    "target_checks": {
        "observed_on": TargetCheck(verdict="invalid", reason="text with 340 different values (an ID or free text)"),
        "rainfall_mm": TargetCheck(verdict="unclear", reason="whole numbers with only 4 values (1, 2, 5, 50)"),
        "storm_warning": TargetCheck(verdict="classification", reason="has 2 values"),
    },
    "flag_count": 3,
    "mixed_units": {"temperature_reading": {"C": 3, "F": 3}},
}


def first_problem(raw_json: str) -> str:
    """'valid', or the first problem the checks found."""
    try:
        PlannerAnswer.model_validate_json(raw_json, context=CONTEXT)
        return "valid"
    except ValidationError as error:
        return str(error.errors()[0]["msg"]).removeprefix("Value error, ").split("; ")[0]


def main() -> None:
    runner = SmokeRunner("analysis_plan")

    runner.section("Realistic planner answer, with the noise LLMs add")
    raw = json.dumps({
        "rationale": "Storm warnings are rare (4.1%). See [details](http://example.com).",
        "focus_columns": [" storm_warning", "rainfall_mm", "storm_warning"],
        "suggested_target": "storm_warning",
        "problem_type": None,
        "unit_meanings": {"temperature_reading": {"C": "degC", "F": "degF"}},
        "flag_order": [1, 3, 3, 2],
        "confidence": "high",
    })
    answer = PlannerAnswer.model_validate_json(raw, context=CONTEXT)
    runner.check("focus: spaces stripped, repeat removed", ["storm_warning", "rainfall_mm"], answer.focus_columns)
    runner.check("link in rationale made plain", "Storm warnings are rare (4.1%). See details.", answer.rationale)
    runner.check("flag order: repeat removed", [1, 3, 2], answer.flag_order)
    runner.check("unit names kept after the pint check", {"temperature_reading": {"C": "degC", "F": "degF"}}, answer.unit_meanings)
    runner.check("extra 'confidence' key ignored", False, "confidence" in answer.model_dump())

    runner.section("Bad answers are rejected (each triggers the one retry)")
    base = json.loads(raw)
    cases = [
        ("made-up focus column", {"focus_columns": ["revenue"]}, "focus_columns not in the dataset: ['revenue']"),
        ("made-up target", {"suggested_target": "storm"}, "suggested_target 'storm' is not in the dataset"),
        ("target that can't be one", {"suggested_target": "observed_on"},
         "suggested_target 'observed_on' can't be a target: text with 340 different values (an ID or free text)"),
        ("unclear target without a type", {"suggested_target": "rainfall_mm"},
         "target 'rainfall_mm' is unclear (whole numbers with only 4 values (1, 2, 5, 50)): problem_type must be "
         "'classification' or 'regression', with problem_type_reason"),
        ("units measuring different things", {"unit_meanings": {"temperature_reading": {"C": "degC", "F": "kilometer"}}},
         "unit_meanings for 'temperature_reading': 'F' (kilometer) and 'C' (degree_Celsius) measure different things "
         "(fix the names or leave the column out)"),
        ("flag that doesn't exist", {"flag_order": [1, 7]}, "flag_order has numbers that are not flags: [7] (flags are 1 to 3)"),
        ("empty rationale", {"rationale": "  "}, "String should have at least 1 character"),
    ]
    for label, override, expected in cases:
        runner.check(label, expected, first_problem(json.dumps({**base, **override})))

    runner.finish()


if __name__ == "__main__":
    main()
