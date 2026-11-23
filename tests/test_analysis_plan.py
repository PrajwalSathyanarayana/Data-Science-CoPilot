"""Tests for backend/models/analysis_plan.py — checking the planner LLM's answer, and the final plan's shape."""
import pytest
from pydantic import ValidationError

from backend.config import PLANNER_FOCUS_MAX, PLANNER_RATIONALE_MAX_CHARS
from backend.models import AnalysisPlan, PlannerAnswer, TargetCheck

COLUMNS = ["observed_on", "station", "temp_c", "rainfall_mm", "storm_warning", "temperature_reading"]
CHECKS = {
    "observed_on": TargetCheck(verdict="invalid", reason="text with 340 different values (an ID or free text)"),
    "station": TargetCheck(verdict="classification", reason="has 2 values"),
    "temp_c": TargetCheck(verdict="regression", reason="whole numbers with 41 different values"),
    "rainfall_mm": TargetCheck(verdict="unclear", reason="whole numbers with only 4 values (1, 2, 5, 50)"),
    "storm_warning": TargetCheck(verdict="classification", reason="has 2 values"),
    "temperature_reading": TargetCheck(verdict="invalid", reason="text with 6 different values"),
}


def context(**overrides) -> dict:
    """The dataset facts the planner validates against. Override any key per test."""
    data = {
        "name_lookup": {name: name for name in COLUMNS} | {"a_very_long_name...": "temp_c"},
        "target_column": None,
        "target_checks": CHECKS,
        "flag_count": 3,
        "mixed_units": {"temperature_reading": {"C": 3, "F": 3}},
    }
    data.update(overrides)
    return data


def answer_data(**overrides) -> dict:
    """A valid answer as the LLM would return it."""
    data = {
        "rationale": "Storm warnings are rare, so the imbalance matters most.",
        "focus_columns": ["storm_warning", "rainfall_mm"],
        "suggested_target": "storm_warning",
        "problem_type": None,
        "problem_type_reason": None,
        "unit_meanings": {"temperature_reading": {"C": "degC", "F": "degF"}},
        "flag_order": [1, 3, 2],
    }
    data.update(overrides)
    return data


def validate(data: dict, **context_overrides) -> PlannerAnswer:
    return PlannerAnswer.model_validate(data, context=context(**context_overrides))


def rejection(data: dict, **context_overrides) -> str:
    with pytest.raises(ValidationError) as error:
        validate(data, **context_overrides)
    return str(error.value)


# --- valid answers ---

def test_valid_answer_parses():
    result = validate(answer_data())
    assert result.focus_columns == ["storm_warning", "rainfall_mm"]
    assert result.suggested_target == "storm_warning"
    assert result.unit_meanings == {"temperature_reading": {"C": "degC", "F": "degF"}}
    assert result.flag_order == [1, 3, 2]


def test_answer_parses_from_json_text():
    raw = PlannerAnswer.model_validate(answer_data()).model_dump_json()
    assert PlannerAnswer.model_validate_json(raw, context=context()).suggested_target == "storm_warning"


def test_extra_keys_are_ignored_not_retried():
    result = validate(answer_data(confidence=0.9, steps=["outlier_analyzer"]))
    assert "confidence" not in result.model_dump()


def test_without_context_only_the_shape_is_checked():
    result = PlannerAnswer.model_validate(answer_data(focus_columns=["anything"], flag_order=[99]))
    assert result.focus_columns == ["anything"]


def test_names_shown_cut_in_the_prompt_map_back_to_real_names():
    assert validate(answer_data(focus_columns=["a_very_long_name..."])).focus_columns == ["temp_c"]


def test_only_rationale_is_required():
    result = validate({"rationale": "Nothing stands out."})
    assert result.focus_columns == [] and result.suggested_target is None and result.flag_order == []


# --- rationale and reason (shown in the report) ---

def test_links_in_rationale_are_made_harmless():
    result = validate(answer_data(rationale="See [prizes](http://win.example) and <b>bold</b>."))
    assert result.rationale == "See prizes and bold."


def test_long_rationale_is_cut_not_rejected():
    result = validate(answer_data(rationale="word " * 300))
    assert len(result.rationale) <= PLANNER_RATIONALE_MAX_CHARS
    assert result.rationale.endswith("...")


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_rationale_is_rejected(bad):
    with pytest.raises(ValidationError):
        validate(answer_data(rationale=bad))


# --- focus columns ---

def test_made_up_focus_column_is_rejected():
    assert "focus_columns not in the dataset: ['revenue']" in rejection(answer_data(focus_columns=["revenue"]))


def test_repeated_focus_columns_are_removed():
    assert validate(answer_data(focus_columns=["station", "station", "temp_c"])).focus_columns == ["station", "temp_c"]


def test_focus_columns_over_the_cap_are_cut_to_the_first_ones():
    many = [f"c{i}" for i in range(12)]
    result = validate(answer_data(focus_columns=many), name_lookup={name: name for name in many + COLUMNS})
    assert result.focus_columns == many[:PLANNER_FOCUS_MAX]


# --- target and problem type ---

@pytest.mark.parametrize("empty", ["", "none", "None", "null", None])
def test_no_target_written_any_way_is_none(empty):
    assert validate(answer_data(suggested_target=empty)).suggested_target is None


def test_made_up_target_is_rejected():
    assert "suggested_target 'storm' is not in the dataset" in rejection(answer_data(suggested_target="storm"))


def test_target_that_cannot_be_one_is_rejected_with_the_reason():
    message = rejection(answer_data(suggested_target="observed_on"))
    assert "can't be a target: text with 340 different values" in message


def test_users_target_wins_over_a_suggestion():
    assert validate(answer_data(suggested_target="station"), target_column="storm_warning").suggested_target is None


def test_unclear_target_needs_a_problem_type_and_reason():
    message = rejection(answer_data(suggested_target="rainfall_mm"))
    assert "target 'rainfall_mm' is unclear" in message


def test_unclear_target_with_problem_type_passes():
    result = validate(answer_data(
        suggested_target="rainfall_mm", problem_type="regression", problem_type_reason="Rain is an amount in mm.",
    ))
    assert result.problem_type == "regression"


def test_unclear_user_target_also_needs_a_problem_type():
    assert "target 'rainfall_mm' is unclear" in rejection(answer_data(suggested_target=None), target_column="rainfall_mm")


@pytest.mark.parametrize("bad_type", ["clustering", "binary", "Regression"])
def test_problem_type_must_be_classification_or_regression(bad_type):
    with pytest.raises(ValidationError):
        validate(answer_data(problem_type=bad_type))


# --- unit names ---

def test_units_that_measure_different_things_are_rejected():
    message = rejection(answer_data(unit_meanings={"temperature_reading": {"C": "degC", "F": "kilometer"}}))
    assert "measure different things" in message


def test_unit_names_must_cover_exactly_the_units_found():
    message = rejection(answer_data(unit_meanings={"temperature_reading": {"C": "degC"}}))
    assert "must name exactly these units: ['C', 'F']" in message


def test_unknown_pint_name_is_rejected():
    assert "unknown unit name" in rejection(answer_data(unit_meanings={"temperature_reading": {"C": "degC", "F": "fahrenheitz"}}))


def test_unit_names_for_a_column_without_mixed_units_are_rejected():
    assert "'temp_c' is not a column with mixed units" in rejection(answer_data(unit_meanings={"temp_c": {"C": "degC"}}))


def test_leaving_a_mixed_unit_column_out_is_allowed():
    assert validate(answer_data(unit_meanings={})).unit_meanings == {}


# --- flag order ---

def test_flag_numbers_that_do_not_exist_are_rejected():
    assert "flag_order has numbers that are not flags: [0, 4]" in rejection(answer_data(flag_order=[1, 0, 4]))


def test_repeated_flag_numbers_are_removed():
    assert validate(answer_data(flag_order=[2, 2, 1])).flag_order == [2, 1]


# --- several problems at once ---

def test_all_problems_are_reported_together_for_one_retry():
    message = rejection(answer_data(focus_columns=["revenue"], suggested_target="storm", flag_order=[9]))
    assert "focus_columns" in message and "suggested_target" in message and "flag_order" in message


# --- AnalysisPlan (built by Python) ---

def plan_data(**overrides) -> dict:
    data = {
        "plan_source": "planner",
        "rationale": "Storm warnings are rare.",
        "focus_columns": ["storm_warning"],
        "target_column": "storm_warning",
        "target_source": "planner",
        "problem_type": "classification",
        "problem_type_source": "rule",
        "problem_type_reason": "storm_warning: has 2 values",
    }
    data.update(overrides)
    return data


def test_plan_has_no_tool_steps():
    assert "steps" not in AnalysisPlan.model_validate(plan_data()).model_dump()


def test_plan_rejects_more_focus_columns_than_the_cap():
    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate(plan_data(focus_columns=[f"c{i}" for i in range(PLANNER_FOCUS_MAX + 1)]))


@pytest.mark.parametrize("field,bad", [
    ("plan_source", "llm"), ("target_source", "guess"), ("problem_type", "binary"), ("problem_type_source", "llm"),
])
def test_plan_rejects_unknown_labels(field, bad):
    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate(plan_data(**{field: bad}))
