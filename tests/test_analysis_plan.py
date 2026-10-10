"""Tests for backend/models/analysis_plan.py — validation of the planner's LLM output."""
import pytest
from pydantic import ValidationError

from backend.config import AGENT_STEPS_MAX
from backend.models import AnalysisPlan

DATASET_COLUMNS = ["age", "income", "signup_date", "churned"]


def plan_data(**overrides) -> dict:
    """A valid plan as the LLM would return it. Override any field per test."""
    data = {
        "steps": ["missing_value_analyzer", "correlation_analyzer"],
        "rationale": "Check data quality first, then relationships between numeric columns.",
        "focus_columns": ["income", "churned"],
        "problem_type": "classification",
    }
    data.update(overrides)
    return data


# --- valid plans ---

def test_valid_plan_parses():
    plan = AnalysisPlan.model_validate(plan_data(), context={"columns": DATASET_COLUMNS})

    assert plan.steps == ["missing_value_analyzer", "correlation_analyzer"]
    assert plan.focus_columns == ["income", "churned"]
    assert plan.problem_type == "classification"


def test_valid_plan_parses_from_json_string():
    raw = AnalysisPlan.model_validate(plan_data()).model_dump_json()
    plan = AnalysisPlan.model_validate_json(raw, context={"columns": DATASET_COLUMNS})

    assert plan.problem_type == "classification"


def test_extra_fields_from_llm_are_ignored():
    plan = AnalysisPlan.model_validate(plan_data(confidence=0.9))
    assert "confidence" not in plan.model_dump()


def test_whitespace_around_names_is_stripped():
    plan = AnalysisPlan.model_validate(
        plan_data(steps=["  outlier_analyzer "], focus_columns=[" age"]),
        context={"columns": DATASET_COLUMNS},
    )
    assert plan.steps == ["outlier_analyzer"]
    assert plan.focus_columns == ["age"]


def test_focus_columns_default_to_empty():
    data = plan_data()
    del data["focus_columns"]
    assert AnalysisPlan.model_validate(data).focus_columns == []


# --- steps ---

@pytest.mark.parametrize("bad_tool", ["dataset_inspector", "notebook_generator", "made_up_tool"])
def test_tools_outside_planner_list_are_rejected(bad_tool):
    with pytest.raises(ValidationError, match="Unknown tool"):
        AnalysisPlan.model_validate(plan_data(steps=["missing_value_analyzer", bad_tool]))


def test_duplicate_steps_are_removed_keeping_first():
    plan = AnalysisPlan.model_validate(
        plan_data(steps=["outlier_analyzer", "missing_value_analyzer", "outlier_analyzer"])
    )
    assert plan.steps == ["outlier_analyzer", "missing_value_analyzer"]


def test_empty_steps_are_rejected():
    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate(plan_data(steps=[]))


def test_steps_over_max_are_rejected(monkeypatch):
    # Only 4 planner tools exist today, so lower the cap to prove the limit is enforced
    monkeypatch.setattr("backend.models.analysis_plan.AGENT_STEPS_MAX", 2)
    with pytest.raises(ValidationError, match="maximum is 2"):
        AnalysisPlan.model_validate(
            plan_data(steps=["missing_value_analyzer", "outlier_analyzer", "correlation_analyzer"])
        )


def test_step_limit_comes_from_config():
    assert AGENT_STEPS_MAX == 10


# --- focus_columns ---

def test_hallucinated_column_is_rejected_with_context():
    with pytest.raises(ValidationError, match="not in the dataset") as error:
        AnalysisPlan.model_validate(
            plan_data(focus_columns=["income", "revenue"]),
            context={"columns": DATASET_COLUMNS},
        )
    # Only the bad name is reported, not the full column list
    assert "revenue" in str(error.value)
    assert "signup_date" not in str(error.value)


def test_columns_are_not_checked_without_context():
    plan = AnalysisPlan.model_validate(plan_data(focus_columns=["anything_goes"]))
    assert plan.focus_columns == ["anything_goes"]


def test_duplicate_focus_columns_are_removed():
    plan = AnalysisPlan.model_validate(
        plan_data(focus_columns=["age", "age", "income"]),
        context={"columns": DATASET_COLUMNS},
    )
    assert plan.focus_columns == ["age", "income"]


# --- problem_type and rationale ---

@pytest.mark.parametrize("bad_type", ["Classification", "binary", "time_series", ""])
def test_invalid_problem_type_is_rejected(bad_type):
    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate(plan_data(problem_type=bad_type))


def test_missing_problem_type_is_rejected():
    data = plan_data()
    del data["problem_type"]
    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate(data)


@pytest.mark.parametrize("bad_rationale", ["", "   "])
def test_empty_rationale_is_rejected(bad_rationale):
    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate(plan_data(rationale=bad_rationale))
