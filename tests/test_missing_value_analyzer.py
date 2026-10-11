"""Tests for backend/tools/missing_value_analyzer.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.config import TOOL_SUMMARY_MAX_TOKENS
from backend.tools.missing_value_analyzer import analyze_missing_values
from backend.utils.file_handler import load_file
from backend.utils.token_manager import estimate_tokens
from tests.test_planner import AWKWARD_INPUTS

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def column(payload: dict, name: str) -> dict:
    return next(c for c in payload["columns"] if c["name"] == name)


@pytest.fixture(scope="module")
def survey():
    df = load_file((SAMPLE_DIR / "survey_with_gaps.csv").read_bytes(), "survey_with_gaps.csv").df
    return analyze_missing_values(df)


# --- the survey sample file ---

def test_survey_summary(survey):
    assert survey.success and survey.tool_name == "missing_value_analyzer"
    assert survey.summary.splitlines() == [
        "Missing values: 4 of 7 columns have gaps (561 cells, 20.0% of all cells)",
        "Rows (ignoring 1 fully empty column(s)): 319 of 400 (79.8%) complete; 81 have at least one gap; 0 are entirely empty",
        "1 column(s) are 100% missing: fax_number",
        "- income: 80 missing (20.0%)",
        "- employer: 80 missing (20.0%)",
        "- workclass: no blanks, 4.0% possible placeholders ('?' x16)",
        "- marital: no blanks, 2.5% possible placeholders ('unknown' x10)",
        "1 more column(s) have a little missing data (1% or less)",
        "Columns missing together:",
        "- income & employer: 100% of rows missing either are missing both (80 rows)",
    ]


def test_payload_keeps_every_column_including_small_and_zero_gaps(survey):
    names = [c["name"] for c in survey.payload["columns"]]
    assert names == ["respondent_id", "age", "income", "employer", "workclass", "marital", "fax_number"]
    assert column(survey.payload, "age")["missing"] == 1  # 0.25%: below the highlight threshold but kept
    assert column(survey.payload, "respondent_id")["missing"] == 0


def test_placeholders_are_counted_but_data_is_not_changed(survey):
    workclass = column(survey.payload, "workclass")
    assert (workclass["missing"], workclass["placeholders"], workclass["placeholder_values"]) == (0, 16, {"?": 16})


def test_row_report_ignores_fully_empty_columns(survey):
    assert survey.payload["rows"] == {
        "complete": 319, "complete_pct": pytest.approx(79.75), "with_missing": 81,
        "with_missing_pct": pytest.approx(20.25), "entirely_empty": 0, "ignored_empty_columns": 1,
    }


def test_missing_together_pairs(survey):
    assert survey.payload["missing_together"] == [
        {"columns": ["income", "employer"], "both_missing": 80, "overlap": 1.0}
    ]


# --- specific behaviours ---

def test_no_missing_values():
    result = analyze_missing_values(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    assert result.summary == "No missing values in any of the 2 columns (2 rows)."


def test_whitespace_only_cells_count_as_placeholders():
    result = analyze_missing_values(pd.DataFrame({"note": ["ok", "   ", "ok", "\t"] * 10}))
    assert column(result.payload, "note")["placeholder_values"] == {"(blank)": 20}


def test_placeholders_are_matched_ignoring_case_and_spaces():
    result = analyze_missing_values(pd.DataFrame({"x": ["a", " Unknown ", "UNKNOWN", "a"] * 10}))
    assert column(result.payload, "x")["placeholders"] == 20


def test_numeric_columns_are_not_checked_for_placeholders():
    result = analyze_missing_values(pd.DataFrame({"x": [1, 2, 3]}))
    assert column(result.payload, "x")["placeholders"] == 0


def test_pairs_need_enough_shared_rows():
    # Both missing in only 3 rows: below CO_MISSING_MIN_ROWS, so not reported even at 100% overlap
    df = pd.DataFrame({"a": [None] * 3 + [1] * 97, "b": [None] * 3 + [1] * 97})
    assert analyze_missing_values(df).payload["missing_together"] == []


def test_unrelated_gaps_are_not_paired():
    df = pd.DataFrame({"a": [None if i % 2 else 1 for i in range(100)], "b": [None if i % 2 == 0 else 1 for i in range(100)]})
    assert analyze_missing_values(df).payload["missing_together"] == []


def test_entirely_empty_rows_are_counted():
    df = pd.DataFrame({"a": [1, None, 3, None], "b": ["x", None, "z", None]})
    assert analyze_missing_values(df).payload["rows"]["entirely_empty"] == 2


def test_many_fully_empty_columns_are_grouped_on_one_line():
    df = pd.DataFrame({f"empty_{i}": [None] * 5 for i in range(12)} | {"kept": range(5)})
    line = analyze_missing_values(df).summary.splitlines()[2]
    assert line.startswith("12 column(s) are 100% missing: empty_0, empty_1")
    assert line.endswith(", and 2 more")


def test_duplicate_column_names_are_each_reported():
    df = pd.DataFrame([[1, None]] * 5, columns=["a", "a"])
    assert [c["missing"] for c in analyze_missing_values(df).payload["columns"]] == [0, 5]


# --- size and robustness ---

def test_wide_file_summary_stays_under_limit():
    df = pd.DataFrame(np.where(np.random.default_rng(0).random((500, 400)) < 0.3, np.nan, 1.0))
    result = analyze_missing_values(df)

    assert estimate_tokens(result.summary) <= TOOL_SUMMARY_MAX_TOKENS
    assert "more columns with over 1% missing" in result.summary.splitlines()[-1]
    assert len(result.payload["columns"]) == 400


@pytest.mark.parametrize("make_df", AWKWARD_INPUTS.values(), ids=AWKWARD_INPUTS.keys())
def test_awkward_inputs_succeed_with_strict_json(make_df):
    result = analyze_missing_values(make_df())

    assert result.success, result.error
    json.dumps(result.payload, allow_nan=False)
    assert estimate_tokens(result.summary) <= TOOL_SUMMARY_MAX_TOKENS
