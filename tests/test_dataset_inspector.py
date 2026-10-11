"""Tests for backend/tools/dataset_inspector.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.config import TOOL_SUMMARY_MAX_TOKENS
from backend.tools.dataset_inspector import inspect_dataset
from backend.utils.file_handler import load_file
from backend.utils.token_manager import estimate_tokens
from tests.test_planner import AWKWARD_INPUTS

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def summary_lines(df: pd.DataFrame) -> list:
    return inspect_dataset(df).summary.splitlines()


# --- results on real files ---

def test_weather_sample_file():
    df = load_file((SAMPLE_DIR / "weather_ddmm.csv").read_bytes(), "weather_ddmm.csv").df
    result = inspect_dataset(df)

    assert result.success and result.tool_name == "dataset_inspector"
    assert result.payload["shape"] == {"rows": 340, "columns": 5}
    assert result.payload["kind_counts"] == {"text": 3, "numeric": 2}
    assert result.summary.splitlines() == [
        "Dataset: 340 rows x 5 columns (3 text, 2 numeric)",
        "- observed_on (text): 340 unique, all different",
        "- station (text): 2 unique, most common 'EDM-01' (170 rows)",
        "- temp_c (numeric): mean -0.5118, median -1, min -20, max 20",
        "- rainfall_mm (numeric): mean 4.588, median 1, min 1, max 50",
        "- storm_warning (text): 2 unique, most common 'no' (326 rows)",
    ]


def test_payload_keeps_full_detail_per_column():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
    column = inspect_dataset(df).payload["columns"][0]

    assert column["name"] == "x" and column["dtype"] == "float64" and column["kind"] == "numeric"
    assert column["missing"] == 1
    assert column["stats"] == pytest.approx(
        {"infinite": 0, "mean": 7 / 3, "median": 2.0, "std": 1.5275, "min": 1.0, "max": 4.0}, rel=1e-3
    )


# --- summary lines per column kind ---

def test_date_column_shows_range_not_number_formatting():
    lines = summary_lines(pd.DataFrame({"when": pd.date_range("2024-01-01", periods=4)}))
    assert lines[1] == "- when (date): 2024-01-01 to 2024-01-04"


def test_date_with_time_keeps_the_time():
    lines = summary_lines(pd.DataFrame({"when": pd.to_datetime(["2024-01-01 08:30", "2024-01-02 17:45"])}))
    assert lines[1] == "- when (date): 2024-01-01 08:30:00 to 2024-01-02 17:45:00"


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0.00012, 0.00034], "mean 0.00023, median 0.00023, min 0.00012, max 0.00034"),
        ([1234567.8, 2.0], "mean 617,285, median 617,285, min 2, max 1,234,568"),
        ([10**18, 1], "mean 5e+17, median 5e+17, min 1, max 1e+18"),
    ],
    ids=["tiny_values_not_zero", "large_values_with_commas", "huge_values"],
)
def test_numbers_stay_readable(values, expected):
    assert summary_lines(pd.DataFrame({"x": values}))[1] == f"- x (numeric): {expected}"


def test_boolean_column_shows_share_true():
    assert summary_lines(pd.DataFrame({"flag": [True, False, True, True]}))[1] == "- flag (boolean): 75% true"


def test_missing_share_is_shown():
    lines = summary_lines(pd.DataFrame({"city": ["a", None, "a", None]}))
    assert lines[1] == "- city (text): 1 unique, most common 'a' (2 rows), 50% missing"


def test_long_text_values_are_shortened():
    long_value = "x" * 100
    line = summary_lines(pd.DataFrame({"note": [long_value, long_value, "y"]}))[1]
    assert f"'{'x' * 40}...'" in line


def test_all_missing_column():
    assert summary_lines(pd.DataFrame({"x": [np.nan, np.nan]}))[1] == "- x (numeric): no usable values, 100% missing"


# --- size limits ---

@pytest.mark.parametrize("rows", [1, 2, 3])
def test_tiny_files_work_and_keep_every_row_as_an_example(rows):
    result = inspect_dataset(pd.DataFrame({"x": range(rows)}))

    assert result.success
    assert len(result.payload["sample_rows"]["rows"]) == rows


def test_example_rows_are_capped_at_three():
    assert len(inspect_dataset(pd.DataFrame({"x": range(100)})).payload["sample_rows"]["rows"]) == 3


def test_wide_file_summary_stays_under_budget_but_payload_has_every_column():
    df = pd.DataFrame(np.random.default_rng(0).random((10, 300)))
    result = inspect_dataset(df)
    lines = result.summary.splitlines()

    assert estimate_tokens(result.summary) <= TOOL_SUMMARY_MAX_TOKENS
    assert lines[0] == "Dataset: 10 rows x 300 columns (300 numeric)"
    assert lines[-1].startswith("... and ") and lines[-1].endswith(" more columns (all columns are in the full results)")
    shown = len(lines) - 2
    assert lines[-1] == f"... and {300 - shown} more columns (all columns are in the full results)"
    assert len(result.payload["columns"]) == 300


# --- robustness ---

@pytest.mark.parametrize("make_df", AWKWARD_INPUTS.values(), ids=AWKWARD_INPUTS.keys())
def test_awkward_inputs_succeed_with_strict_json(make_df):
    result = inspect_dataset(make_df())

    assert result.success, result.error
    json.dumps(result.payload, allow_nan=False)
    assert estimate_tokens(result.summary) <= TOOL_SUMMARY_MAX_TOKENS


def test_duplicate_column_names_are_each_described_and_kept_in_example_rows():
    df = pd.DataFrame([[1, 2]] * 5, columns=["a", "a"])
    payload = inspect_dataset(df).payload

    assert [c["name"] for c in payload["columns"]] == ["a", "a"]
    assert payload["sample_rows"]["columns"] == ["a", "a"]
    assert payload["sample_rows"]["rows"][0] == [1, 2]  # both values kept


def test_infinite_values_are_counted_not_averaged():
    line = summary_lines(pd.DataFrame({"x": [1.0, 2.0, np.inf, -np.inf]}))[1]
    assert line == "- x (numeric): mean 1.5, median 1.5, min 1, max 2, 2 infinite values"


def test_complex_numbers_are_treated_as_text():
    assert inspect_dataset(pd.DataFrame({"z": [1 + 2j, 3 + 0j] * 3})).payload["columns"][0]["kind"] == "text"
