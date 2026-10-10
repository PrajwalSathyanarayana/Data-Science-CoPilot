"""Tests for backend/agents/planner.py — autonomous column detection."""
import io
import logging

import numpy as np
import pandas as pd
import pytest

from backend.agents.planner import detect_interesting_columns
from backend.config import IMBALANCE_THRESHOLD, SKEW_MIN_VALUES


def flags_by_kind(df: pd.DataFrame, kind: str, **kwargs) -> dict:
    """Run detection and return {column: detail} for one kind of flag."""
    return {f.column: f.detail for f in detect_interesting_columns(df, **kwargs) if f.kind == kind}


def binary_column(minority: int, total: int, labels=("yes", "no")) -> list:
    return [labels[0]] * minority + [labels[1]] * (total - minority)


# --- date columns ---

def test_text_dates_are_detected_with_full_range():
    dates = pd.date_range("2023-01-01", periods=500, freq="D").strftime("%Y-%m-%d")
    flags = flags_by_kind(pd.DataFrame({"order_date": dates}), "date_column")

    # Range must come from every row, not the 100-value sample
    assert "ranging from 2023-01-01 to 2024-05-14" in flags["order_date"]


def test_datetime_dtype_is_detected():
    df = pd.DataFrame({"created": pd.date_range("2024-01-01", periods=10, freq="D")})
    assert "2024-01-01 to 2024-01-10" in flags_by_kind(df, "date_column")["created"]


def test_mixed_date_formats_are_detected():
    df = pd.DataFrame({"when": ["2024-01-05", "Jan 6 2024", "07/01/2024", "2024-01-08"] * 5})
    assert "when" in flags_by_kind(df, "date_column")


@pytest.mark.parametrize(
    "values",
    [
        [str(10000 + i) for i in range(50)],     # IDs stored as text
        ["2021", "2022", "2023", "2024"] * 10,  # years stored as text
        ["apple", "banana", "cherry"] * 10,     # ordinary text
    ],
    ids=["numeric_ids", "year_strings", "plain_text"],
)
def test_non_date_text_is_not_flagged(values):
    assert flags_by_kind(pd.DataFrame({"col": values}), "date_column") == {}


def test_column_with_too_few_dates_is_not_flagged():
    values = ["2024-01-05", "not a date"] * 20
    assert flags_by_kind(pd.DataFrame({"col": values}), "date_column") == {}


# --- day/month order ---

SIXTY_DAYS = pd.date_range("2022-01-01", periods=60, freq="D")   # 1 Jan to 1 Mar 2022
# Every day is 12 or less, so dd/mm and mm/dd both parse: truly ambiguous without a setting
AMBIGUOUS_DMY = [f"{day:02d}/{month:02d}/2022" for month in (1, 2, 3) for day in range(1, 13)]


@pytest.mark.parametrize("fmt", ["%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"], ids=["dd/mm", "mm/dd", "dd-mm"])
def test_day_month_order_is_detected_from_the_data(fmt):
    df = pd.DataFrame({"col": SIXTY_DAYS.strftime(fmt)})
    detail = flags_by_kind(df, "date_column")["col"]

    assert "ranging from 2022-01-01 to 2022-03-01" in detail
    assert "ambiguous" not in detail


def test_ambiguous_order_defaults_to_month_first_and_says_so():
    detail = flags_by_kind(pd.DataFrame({"col": AMBIGUOUS_DMY}), "date_column")["col"]

    assert "ranging from 2022-01-01 to 2022-12-03" in detail
    assert "day/month order is ambiguous" in detail


def test_dayfirst_setting_resolves_ambiguous_order():
    detail = flags_by_kind(pd.DataFrame({"col": AMBIGUOUS_DMY}), "date_column", date_dayfirst=True)["col"]

    assert detail == "text column where 100% of values parse as dates, ranging from 2022-01-01 to 2022-03-12"


@pytest.mark.parametrize("dayfirst", [True, False, None])
def test_year_first_dates_ignore_dayfirst_setting(dayfirst):
    # Forcing day-first on 2022-01-05 must not turn it into 1 May
    df = pd.DataFrame({"col": SIXTY_DAYS.strftime("%Y-%m-%d")})
    detail = flags_by_kind(df, "date_column", date_dayfirst=dayfirst)["col"]

    assert "ranging from 2022-01-01 to 2022-03-01" in detail


def test_wrong_dayfirst_setting_falls_back_to_the_data():
    # User says month-first, but the file is clearly day-first (it has 13/01, 14/01, ...)
    df = pd.DataFrame({"col": SIXTY_DAYS.strftime("%d/%m/%Y")})
    detail = flags_by_kind(df, "date_column", date_dayfirst=False)["col"]

    assert "ranging from 2022-01-01 to 2022-03-01" in detail
    assert "did not fit these dates, so it was detected from the data" in detail


# --- skewed (long-tailed) numeric columns ---

RIGHT_SKEWED = [1] * 50 + [2] * 20 + [5] * 10 + [50] * 5  # skew 3.74, mean 4.59, median 1.00


def test_right_skewed_column_is_flagged_with_mean_and_median():
    detail = flags_by_kind(pd.DataFrame({"col": RIGHT_SKEWED}), "skewed_distribution")["col"]
    assert detail == "long right tail (skew 3.74): mean 4.59 vs median 1.00, range 1.00 to 50.00"


def test_left_skewed_column_is_flagged():
    df = pd.DataFrame({"col": [-v for v in RIGHT_SKEWED]})
    assert flags_by_kind(df, "skewed_distribution")["col"].startswith("long left tail")


@pytest.mark.parametrize("name", ["revenue", "rainfall_mm", "patient_stay_days", "x1"])
def test_skew_flag_does_not_depend_on_column_name(name):
    # Same values under any name, from any domain, must give the same result
    detail = flags_by_kind(pd.DataFrame({name: RIGHT_SKEWED}), "skewed_distribution")[name]
    assert detail.startswith("long right tail (skew 3.74)")


@pytest.mark.parametrize(
    "values",
    [
        list(range(100)),               # symmetric
        [7.5] * 50,                      # constant
        RIGHT_SKEWED[:SKEW_MIN_VALUES - 1],  # too few values to trust skew
        [0] * 95 + [1] * 5,              # 0/1 flag: handled by the imbalance check instead
    ],
    ids=["symmetric", "constant", "too_few_values", "binary_numeric"],
)
def test_non_skewed_or_unsuitable_numeric_columns_are_not_flagged(values):
    assert flags_by_kind(pd.DataFrame({"col": values}), "skewed_distribution") == {}


def test_text_and_boolean_columns_are_not_checked_for_skew():
    df = pd.DataFrame({"text": ["a"] * 90 + ["b"] * 10, "flag": [True] * 95 + [False] * 5})
    assert flags_by_kind(df, "skewed_distribution") == {}


# --- class imbalance ---

def test_imbalanced_binary_column_is_flagged_as_possible_target():
    df = pd.DataFrame({"churned": binary_column(minority=8, total=100)})
    detail = flags_by_kind(df, "class_imbalance")["churned"]
    assert detail == "possible target: minority class 'yes' is 8.0% of rows (8 of 100)"


def test_balanced_binary_column_is_not_flagged():
    df = pd.DataFrame({"churned": binary_column(minority=40, total=100)})
    assert flags_by_kind(df, "class_imbalance") == {}


def test_minority_exactly_at_threshold_is_not_flagged():
    minority = int(IMBALANCE_THRESHOLD * 100)
    df = pd.DataFrame({"churned": binary_column(minority=minority, total=100)})
    assert flags_by_kind(df, "class_imbalance") == {}


def test_with_target_only_the_target_is_checked():
    df = pd.DataFrame({
        "churned": binary_column(minority=5, total=100),
        "is_vip": binary_column(minority=3, total=100, labels=(True, False)),
    })
    flags = flags_by_kind(df, "class_imbalance", target_column="churned")

    assert list(flags) == ["churned"]
    assert flags["churned"].startswith("target column:")


def test_multiclass_target_is_not_flagged():
    df = pd.DataFrame({"segment": ["a"] * 90 + ["b"] * 5 + ["c"] * 5})
    assert flags_by_kind(df, "class_imbalance", target_column="segment") == {}


def test_missing_values_are_ignored_when_counting_classes():
    df = pd.DataFrame({"churned": binary_column(minority=5, total=50) + [None] * 50})
    assert "(5 of 50)" in flags_by_kind(df, "class_imbalance")["churned"]


def test_unknown_target_column_raises():
    with pytest.raises(ValueError, match="not a column"):
        detect_interesting_columns(pd.DataFrame({"a": [1, 2]}), target_column="b")


# --- edge cases ---

def test_empty_dataframe_returns_no_flags():
    assert detect_interesting_columns(pd.DataFrame()) == []


def test_all_missing_column_returns_no_flags():
    assert detect_interesting_columns(pd.DataFrame({"col": [None, None, None]})) == []


# --- robustness: awkward inputs real uploads produce ---

N = 60
MESSY_CSV = "Date , Amount,Notes\n01/02/2024, 1200 ,ok\n13/02/2024,abc,\n,99,late\n"

AWKWARD_INPUTS = {
    # empty or tiny
    "empty_dataframe": lambda: pd.DataFrame(),
    "zero_rows": lambda: pd.DataFrame({"a": [], "b": []}),
    "one_row": lambda: pd.DataFrame({"a": [1.5], "b": ["2024-01-01"]}),
    "all_none": lambda: pd.DataFrame({"a": [None] * N}),
    "all_nan": lambda: pd.DataFrame({"a": [np.nan] * N}),
    # messy values
    "numbers_and_text_mixed": lambda: pd.DataFrame({"a": [1, "two", 3.0, None] * 15}),
    "numbers_with_commas": lambda: pd.DataFrame({"a": ["1,200", "3,400", "15"] * 20}),
    "infinity": lambda: pd.DataFrame({"a": [1.0, np.inf, 2.0, -np.inf] * 15}),
    "huge_integers": lambda: pd.DataFrame({"a": [10**18, 1, 2] * 20}),
    "unicode_and_emoji": lambda: pd.DataFrame({"a": ["café", "東京", "\U0001F600"] * 20}),
    "empty_strings_and_spaces": lambda: pd.DataFrame({"a": ["", " ", "x"] * 20}),
    "complex_numbers": lambda: pd.DataFrame({"a": [1 + 2j, 3 + 0j] * 30}),
    # column names
    "integer_column_names": lambda: pd.DataFrame({0: range(N), 1: ["x"] * N}),
    "duplicate_column_names": lambda: pd.DataFrame([[1, 2]] * N, columns=["a", "a"]),
    "wide_2000_columns": lambda: pd.DataFrame(np.random.default_rng(0).random((50, 2000))),
    # special pandas types
    "nullable_int": lambda: pd.DataFrame({"a": pd.array([1, None, 3] * 20, dtype="Int64")}),
    "nullable_bool": lambda: pd.DataFrame({"a": pd.array([True, None, False] * 20, dtype="boolean")}),
    "category": lambda: pd.DataFrame({"a": pd.Categorical(["x", "y", "x"] * 20)}),
    "timedelta": lambda: pd.DataFrame({"a": pd.to_timedelta(range(N), unit="D")}),
    "period": lambda: pd.DataFrame({"a": pd.period_range("2024-01", periods=N, freq="M")}),
    # dates and times
    "timezone_aware": lambda: pd.DataFrame({"a": pd.date_range("2024-01-01", periods=N, tz="America/Edmonton")}),
    "mixed_timezones": lambda: pd.DataFrame({"a": ["2024-01-01T10:00:00+05:30", "2024-01-02T10:00:00-07:00"] * 30}),
    "dates_before_1677": lambda: pd.DataFrame({"a": ["1500-01-01", "1600-06-15"] * 30}),
    "time_only": lambda: pd.DataFrame({"a": ["10:30", "11:45", "23:59"] * 20}),
    "excel_serial_numbers": lambda: pd.DataFrame({"a": [45292, 45293, 45294] * 20}),
    # can't come from a file, but must not crash either
    "lists_in_cells": lambda: pd.DataFrame({"a": [[1, 2], [3], []] * 20}),
    "dicts_in_cells": lambda: pd.DataFrame({"a": [{"k": 1}, {"k": 2}] * 30}),
    "minority_only_blank": lambda: pd.DataFrame({"a": ["yes"] + [None] * (N - 1)}),
    # real file reading
    "messy_csv": lambda: pd.read_csv(io.StringIO(MESSY_CSV)),
}


@pytest.mark.parametrize("make_df", AWKWARD_INPUTS.values(), ids=AWKWARD_INPUTS.keys())
def test_awkward_inputs_never_crash(make_df):
    flags = detect_interesting_columns(make_df())
    assert isinstance(flags, list)


def test_target_given_as_integer_column_name_works():
    df = pd.DataFrame({0: binary_column(minority=5, total=100)})
    # Column names are stored as text in flags
    assert flags_by_kind(df, "class_imbalance", target_column=0)["0"].startswith("target column:")


def test_duplicate_column_names_are_each_checked():
    # e.g. "Amount" and "amount " both become "amount" after names are cleaned up
    df = pd.DataFrame({"x": RIGHT_SKEWED, "y": RIGHT_SKEWED})
    df.columns = ["amount", "amount"]
    flags = [f for f in detect_interesting_columns(df) if f.kind == "skewed_distribution"]

    assert [f.column for f in flags] == ["amount", "amount"]


@pytest.mark.parametrize(
    "values",
    [
        ["10:30", "11:45", "23:59"] * 20,
        ["10:30:15", "08:05:00"] * 30,
        ["Jan 5", "Feb 11", "Mar 7"] * 20,
    ],
    ids=["hh:mm", "hh:mm:ss", "no_year"],
)
def test_values_without_a_year_are_not_reported_as_dates(values):
    # pandas would fill in today's date or year, which is a false fact
    assert flags_by_kind(pd.DataFrame({"col": values}), "date_column") == {}


@pytest.mark.parametrize("fmt", ["%d/%m/%y", "%m/%d/%y", "%d.%m.%y"], ids=["dd/mm/yy", "mm/dd/yy", "dd.mm.yy"])
def test_two_digit_year_dates_are_detected_in_the_right_order(fmt):
    df = pd.DataFrame({"col": SIXTY_DAYS.strftime(fmt)})
    assert "ranging from 2022-01-01 to 2022-03-01" in flags_by_kind(df, "date_column")["col"]


def test_failing_check_is_logged_and_other_checks_still_run(monkeypatch, caplog):
    # Force the date check to blow up, as an unforeseen bug would
    def broken_check(*args):
        raise RuntimeError("unexpected")

    monkeypatch.setattr("backend.agents.planner._date_detail", broken_check)
    df = pd.DataFrame({"col": RIGHT_SKEWED})

    with caplog.at_level(logging.WARNING, logger="backend.agents.planner"):
        flags = detect_interesting_columns(df)

    # The failure is logged loudly, and the other checks on the same column still ran
    assert "date_column check failed on column 'col'" in caplog.text
    assert [f.kind for f in flags] == ["skewed_distribution"]
