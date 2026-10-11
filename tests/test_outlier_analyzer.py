"""Tests for backend/tools/outlier_analyzer.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.config import OUTLIER_ROWS_KEPT, ROW_CHECK_MAX_COLUMNS, TOOL_SUMMARY_MAX_TOKENS
from backend.tools.outlier_analyzer import analyze_outliers
from backend.utils.file_handler import load_file
from backend.utils.token_manager import estimate_tokens
from tests.test_planner import AWKWARD_INPUTS

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def column(payload: dict, name: str) -> dict:
    return next(c for c in payload["columns"] if c["name"] == name)


def skipped(payload: dict) -> dict:
    return {c["name"]: c["reason"] for c in payload["skipped_columns"]}


def line_for(summary: str, name: str) -> str:
    return next(line for line in summary.splitlines() if line.startswith(f"- {name}:"))


def customers() -> pd.DataFrame:
    """A bank's customers.csv: long-tailed income, two ages typed as 999, discounts that are mostly 0 plus one 95."""
    rng = np.random.default_rng(7)
    n = 1000
    df = pd.DataFrame({
        "income": np.round(np.exp(rng.normal(np.log(55000), 0.6, n)), -2),
        "age": np.r_[np.round(rng.normal(40, 12, n - 2)).clip(18, 90), [999, 999]],
        "discount_pct": np.where(rng.random(n) < 0.7, 0, rng.integers(5, 30, n)),
    })
    df.loc[0, "discount_pct"] = 95
    return df


def employees(normal_rows: int = 2000, impossible_rows: int = 5) -> pd.DataFrame:
    """HR's employees.csv: age, experience and salary move together; the last rows are aged ~25 with ~20 years' experience."""
    rng = np.random.default_rng(11)
    age = rng.normal(42, 10, normal_rows).clip(21, 65)
    experience = (age - 22 - rng.uniform(0, 6, normal_rows)).clip(0)
    salary = 40000 + 2500 * experience + rng.normal(0, 8000, normal_rows)
    bad = pd.DataFrame({
        "age": rng.integers(23, 27, impossible_rows).astype(float),
        "years_experience": rng.integers(18, 22, impossible_rows).astype(float),
        "salary": rng.normal(89000, 2000, impossible_rows).round(-2),
    })
    good = pd.DataFrame({"age": age.round(), "years_experience": experience.round(), "salary": salary.round(-2)})
    return pd.concat([good, bad], ignore_index=True)


@pytest.fixture(scope="module")
def customer_result():
    return analyze_outliers(customers())


@pytest.fixture(scope="module")
def orders_result():
    df = load_file((SAMPLE_DIR / "shop_orders.csv").read_bytes(), "shop_orders.csv").df
    return analyze_outliers(df)


# --- one column at a time ---

def test_typos_are_shown_with_the_most_extreme_values(customer_result):
    # Two 999 typos and one real 76-year-old: the extreme values let the LLM tell them apart
    assert line_for(customer_result.summary, "age") == "- age: 5 high (0.5%), normal range 5.5 to 73.5, highest: 999, 999, 76"


def test_typos_do_not_switch_a_column_to_the_log_scale(customer_result):
    # Ordinary skew of this column is 20.28 because of the typos; quartile skew ignores them
    assert column(customer_result.payload, "age")["log_scale"] is False


def test_long_tailed_column_uses_the_log_scale(customer_result):
    income = column(customer_result.payload, "income")
    assert (income["log_scale"], income["outlier_count"]) == (True, 8)
    assert line_for(customer_result.summary, "income").startswith(
        "- income: 4 high and 4 low (0.8%), normal range 11,904 to 237,503, log scale"
    )


def test_mostly_zero_column_is_checked_among_the_other_values(customer_result):
    assert line_for(customer_result.summary, "discount_pct") == (
        "- discount_pct: 68% are 0; checked among the other 315 values, 1 high (0.32%), normal range -7 to 41, highest: 95"
    )


def test_flagged_row_numbers_start_at_one_and_most_extreme_first(customer_result):
    # The two 999s are the last two data rows of the file
    assert column(customer_result.payload, "age")["rows"][:2] == [999, 1000]


def test_few_value_column_flags_only_rare_values_outside_the_range(orders_result):
    assert line_for(orders_result.summary, "basket_items") == (
        "- basket_items: 1 high (0.33%), normal range -4 to 12, few different values: only values under 1% of rows flagged, highest: 200"
    )
    assert column(orders_result.payload, "support_calls")["highest"] == [60]


def test_star_ratings_are_not_outliers():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"rating": rng.choice([1, 2, 3, 4, 5], 1000, p=[0.08, 0.04, 0.08, 0.25, 0.55])})
    assert column(analyze_outliers(df).payload, "rating")["outlier_count"] == 0


def test_zero_one_columns_are_not_checked():
    # The old version reported every readmitted patient as an outlier
    df = load_file((SAMPLE_DIR / "hospital.xlsx").read_bytes(), "hospital.xlsx").df
    assert skipped(analyze_outliers(df).payload)["readmitted"] == "two values or fewer"


def test_ids_dates_and_text_are_not_checked(orders_result):
    reasons = skipped(orders_result.payload)
    assert reasons["order_id"] == "looks like a row number, ID or evenly spaced sequence"
    assert reasons["order_date"] == "not a number column"
    dates = pd.DataFrame({"day": pd.date_range("2024-01-01", periods=30), "x": range(30)})
    assert skipped(analyze_outliers(dates).payload)["day"] == "date or time"


def test_infinite_values_are_counted_separately():
    values = [float(i % 37) for i in range(100)] + [np.inf, -np.inf]
    result = analyze_outliers(pd.DataFrame({"x": values}))
    assert column(result.payload, "x")["infinite"] == 2
    assert "2 infinite values" in result.summary


def test_too_few_values_are_not_judged():
    assert skipped(analyze_outliers(pd.DataFrame({"x": [1.5, 2.5, 3.5, 9.0] * 4})).payload)["x"] == "fewer than 20 values to check"


# --- unusual combinations across columns ---

def test_impossible_combinations_are_found():
    result = analyze_outliers(employees())
    check = result.payload["row_check"]
    assert (check["status"], check["unusual_count"], check["warning"]) == ("done", 5, False)
    assert sorted(r["row"] for r in check["rows"]) == [2001, 2002, 2003, 2004, 2005]
    assert check["main_columns"] == {"age": 5}


def test_robust_check_still_works_when_many_rows_are_bad():
    # 10% impossible rows: the classic (non-robust) version caught none of them
    check = analyze_outliers(employees(normal_rows=3000, impossible_rows=333)).payload["row_check"]
    assert check["unusual_count"] == 333  # every bad row, no false alarms
    assert len(check["rows"]) == OUTLIER_ROWS_KEPT and min(r["row"] for r in check["rows"]) > 3000
    # 10% of rows forming their own group is far above the ~0.1% expected, so the caution is shown too
    assert check["warning"] is True


def test_row_check_is_repeatable():
    df = employees()
    assert analyze_outliers(df).payload == analyze_outliers(df).payload


def test_row_check_leaves_out_sparse_near_copy_and_few_value_columns(orders_result):
    check = orders_result.payload["row_check"]
    left_out = {c["name"]: c["reason"] for c in check["left_out"]}
    assert check["columns"] == ["order_value", "rating", "outdoor_temp_c", "shipping_cost"]
    assert left_out["loyalty_points"] == "93% missing"
    assert left_out["order_value_cad"] == "near-copy of order_value"
    assert left_out["basket_items"] == "only 10 or fewer different values"


def test_curved_columns_get_a_caution(orders_result):
    # shipping cost is U-shaped in temperature, which a straight-line shape can't describe
    assert orders_result.payload["row_check"]["warning"] is True
    assert any(line.startswith("Caution: far more unusual rows") for line in orders_result.summary.splitlines())


def test_row_check_needs_enough_complete_rows():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    check = analyze_outliers(df).payload["row_check"]
    assert (check["status"], check["reason"]) == ("skipped", "only 60 complete rows (at least 100 needed for 3 columns)")


def test_row_check_prefers_target_and_focus_columns():
    rng = np.random.default_rng(1)
    names = [f"c{i}" for i in range(ROW_CHECK_MAX_COLUMNS + 5)]
    df = pd.DataFrame(rng.normal(size=(500, len(names))), columns=names)
    df.iloc[:5, -2:] = np.nan  # the last two have gaps, so they'd normally be left out first
    check = analyze_outliers(df, target_column=names[-1], focus_columns=[names[-2]]).payload["row_check"]
    assert len(check["columns"]) == ROW_CHECK_MAX_COLUMNS
    assert {names[-1], names[-2]} <= set(check["columns"])


def test_unknown_target_fails_cleanly():
    result = analyze_outliers(pd.DataFrame({"a": [1, 2]}), target_column="nope")
    assert (result.success, result.error) == (False, "target_column 'nope' is not a column in the dataset")


def test_duplicate_column_names_are_each_checked():
    df = pd.DataFrame({"x": [float(i % 30) for i in range(60)] + [500.0], "y": [float(i % 30) for i in range(61)]})
    df.columns = ["amount", "amount"]
    payload = analyze_outliers(df).payload
    assert [c["outlier_count"] for c in payload["columns"]] == [1, 0]


# --- size and robustness ---

def test_wide_file_summary_stays_under_limit():
    result = analyze_outliers(pd.DataFrame(np.random.default_rng(0).standard_t(2, size=(500, 400))))
    assert estimate_tokens(result.summary) <= TOOL_SUMMARY_MAX_TOKENS
    assert result.summary.splitlines()[-1].startswith("... and")
    assert len(result.payload["columns"]) == 400


@pytest.mark.parametrize("make_df", AWKWARD_INPUTS.values(), ids=AWKWARD_INPUTS.keys())
def test_awkward_inputs_succeed_with_strict_json(make_df):
    result = analyze_outliers(make_df())

    assert result.success, result.error
    json.dumps(result.payload, allow_nan=False)
    assert estimate_tokens(result.summary) <= TOOL_SUMMARY_MAX_TOKENS
