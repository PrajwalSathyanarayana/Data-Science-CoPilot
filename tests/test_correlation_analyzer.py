"""Tests for backend/tools/correlation_analyzer.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.config import CORR_MIN_ROWS, PAIR_CHECK_MAX_COLUMNS, TOOL_SUMMARY_MAX_TOKENS
from backend.tools.correlation_analyzer import analyze_correlations
from backend.utils.file_handler import load_file
from backend.utils.token_manager import estimate_tokens
from tests.test_planner import AWKWARD_INPUTS

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def pair(payload: dict, a: str, b: str) -> dict:
    return next(p for p in payload["pairs"] if set(p["columns"]) == {a, b})


def skipped(payload: dict) -> dict:
    return {c["name"]: c["reason"] for c in payload["skipped_columns"]}


@pytest.fixture(scope="module")
def orders():
    return load_file((SAMPLE_DIR / "shop_orders.csv").read_bytes(), "shop_orders.csv").df


@pytest.fixture(scope="module")
def with_target(orders):
    return analyze_correlations(orders, target_column="returned")


# --- the shop_orders sample file ---

def test_overview_line(with_target):
    assert with_target.success and with_target.tool_name == "correlation_analyzer"
    assert with_target.summary.splitlines()[0] == (
        "Relationships: measured 14 of 18 columns (9 number, 3 two-value, 2 category); "
        "78 pairs compared, each using every row where both values are present"
    )


def test_currency_copy_is_a_near_duplicate(with_target):
    p = pair(with_target.payload, "order_value", "order_value_cad")
    assert (p["near_duplicate"], round(p["pearson"], 4), round(p["spearman"], 4)) == (True, 1.0, 1.0)
    assert "- order_value & order_value_cad: Pearson 1.00, Spearman 1.00; rise together - near-duplicate" in with_target.summary


def test_refund_is_reported_as_possible_leakage_of_the_target(with_target):
    p = pair(with_target.payload, "returned", "refund_issued")
    assert (p["measure"], p["near_duplicate"], p["involves_target"]) == ("phi", True, True)
    assert "involves the target: possible leakage" in with_target.summary


def test_target_section_lists_only_meaningful_pairs(with_target):
    lines = with_target.summary.splitlines()
    start = lines.index("Strongest with target 'returned':")
    assert lines[start + 1:start + 4] == [
        "- delivery_days & returned: 0.58 (point-biserial); delivery_days is higher when returned is 'yes'",
        "- rating & returned: -0.42 (point-biserial); rating is higher when returned is 'no'",
        "9 other column(s) relate to target 'returned' more weakly (below 0.3)",
    ]


def test_negative_relationship_keeps_its_sign(with_target):
    p = pair(with_target.payload, "delivery_days", "rating")
    assert (round(p["pearson"], 4), round(p["spearman"], 4), p["rows"], p["spearman_exact"]) == (-0.7206, -0.7175, 294, True)
    assert "- delivery_days & rating: Pearson -0.72, Spearman -0.72; as one rises, the other falls" in with_target.summary


def test_u_shape_is_found_by_the_banded_check(with_target):
    p = pair(with_target.payload, "outdoor_temp_c", "shipping_cost")
    assert (p["curve"], round(p["banded"], 4), round(p["pearson"], 4), round(p["spearman"], 4)) == (True, 0.9564, 0.0604, 0.0739)


def test_one_extreme_row_is_called_out_and_not_a_near_duplicate(with_target):
    p = pair(with_target.payload, "basket_items", "support_calls")
    assert (round(p["pearson"], 4), round(p["spearman"], 4), p["near_duplicate"]) == (0.9133, 0.1131, False)
    assert p["note"] == "may be driven by a few extreme values"


def test_category_vs_number_uses_eta_with_group_averages(with_target):
    p = pair(with_target.payload, "region", "order_value")
    assert (p["measure"], round(p["value"], 4)) == ("eta", 0.8532)
    assert p["detail"] == "average order_value highest for 'north' (116.8), lowest for 'south' (37.22)"


def test_category_vs_category_uses_cramers_v(with_target):
    p = pair(with_target.payload, "customer_segment", "payment_method")
    assert (p["measure"], round(p["value"], 4)) == ("cramers_v", 0.7617)


def test_columns_that_cannot_be_measured_are_listed_with_reasons(with_target):
    assert skipped(with_target.payload) == {
        "order_id": "looks like a row number, ID or evenly spaced sequence",
        "order_date": "too many different values to compare as categories (300): e.g. IDs, free text, dates stored as text",
        "notes": "too many different values to compare as categories (300): e.g. IDs, free text, dates stored as text",
        "gift_wrap": "only one value",
    }


def test_sparse_column_pairs_are_not_measured(with_target):
    # loyalty_points has 20 values, so all 13 of its pairs have fewer than CORR_MIN_ROWS shared rows
    assert len(with_target.payload["too_few_rows"]) == 13
    assert with_target.summary.splitlines()[-1] == "13 pair(s) not measured: fewer than 30 rows have both values"


def test_heatmap_grid_has_number_columns_only(with_target):
    grid = with_target.payload["number_grid"]
    assert grid["columns"][:3] == ["order_value", "order_value_cad", "delivery_days"]
    assert len(grid["pearson"]) == len(grid["columns"]) == 9
    loyalty = grid["columns"].index("loyalty_points")
    assert grid["pearson"][0][loyalty] is None  # too few shared rows


# --- choosing the measure ---

def test_two_value_vs_two_value_uses_phi_with_direction():
    # 50 card orders (5 returned), 50 cash orders (25 returned)
    df = pd.DataFrame({
        "payment_method": ["card"] * 50 + ["cash"] * 50,
        "returned": ["yes"] * 5 + ["no"] * 45 + ["yes"] * 25 + ["no"] * 25,
    })
    p = analyze_correlations(df).payload["pairs"][0]
    assert (p["measure"], round(p["value"], 4)) == ("phi", 0.4364)
    assert p["detail"] == "'yes' in returned is more common when payment_method is 'cash'"


def test_point_biserial_equals_pearson_on_zero_one():
    df = pd.DataFrame({"returned": [0, 0, 0, 1, 0, 1, 0, 1, 1, 1] * 4, "delivery_days": [1, 2, 2, 3, 3, 4, 4, 5, 6, 7.5] * 4})
    p = analyze_correlations(df).payload["pairs"][0]
    assert p["measure"] == "point_biserial"
    assert p["value"] == pytest.approx(df["returned"].corr(df["delivery_days"]))
    assert p["detail"] == "delivery_days is higher when returned is '1'"


def test_boolean_column_is_a_two_value_column():
    df = pd.DataFrame({"express": [True, False, False] * 20, "cost": [9.5, 4.1, 3.9] * 20})
    payload = analyze_correlations(df).payload
    assert [c["kind"] for c in payload["measured_columns"]] == ["two-value", "number"]


def test_curving_but_always_rising_pair_gets_a_note():
    x = [1, 2, 3, 4, 5, 6, 7, 8] * 5
    df = pd.DataFrame({"ad_spend": x, "revenue": [float(np.exp(v)) for v in x]})
    p = analyze_correlations(df).payload["pairs"][0]
    assert p["spearman"] == pytest.approx(1.0)
    assert p["note"] == "consistent order but not a straight line"


# --- reliability guards ---

@pytest.mark.parametrize("rows, measured", [(CORR_MIN_ROWS - 1, False), (CORR_MIN_ROWS, True)])
def test_minimum_shared_rows(rows, measured):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=rows), "b": rng.normal(size=rows)})
    payload = analyze_correlations(df).payload
    assert (len(payload["pairs"]) == 1) is measured


def test_unrelated_columns_give_no_findings():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(500, 6)), columns=list("abcdef"))
    result = analyze_correlations(df)
    assert not any(p["curve"] or p["near_duplicate"] for p in result.payload["pairs"])
    assert "No other relationships of 0.3 or more between measured columns" in result.summary


def test_evenly_spaced_whole_numbers_are_skipped_as_ids_but_irregular_ones_are_not():
    df = pd.DataFrame({"row_no": range(1, 41), "amount": [3, 1, 4, 1, 5, 9, 2, 6] * 5, "irregular": [i * i for i in range(40)]})
    payload = analyze_correlations(df).payload
    assert skipped(payload) == {"row_no": "looks like a row number, ID or evenly spaced sequence"}


def test_category_limit():
    df = pd.DataFrame({
        "fifty": [f"v{i % 50}" for i in range(200)],
        "fifty_one": [f"v{i % 51}" for i in range(200)],
        "amount": [float(i % 7) for i in range(200)],
    })
    payload = analyze_correlations(df).payload
    assert [c["name"] for c in payload["measured_columns"]] == ["fifty", "amount"]
    assert skipped(payload)["fifty_one"].startswith("too many different values to compare as categories (51)")


def test_dates_are_skipped():
    df = pd.DataFrame({"day": pd.date_range("2024-01-01", periods=40), "a": [1.5, 2.5] * 20})
    assert skipped(analyze_correlations(df).payload)["day"] == "date or time (trends over time are a separate analysis)"


# --- column limit, target and focus ---

def test_column_limit_keeps_target_and_focus_columns_first():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(40, PAIR_CHECK_MAX_COLUMNS + 5)), columns=[f"c{i}" for i in range(PAIR_CHECK_MAX_COLUMNS + 5)])
    df.iloc[:5, :10] = np.nan  # the first 10 columns have gaps, so they'd normally lose to complete columns
    payload = analyze_correlations(df, target_column="c0", focus_columns=["c1"]).payload

    measured = [c["name"] for c in payload["measured_columns"]]
    assert len(measured) == PAIR_CHECK_MAX_COLUMNS
    assert {"c0", "c1"} <= set(measured)
    limited = [name for name, reason in skipped(payload).items() if "limit" in reason]
    # c0 (target), c1 (focus), the 70 complete columns, then c2-c4 by file order among the columns with gaps
    assert limited == ["c5", "c6", "c7", "c8", "c9"]


def test_unknown_target_fails_cleanly():
    result = analyze_correlations(pd.DataFrame({"a": [1, 2]}), target_column="nope")
    assert not result.success
    assert result.error == "target_column 'nope' is not a column in the dataset"


def test_target_that_cannot_be_measured_is_explained():
    df = pd.DataFrame({"id": [f"x{i}" for i in range(60)], "a": [1.5, 2.5, 4.0] * 20, "b": [3.0, 1.0, 2.0] * 20})
    result = analyze_correlations(df, target_column="id")
    assert "Target 'id' could not be measured: too many different values" in result.summary


def test_duplicate_column_names_are_compared_with_each_other():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.5] * 20, "y": [2.0, 4.0, 7.0] * 20})
    df.columns = ["amount", "amount"]
    p = analyze_correlations(df).payload["pairs"][0]
    assert (p["columns"], p["positions"]) == (["amount", "amount"], [0, 1])


# --- size, speed shortcuts and robustness ---

def test_many_strong_pairs_stay_under_the_summary_limit():
    base = np.random.default_rng(2).normal(size=60)
    df = pd.DataFrame({f"copy_{i}": base * (i + 1) for i in range(PAIR_CHECK_MAX_COLUMNS)})
    result = analyze_correlations(df)

    assert estimate_tokens(result.summary) <= TOOL_SUMMARY_MAX_TOKENS
    assert any(line.startswith("... and") and "near-duplicate pairs" in line for line in result.summary.splitlines())
    assert len(result.payload["pairs"]) == PAIR_CHECK_MAX_COLUMNS * (PAIR_CHECK_MAX_COLUMNS - 1) // 2


def test_reported_pairs_with_gaps_get_the_exact_spearman():
    # The quick per-column ranking gives 0.6851 here; the exact value is 0.6758
    rng = np.random.default_rng(3)
    x = rng.normal(size=2000)
    df = pd.DataFrame({"x": x, "y": x ** 3 + rng.normal(size=2000)})
    df[rng.random(df.shape) < 0.3] = np.nan
    p = analyze_correlations(df).payload["pairs"][0]
    assert p["spearman_exact"] is True
    assert p["spearman"] == pytest.approx(df["x"].corr(df["y"], method="spearman"), abs=1e-12)


def test_pairs_without_gaps_are_exact_without_recalculating():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"x": rng.normal(size=100), "y": rng.normal(size=100)})
    p = analyze_correlations(df).payload["pairs"][0]
    assert p["spearman_exact"] is True
    assert p["spearman"] == pytest.approx(df["x"].corr(df["y"], method="spearman"), abs=1e-12)


@pytest.mark.parametrize("make_df", AWKWARD_INPUTS.values(), ids=AWKWARD_INPUTS.keys())
def test_awkward_inputs_succeed_with_strict_json(make_df):
    result = analyze_correlations(make_df())

    assert result.success, result.error
    json.dumps(result.payload, allow_nan=False)
    assert estimate_tokens(result.summary) <= TOOL_SUMMARY_MAX_TOKENS
