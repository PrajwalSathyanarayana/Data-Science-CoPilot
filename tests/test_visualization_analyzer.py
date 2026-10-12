"""Tests for backend/tools/visualization_analyzer.py (and the planner's parse_date_column it relies on)."""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from backend.agents.planner import detect_interesting_columns, parse_date_column
from backend.config import CHART_HIST_BINS, CHART_MAX_TOTAL, CHART_SCATTER_SAMPLE, TOOL_SUMMARY_MAX_TOKENS
from backend.tools.correlation_analyzer import analyze_correlations
from backend.tools.missing_value_analyzer import analyze_missing_values
from backend.tools.outlier_analyzer import analyze_outliers
from backend.tools.visualization_analyzer import analyze_visualizations
from backend.utils.file_handler import load_file
from backend.utils.token_manager import estimate_tokens
from tests.test_planner import AWKWARD_INPUTS

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def run_all(df: pd.DataFrame, target: Optional[str] = None, dayfirst: Optional[bool] = None):
    """The tools in the order the orchestrator will run them, then the charts."""
    flags = detect_interesting_columns(df, target_column=target, date_dayfirst=dayfirst)
    results = (analyze_missing_values(df), analyze_correlations(df, target), analyze_outliers(df, target))
    payloads = {r.tool_name: r.payload for r in results if r.success}
    return analyze_visualizations(df, payloads, target_column=target, column_flags=flags, date_dayfirst=dayfirst)


def chart(result, chart_id: str) -> dict:
    return next(c for c in result.payload["charts"] if c["id"] == chart_id)


def load(name: str) -> pd.DataFrame:
    return load_file((SAMPLE_DIR / name).read_bytes(), name).df


@pytest.fixture(scope="module")
def orders():
    return run_all(load("shop_orders.csv"), target="returned")


# --- which charts, in which order ---

def test_charts_follow_the_findings_in_report_order(orders):
    ids = [c["id"] for c in orders.payload["charts"]]
    assert ids[:3] == ["missing_values", "heatmap", "balance_1"]
    assert ids[3:15] == [f"rel_{i}" for i in range(1, 13)]
    assert ids[15:19] == ["box_1", "box_2", "time_1", "time_2"]
    assert ids[19:] == [f"dist_{i}" for i in range(1, 16)]


def test_summary_lists_chart_ids_and_titles_for_the_llm(orders):
    lines = orders.summary.splitlines()
    assert lines[0] == "Charts (34) - refer to a chart by its id:"
    assert "- rel_6: Average order_value per region" in lines
    assert estimate_tokens(orders.summary) <= TOOL_SUMMARY_MAX_TOKENS


def test_whole_report_is_small(orders):
    assert len(json.dumps(orders.payload)) < 60_000


def test_leakage_pair_caption(orders):
    assert chart(orders, "rel_2")["caption"] == (
        "1.00 (phi); 'yes' in refund_issued is more common when returned is 'yes'; "
        "near-duplicate involving the target: possible leakage"
    )


def test_category_vs_number_shows_averages_per_group(orders):
    bar = chart(orders, "rel_6")["figure"]["data"][0]
    assert bar["x"][0] == "north" and bar["x"][-1] == "south"
    assert round(bar["y"][0], 1) == 116.8


def test_curve_gets_the_band_average_line(orders):
    names = [trace.get("name") for trace in chart(orders, "rel_12")["figure"]["data"]]
    assert names == ["rows", "average per band"]


def test_extreme_row_is_marked_on_the_scatter(orders):
    traces = chart(orders, "rel_5")["figure"]["data"]
    flagged = next(t for t in traces if t["name"] == "flagged outliers")
    assert (flagged["x"], flagged["y"]) == ([200.0], [60.0])


def test_box_plot_uses_the_outlier_tools_range(orders):
    box = chart(orders, "box_1")
    assert box["caption"] == "1 outlier(s) (0.33%), normal range -4 to 12"
    assert box["figure"]["data"][1]["y"] == [200.0]


def test_text_dates_get_a_time_chart_and_the_target_over_time(orders):
    assert chart(orders, "time_1")["caption"] == "300 dated rows from 2024-01-01 to 2024-10-26, counted per week"
    assert chart(orders, "time_2")["title"] == "% of rows where returned is 'yes' per week over order_date"


def test_distributions_skip_ids_and_free_text(orders):
    columns = {c["columns"][0] for c in orders.payload["charts"] if c["id"].startswith("dist_")}
    assert {"order_id", "notes", "order_date"}.isdisjoint(columns)
    assert {"region", "order_value", "gift_wrap"} <= columns


# --- other sample files ---

def test_missing_together_grid():
    result = run_all(load("survey_with_gaps.csv"))
    assert chart(result, "missing_together")["caption"].startswith("strongest: income & employer, 100%")


def test_day_first_dates_are_read_the_same_way_as_the_planner():
    result = run_all(load("weather_ddmm.csv"), dayfirst=True)
    assert chart(result, "time_1")["caption"] == "340 dated rows from 2022-01-01 to 2022-12-06, counted per week"


def test_target_class_balance():
    result = run_all(load("hospital.xlsx"), target="readmitted")
    assert chart(result, "balance_1")["caption"] == "smallest group '1': 11.8% (20 of 170 rows)"


# --- size: charts carry counts, not rows ---

@pytest.fixture(scope="module")
def big():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=75000)})
    df["y"] = 2 * df["x"] + rng.normal(size=75000)
    df.loc[0, ["x", "y"]] = [40.0, 90.0]  # one extreme row
    return df, run_all(df)


def test_histogram_carries_bin_counts_only(big):
    _, result = big
    hist = next(c for c in result.payload["charts"] if c["kind"] == "histogram")
    assert len(hist["figure"]["data"][0]["y"]) == CHART_HIST_BINS


def test_large_scatter_is_a_labelled_sample_plus_flagged_rows(big):
    df, result = big
    scatter = next(c for c in result.payload["charts"] if c["kind"] == "scatter")
    rows, flagged = scatter["figure"]["data"][0], scatter["figure"]["data"][1]
    assert len(rows["x"]) == CHART_SCATTER_SAMPLE
    assert scatter["sampled"] == {"shown": CHART_SCATTER_SAMPLE, "of": 75000}
    assert f"random sample of 2,000 of 75,000 rows" in scatter["caption"]
    assert 40.0 in flagged["x"]  # the extreme row is always shown


def test_small_scatter_shows_every_point(orders):
    scatter = chart(orders, "rel_1")
    assert scatter["sampled"] is None and len(scatter["figure"]["data"][0]["x"]) == 300


def test_time_period_keeps_points_under_the_limit():
    days = pd.date_range("2023-01-01", periods=730, freq="D")
    result = analyze_visualizations(pd.DataFrame({"day": days, "sales": np.arange(730) % 17}))
    assert chart(result, "time_1")["caption"].endswith("counted per month")
    assert len(chart(result, "time_1")["figure"]["data"][0]["x"]) == 24


def test_many_values_are_grouped_as_other():
    df = pd.DataFrame({"city": [f"city_{i % 60}" for i in range(600)]})
    bar = chart(analyze_visualizations(df), "dist_1")["figure"]["data"][0]
    assert len(bar["x"]) == 20 and bar["x"][-1] == "other (41 values)"


def test_long_tailed_histogram_uses_a_log_axis():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"income": np.round(np.exp(rng.normal(np.log(55000), 0.6, 1000)), -2)})
    result = run_all(df)
    hist = next(c for c in result.payload["charts"] if c["kind"] == "histogram")
    assert hist["figure"]["layout"]["xaxis"]["type"] == "log"


def test_total_chart_limit(monkeypatch):
    # Each section has its own cap, so the overall cap is tested by lowering it
    import backend.tools.visualization_analyzer as viz
    monkeypatch.setattr(viz, "CHART_MAX_TOTAL", 10)
    ids = [c["id"] for c in run_all(load("shop_orders.csv"), target="returned").payload["charts"]]
    assert ids == ["missing_values", "heatmap", "balance_1"] + [f"rel_{i}" for i in range(1, 8)]


def test_wide_file_charts_stay_within_limits():
    rng = np.random.default_rng(2)
    base = rng.normal(size=300)
    df = pd.DataFrame({f"c{i}": base * (i + 1) + rng.normal(scale=0.01, size=300) for i in range(40)})
    result = run_all(df)
    ids = [c["id"] for c in result.payload["charts"]]
    assert len(ids) <= CHART_MAX_TOTAL and len(set(ids)) == len(ids)
    assert estimate_tokens(result.summary) <= TOOL_SUMMARY_MAX_TOKENS


# --- robustness ---

def test_works_without_other_tools_results():
    df = load("shop_orders.csv")
    kinds = {c["kind"] for c in analyze_visualizations(df).payload["charts"]}
    assert kinds == {"histogram", "bar"}


@pytest.mark.parametrize("make_df", AWKWARD_INPUTS.values(), ids=AWKWARD_INPUTS.keys())
def test_awkward_inputs_succeed_with_strict_json(make_df):
    for result in (analyze_visualizations(make_df()), run_all(make_df())):
        assert result.success, result.error
        assert result.payload["skipped"] == []
        json.dumps(result.payload, allow_nan=False)
        assert estimate_tokens(result.summary) <= TOOL_SUMMARY_MAX_TOKENS


# --- parse_date_column (planner) ---

def test_parse_date_column_follows_day_first_setting():
    series = pd.Series(["05/01/2024", "06/01/2024", None])
    assert parse_date_column(series, dayfirst=True).tolist()[:2] == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-06")]
    assert pd.isna(parse_date_column(series, dayfirst=True).iloc[2])


def test_parse_date_column_keeps_real_dates():
    series = pd.Series(pd.date_range("2024-01-01", periods=3))
    assert parse_date_column(series).equals(series)
