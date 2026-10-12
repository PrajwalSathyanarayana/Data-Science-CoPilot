"""Smoke test for backend/tools/visualization_analyzer.py: the analysis tools run first, then the charts.

Run from the project root:
    python -m scripts.smoke.smoke_visualization_analyzer
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backend.agents.planner import detect_interesting_columns
from backend.tools.correlation_analyzer import analyze_correlations
from backend.tools.missing_value_analyzer import analyze_missing_values
from backend.tools.outlier_analyzer import analyze_outliers
from backend.tools.visualization_analyzer import analyze_visualizations
from backend.utils.file_handler import load_file
from scripts.smoke.common import SmokeRunner

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "sample_data"


def run_all(df: pd.DataFrame, target=None):
    flags = detect_interesting_columns(df, target_column=target)
    results = (analyze_missing_values(df), analyze_correlations(df, target), analyze_outliers(df, target))
    payloads = {r.tool_name: r.payload for r in results if r.success}
    return analyze_visualizations(df, payloads, target_column=target, column_flags=flags)


def main() -> None:
    runner = SmokeRunner("visualization_analyzer")

    runner.section("shop_orders.csv, target 'returned' (chart list as the LLM will see it)")
    df = load_file((SAMPLE_DIR / "shop_orders.csv").read_bytes(), "shop_orders.csv").df
    result = run_all(df, target="returned")
    print("\n".join(f"         | {line}" for line in result.summary.splitlines()[:8]) + "\n         | ...")
    charts = {c["id"]: c for c in result.payload["charts"]}
    size_kb = len(json.dumps(result.payload)) / 1000
    print(f"         whole chart payload: {size_kb:.0f} KB")

    runner.check("number of charts", 34, len(charts))
    runner.check("report under 60 KB", True, size_kb < 60)
    runner.check(
        "leakage pair chart caption",
        "1.00 (phi); 'yes' in refund_issued is more common when returned is 'yes'; near-duplicate involving the target: possible leakage",
        charts["rel_2"]["caption"],
    )
    flagged = next(t for t in charts["rel_5"]["figure"]["data"] if t["name"] == "flagged outliers")
    runner.check("200-item order marked on the scatter", ([200.0], [60.0]), (flagged["x"], flagged["y"]))
    runner.check("U-shape drawn with band averages", "average per band", charts["rel_12"]["figure"]["data"][-1]["name"])
    runner.check("dates stored as text get a time chart", "300 dated rows from 2024-01-01 to 2024-10-26, counted per week", charts["time_1"]["caption"])

    runner.section("Large file: 75,000 rows, one extreme row")
    rng = np.random.default_rng(0)
    big = pd.DataFrame({"x": rng.normal(size=75000)})
    big["y"] = 2 * big["x"] + rng.normal(size=75000)
    big.loc[0, ["x", "y"]] = [40.0, 90.0]
    started = time.perf_counter()
    big_result = run_all(big)
    scatter = next(c for c in big_result.payload["charts"] if c["kind"] == "scatter")
    print(f"         all tools + charts took {time.perf_counter() - started:.1f} s | chart payload {len(json.dumps(big_result.payload)) / 1000:.0f} KB")
    runner.check("scatter is a labelled sample", {"shown": 2000, "of": 75000}, scatter["sampled"])
    runner.check("extreme row still shown", True, 40.0 in scatter["figure"]["data"][1]["x"])

    runner.finish()


if __name__ == "__main__":
    main()
