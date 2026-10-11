"""Smoke test for backend/tools/correlation_analyzer.py, using the shop_orders sample file.

Run from the project root:
    python -m scripts.smoke.smoke_correlation_analyzer
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import PAIR_CHECK_MAX_COLUMNS, TOOL_SUMMARY_MAX_TOKENS
from backend.tools.correlation_analyzer import analyze_correlations
from backend.utils.file_handler import load_file
from backend.utils.token_manager import estimate_tokens
from scripts.smoke.common import SmokeRunner

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "sample_data"


def main() -> None:
    runner = SmokeRunner("correlation_analyzer")

    runner.section("Shop orders file, target 'returned' (summary as the LLM will see it)")
    df = load_file((SAMPLE_DIR / "shop_orders.csv").read_bytes(), "shop_orders.csv").df
    result = analyze_correlations(df, target_column="returned")
    print("\n".join(f"         | {line}" for line in result.summary.splitlines()))
    lines = result.summary.splitlines()

    def line_starting(prefix: str) -> str:
        return next((line for line in lines if line.startswith(prefix)), "(missing)")

    runner.check("measured columns", "Relationships: measured 14 of 18 columns (9 number, 3 two-value, 2 category)", lines[0].split(";")[0])
    runner.check(
        "leakage: refund is issued because the order came back",
        "- returned & refund_issued: 1.00 (phi); 'yes' in refund_issued is more common when returned is 'yes' "
        "- involves the target: possible leakage, check this column is known before the outcome",
        line_starting("- returned & refund_issued"),
    )
    runner.check(
        "near-duplicate: same amount in two currencies",
        True,
        line_starting("- order_value & order_value_cad").endswith("(a copy, a calculation, or the same fact recorded twice)"),
    )
    runner.check(
        "negative relationship keeps its sign",
        "- delivery_days & rating: Pearson -0.72, Spearman -0.72; as one rises, the other falls",
        line_starting("- delivery_days & rating"),
    )
    runner.check(
        "U-shape found by the banded check",
        "- outdoor_temp_c & shipping_cost: banded 0.96 vs Pearson 0.06, Spearman 0.07 (a curve such as a U-shape)",
        line_starting("- outdoor_temp_c & shipping_cost"),
    )
    runner.check(
        "one extreme order called out",
        "- basket_items & support_calls: Pearson 0.91, Spearman 0.11; rise together; may be driven by a few extreme values",
        line_starting("- basket_items & support_calls"),
    )
    runner.check(
        "category vs number (eta) with group averages",
        "- region & order_value: 0.85 (eta); average order_value highest for 'north' (116.8), lowest for 'south' (37.22)",
        line_starting("- region & order_value:"),
    )
    runner.check("ID column skipped", "- looks like a row number, ID or evenly spaced sequence: order_id", line_starting("- looks like"))
    runner.check("sparse column pairs skipped", "13 pair(s) not measured: fewer than 30 rows have both values", lines[-1])

    runner.section(f"Wide file: 75,000 rows x 120 columns (only {PAIR_CHECK_MAX_COLUMNS} are compared in pairs)")
    rng = np.random.default_rng(0)
    wide = pd.DataFrame(rng.normal(size=(75000, 100)), columns=[f"n{i}" for i in range(100)])
    for i in range(20):
        wide[f"t{i}"] = rng.integers(0, 3 + i, 75000).astype(str)
    started = time.perf_counter()
    wide_result = analyze_correlations(wide)
    seconds = time.perf_counter() - started
    print(f"         took {seconds:.1f} s | summary tokens: {estimate_tokens(wide_result.summary)}")
    runner.check("columns measured", PAIR_CHECK_MAX_COLUMNS, len(wide_result.payload["measured_columns"]))
    runner.check("summary within limit", True, estimate_tokens(wide_result.summary) <= TOOL_SUMMARY_MAX_TOKENS)
    runner.check("payload strict JSON", True, json.dumps(wide_result.payload, allow_nan=False) is not None)

    runner.finish()


if __name__ == "__main__":
    main()
