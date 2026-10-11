"""Smoke test for backend/tools/outlier_analyzer.py, using example customer and employee uploads and the shop_orders sample file.

Run from the project root:
    python -m scripts.smoke.smoke_outlier_analyzer
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import TOOL_SUMMARY_MAX_TOKENS
from backend.tools.outlier_analyzer import analyze_outliers
from backend.utils.file_handler import load_file
from backend.utils.token_manager import estimate_tokens
from scripts.smoke.common import SmokeRunner
from tests.test_outlier_analyzer import customers, employees

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "sample_data"


def show(summary: str) -> None:
    print("\n".join(f"         | {line}" for line in summary.splitlines()))


def line_for(summary: str, name: str) -> str:
    return next((line for line in summary.splitlines() if line.startswith(f"- {name}:")), "(missing)")


def main() -> None:
    runner = SmokeRunner("outlier_analyzer")

    runner.section("customers.csv: long-tailed income, two ages typed as 999, discounts mostly 0 plus one 95")
    result = analyze_outliers(customers())
    show(result.summary)
    runner.check(
        "typos shown next to a real 76-year-old",
        "- age: 5 high (0.5%), normal range 5.5 to 73.5, highest: 999, 999, 76",
        line_for(result.summary, "age"),
    )
    runner.check(
        "income judged on the log scale (normal high earners not flagged)",
        True,
        line_for(result.summary, "income").startswith("- income: 4 high and 4 low (0.8%), normal range 11,904 to 237,503, log scale"),
    )
    runner.check(
        "mostly-zero discounts checked among the rest: the 95 is found",
        "- discount_pct: 68% are 0; checked among the other 315 values, 1 high (0.32%), normal range -7 to 41, highest: 95",
        line_for(result.summary, "discount_pct"),
    )

    runner.section("employees.csv: 5 people aged ~25 with ~20 years' experience")
    result = analyze_outliers(employees())
    show(result.summary)
    check = result.payload["row_check"]
    runner.check("impossible rows found", [2001, 2002, 2003, 2004, 2005], sorted(r["row"] for r in check["rows"]))
    runner.check("column adding most to them", {"age": 5}, check["main_columns"])

    runner.section("shop_orders.csv: one 200-item order, one 60-call order, a U-shaped cost")
    df = load_file((SAMPLE_DIR / "shop_orders.csv").read_bytes(), "shop_orders.csv").df
    result = analyze_outliers(df)
    show(result.summary)
    runner.check("200-item order found in a few-value column", [200], next(c for c in result.payload["columns"] if c["name"] == "basket_items")["highest"])
    runner.check("rows flagged because of the U-shape get a caution", True, result.payload["row_check"]["warning"])

    runner.section("Large file: 75,000 rows x 40 columns")
    big = pd.DataFrame(np.random.default_rng(0).normal(size=(75000, 40)), columns=[f"n{i}" for i in range(40)])
    started = time.perf_counter()
    big_result = analyze_outliers(big)
    print(f"         took {time.perf_counter() - started:.1f} s | summary tokens: {estimate_tokens(big_result.summary)}")
    runner.check("summary within limit", True, estimate_tokens(big_result.summary) <= TOOL_SUMMARY_MAX_TOKENS)
    runner.check("payload strict JSON", True, json.dumps(big_result.payload, allow_nan=False) is not None)

    runner.finish()


if __name__ == "__main__":
    main()
