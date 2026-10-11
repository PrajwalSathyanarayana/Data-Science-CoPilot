"""Smoke test for backend/tools/missing_value_analyzer.py, using the survey sample file.

Run from the project root:
    python -m scripts.smoke.smoke_missing_value_analyzer
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import TOOL_SUMMARY_MAX_TOKENS
from backend.tools.missing_value_analyzer import analyze_missing_values
from backend.utils.file_handler import load_file
from backend.utils.token_manager import estimate_tokens
from scripts.smoke.common import SmokeRunner

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "sample_data"


def main() -> None:
    runner = SmokeRunner("missing_value_analyzer")

    runner.section("Survey file (summary as the LLM will see it)")
    df = load_file((SAMPLE_DIR / "survey_with_gaps.csv").read_bytes(), "survey_with_gaps.csv").df
    result = analyze_missing_values(df)
    print("\n".join(f"         | {line}" for line in result.summary.splitlines()))
    lines = result.summary.splitlines()

    runner.check("overview", "Missing values: 4 of 7 columns have gaps (561 cells, 20.0% of all cells)", lines[0])
    runner.check(
        "row view ignores the empty fax column",
        "Rows (ignoring 1 fully empty column(s)): 319 of 400 (79.8%) complete; 81 have at least one gap; 0 are entirely empty",
        lines[1],
    )
    runner.check("fully empty column grouped", "1 column(s) are 100% missing: fax_number", lines[2])
    runner.check("'?' counted as possible placeholder", "- workclass: no blanks, 4.0% possible placeholders ('?' x16)", lines[5])
    runner.check(
        "skipped section found: income and employer missing together",
        "- income & employer: 100% of rows missing either are missing both (80 rows)",
        lines[-1],
    )
    runner.check("tiny gap in age kept in the payload", 1, next(c for c in result.payload["columns"] if c["name"] == "age")["missing"])
    runner.check("data unchanged: workclass still has its '?' values", 16, int((df["workclass"] == "?").sum()))

    runner.section(f"Wide file (400 columns, 30% gaps) vs the {TOOL_SUMMARY_MAX_TOKENS}-token limit")
    wide = analyze_missing_values(pd.DataFrame(np.where(np.random.default_rng(0).random((500, 400)) < 0.3, np.nan, 1.0)))
    print(f"         summary tokens: {estimate_tokens(wide.summary)} | last line: {wide.summary.splitlines()[-1]}")
    runner.check("summary within limit", True, estimate_tokens(wide.summary) <= TOOL_SUMMARY_MAX_TOKENS)
    runner.check("payload strict JSON", True, json.dumps(wide.payload, allow_nan=False) is not None)

    runner.finish()


if __name__ == "__main__":
    main()
