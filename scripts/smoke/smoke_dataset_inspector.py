"""Smoke test for backend/tools/dataset_inspector.py.

Run from the project root:
    python -m scripts.smoke.smoke_dataset_inspector
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import TOOL_SUMMARY_MAX_TOKENS
from backend.tools.dataset_inspector import inspect_dataset
from backend.utils.file_handler import load_file
from backend.utils.token_manager import estimate_tokens
from scripts.smoke.common import SmokeRunner

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "sample_data"


def is_strict_json(payload: dict) -> bool:
    try:
        json.dumps(payload, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


def main() -> None:
    runner = SmokeRunner("dataset_inspector")

    runner.section("Hospital Excel file (summary as the LLM will see it)")
    df = load_file((SAMPLE_DIR / "hospital.xlsx").read_bytes(), "hospital.xlsx").df
    result = inspect_dataset(df)
    print("\n".join(f"         | {line}" for line in result.summary.splitlines()))
    runner.check(
        "summary",
        [
            "Dataset: 170 rows x 4 columns (1 date, 3 numeric)",
            "- admitted (date): 2021-06-01 to 2021-11-17",
            "- age (numeric): mean 48.03, median 48, min 20, max 79",
            "- stay_days (numeric): mean 4.588, median 1, min 1, max 50",
            "- readmitted (numeric): mean 0.1176, median 0, min 0, max 1",
        ],
        result.summary.splitlines(),
    )
    runner.check("payload is strict JSON (no NaN, no date objects)", True, is_strict_json(result.payload))

    runner.section("Tricky values")
    tricky = pd.DataFrame({
        "tiny": [0.00012, 0.00034, np.nan, 0.0002],
        "with_inf": [1.0, 2.0, np.inf, 3.0],
        "flag": [True, False, True, True],
    })
    lines = inspect_dataset(tricky).summary.splitlines()
    runner.check("small numbers not rounded to zero", "- tiny (numeric): mean 0.00022, median 0.0002, min 0.00012, max 0.00034, 25% missing", lines[1])
    runner.check("infinity counted, not averaged", "- with_inf (numeric): mean 2, median 2, min 1, max 3, 1 infinite values", lines[2])
    runner.check("true/false share", "- flag (boolean): 75% true", lines[3])

    runner.section("Tiny file (2 rows)")
    tiny = inspect_dataset(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    runner.check("works, keeps both rows as examples", (True, 2), (tiny.success, len(tiny.payload["sample_rows"]["rows"])))

    runner.section(f"Wide file (300 columns) vs the {TOOL_SUMMARY_MAX_TOKENS}-token summary limit")
    wide = inspect_dataset(pd.DataFrame(np.random.default_rng(0).random((10, 300))))
    wide_lines = wide.summary.splitlines()
    print(f"         summary tokens: {estimate_tokens(wide.summary)} | last line: {wide_lines[-1]}")
    runner.check("summary within limit", True, estimate_tokens(wide.summary) <= TOOL_SUMMARY_MAX_TOKENS)
    runner.check("overview line first", "Dataset: 10 rows x 300 columns (300 numeric)", wide_lines[0])
    runner.check("payload still has all 300 columns", 300, len(wide.payload["columns"]))

    runner.finish()


if __name__ == "__main__":
    main()
