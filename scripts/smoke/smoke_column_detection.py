"""Smoke test for detect_interesting_columns() in backend/agents/planner.py.

Runs the same detection on three unrelated domains (weather, sales, hospital) to show
that every check is driven by the values, never by column names.

Run from the project root:
    python -m scripts.smoke.smoke_column_detection
"""
from typing import Dict, List, Tuple

import pandas as pd

from backend.agents.planner import detect_interesting_columns
from scripts.smoke.common import SmokeRunner

# A long right tail: mostly small values, a few very large ones
LONG_TAIL = [1] * 50 + [2] * 20 + [5] * 10 + [50] * 5   # 85 values


def weather_dataset() -> pd.DataFrame:
    n = 340
    return pd.DataFrame({
        "observed_on": pd.date_range("2022-01-01", periods=n, freq="D").strftime("%d/%m/%Y"),
        "station": ["EDM-01", "EDM-02"] * (n // 2),               # balanced two-class: nothing
        "temp_c": [(i % 41) - 20 for i in range(n)],              # symmetric: nothing
        "rainfall_mm": LONG_TAIL * 4,                             # long tail
        "storm_warning": ["yes"] * 14 + ["no"] * (n - 14),        # 4.1% minority
    })


def sales_dataset() -> pd.DataFrame:
    n = 200
    return pd.DataFrame({
        "order_date": pd.date_range("2024-03-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "customer_id": [str(10000 + i) for i in range(n)],       # numbers stored as text: not a date
        "order_value": [round(1.02 ** i, 2) for i in range(n)],  # long tail
        "region": ["north", "south"] * (n // 2),                  # balanced: nothing
    })


def hospital_dataset() -> pd.DataFrame:
    n = 170
    return pd.DataFrame({
        "admitted": pd.date_range("2021-06-01", periods=n, freq="D"),  # real datetime type
        "age": [20 + (i % 60) for i in range(n)],                      # roughly uniform: nothing
        "stay_days": LONG_TAIL * 2,                                    # long tail
        "readmitted": [1] * 20 + [0] * (n - 20),                       # 0/1 target, 11.8% minority
    })


def found(flags) -> List[Tuple[str, str]]:
    return [(f.kind, f.column) for f in flags]


def details(flags) -> Dict[str, str]:
    return {f.column: f.detail for f in flags}


def main() -> None:
    runner = SmokeRunner("column detection")

    runner.section("Weather data")
    flags = detect_interesting_columns(weather_dataset())
    runner.check(
        "flags found (kind, column)",
        [
            ("date_column", "observed_on"),
            ("skewed_distribution", "rainfall_mm"),
            ("class_imbalance", "storm_warning"),
        ],
        found(flags),
    )
    runner.check(
        "dd/mm/yyyy dates, range from every row",
        "text column where 100% of values parse as dates, ranging from 2022-01-01 to 2022-12-06",
        details(flags).get("observed_on"),
    )
    runner.check(
        "rainfall long tail",
        "long right tail (skew 3.69): mean 4.59 vs median 1.00, range 1.00 to 50.00",
        details(flags).get("rainfall_mm"),
    )

    runner.section("Sales data")
    flags = detect_interesting_columns(sales_dataset())
    runner.check(
        "flags found (kind, column)",
        [("date_column", "order_date"), ("skewed_distribution", "order_value")],
        found(flags),
    )
    runner.check("customer_id (numbers as text) not flagged as a date", False, "customer_id" in details(flags))

    runner.section("Hospital data, with target column")
    flags = detect_interesting_columns(hospital_dataset(), target_column="readmitted")
    runner.check(
        "flags found (kind, column)",
        [
            ("date_column", "admitted"),
            ("skewed_distribution", "stay_days"),
            ("class_imbalance", "readmitted"),
        ],
        found(flags),
    )
    runner.check(
        "readmitted labelled as the target",
        "target column: minority class '1' is 11.8% of rows (20 of 170)",
        details(flags).get("readmitted"),
    )

    runner.section("Day/month order (date_dayfirst setting)")
    # Every day here is 12 or less, so 05/02/2022 could be 5 Feb or 2 May
    ambiguous = pd.DataFrame({"reading_date": [f"{d:02d}/{m:02d}/2022" for m in (1, 2, 3) for d in range(1, 13)]})
    runner.check(
        "no setting: month-first assumed, and the detail says so",
        "text column where 100% of values parse as dates, ranging from 2022-01-01 to 2022-12-03; "
        "day/month order is ambiguous (every day is 12 or less), assumed month-first",
        details(detect_interesting_columns(ambiguous)).get("reading_date"),
    )
    runner.check(
        "date_dayfirst=True: read as dd/mm, correct range",
        "text column where 100% of values parse as dates, ranging from 2022-01-01 to 2022-03-12",
        details(detect_interesting_columns(ambiguous, date_dayfirst=True)).get("reading_date"),
    )
    iso = pd.DataFrame({"reading_date": pd.date_range("2022-01-01", periods=60, freq="D").strftime("%Y-%m-%d")})
    runner.check(
        "date_dayfirst=True does not corrupt year-first dates",
        "text column where 100% of values parse as dates, ranging from 2022-01-01 to 2022-03-01",
        details(detect_interesting_columns(iso, date_dayfirst=True)).get("reading_date"),
    )

    runner.section("Bad input")
    try:
        detect_interesting_columns(sales_dataset(), target_column="does_not_exist")
        error = "no error raised"
    except ValueError as exc:
        error = str(exc)
    runner.check(
        "unknown target column",
        "target_column 'does_not_exist' is not a column in the dataset",
        error,
    )

    runner.finish()


if __name__ == "__main__":
    main()
