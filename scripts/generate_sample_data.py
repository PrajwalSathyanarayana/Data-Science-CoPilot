"""Build the small sample files in tests/sample_data/. Used by tests, smoke scripts, and the README "Try it" section.

Every value is computed from a formula (no randomness), so re-running produces identical files.
Run from the project root:
    python -m scripts.generate_sample_data
"""
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "sample_data"

# Mostly small values with a few very large ones: a long right tail
LONG_TAIL = [1] * 50 + [2] * 20 + [5] * 10 + [50] * 5


def weather() -> pd.DataFrame:
    """Day-first dates, a long-tailed column, a rare event column."""
    n = 340
    return pd.DataFrame({
        "observed_on": pd.date_range("2022-01-01", periods=n, freq="D").strftime("%d/%m/%Y"),
        "station": ["EDM-01", "EDM-02"] * (n // 2),
        "temp_c": [(i % 41) - 20 for i in range(n)],
        "rainfall_mm": LONG_TAIL * 4,
        "storm_warning": ["yes"] * 14 + ["no"] * (n - 14),
    })


def sales() -> pd.DataFrame:
    """ISO dates, numeric IDs stored as text, a long-tailed value column."""
    n = 200
    return pd.DataFrame({
        "order_date": pd.date_range("2024-03-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "customer_id": [f"{10000 + i}" for i in range(n)],
        "order_value": [round(1.02 ** i, 2) for i in range(n)],
        "region": ["north", "south"] * (n // 2),
    })


def hospital() -> pd.DataFrame:
    """Real dates, a long-tailed stay length, and a 0/1 target."""
    n = 170
    return pd.DataFrame({
        "admitted": pd.date_range("2021-06-01", periods=n, freq="D"),
        "age": [20 + (i % 60) for i in range(n)],
        "stay_days": LONG_TAIL * 2,
        "readmitted": [1] * 20 + [0] * (n - 20),
    })


def messy_headers_csv() -> str:
    """Header problems seen in real exports: padding, case clashes, a blank name, a name that looks like a suffix."""
    header = " Amount ,amount,,Date ,amount_2,Region"
    rows = [f"{10 * i},{i},{i % 3},2024-01-{i:02d},{i * 2},{'east' if i % 2 else 'west'}" for i in range(1, 21)]
    return "\n".join([header, *rows]) + "\n"


def latin1_rows() -> pd.DataFrame:
    """Accented names that are not valid UTF-8 once saved as latin-1."""
    cities = ["Montréal", "Québec", "Zürich", "São Paulo", "Malmö"]
    return pd.DataFrame({"city": cities * 4, "visits": list(range(20))})


def semicolon_rows() -> pd.DataFrame:
    """Shaped like UCI Bank Marketing exports, which separate columns with ';'."""
    n = 40
    return pd.DataFrame({
        "age": [25 + (i % 30) for i in range(n)],
        "job": ["admin", "technician", "services", "management"] * (n // 4),
        "balance": [100 * (i % 7) for i in range(n)],
        "subscribed": ["yes"] * 5 + ["no"] * (n - 5),
    })


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weather().to_csv(OUT_DIR / "weather_ddmm.csv", index=False)
    sales().to_csv(OUT_DIR / "sales.csv", index=False)
    (OUT_DIR / "messy_headers.csv").write_text(messy_headers_csv(), encoding="utf-8")
    latin1_rows().to_csv(OUT_DIR / "latin1_encoded.csv", index=False, encoding="latin-1")
    semicolon_rows().to_csv(OUT_DIR / "semicolon_separated.csv", index=False, sep=";")

    # Two sheets: the loader should read the first one and report that there were two
    with pd.ExcelWriter(OUT_DIR / "hospital.xlsx", engine="openpyxl") as writer:
        hospital().to_excel(writer, sheet_name="admissions", index=False)
        pd.DataFrame({"note": ["Synthetic data for testing"]}).to_excel(writer, sheet_name="readme", index=False)

    for path in sorted(OUT_DIR.glob("*.*")):
        if path.suffix != ".gitkeep" and path.name != ".gitkeep":
            print(f"wrote {path.relative_to(OUT_DIR.parent.parent)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
