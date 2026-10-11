"""Build the small sample files in tests/sample_data/. Used by tests, smoke scripts, and the README "Try it" section.

Values come from formulas or from random.Random with a fixed seed (Python guarantees random() repeats
exactly for the same seed), so re-running produces identical files.
Run from the project root:
    python -m scripts.generate_sample_data
"""
import random
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


def survey_with_gaps() -> pd.DataFrame:
    """Survey-style data with every kind of gap: blanks that occur together, '?' placeholders,
    'unknown' as a real answer, a fully empty column, and a tiny gap."""
    n = 400
    skipped_section = [i % 5 == 0 for i in range(n)]  # 20% of people skipped the employment section
    return pd.DataFrame({
        "respondent_id": [f"R{i:04d}" for i in range(n)],
        "age": [None if i == 13 else 18 + (i % 50) for i in range(n)],
        "income": [None if skipped else 30000 + 250 * (i % 120) for i, skipped in enumerate(skipped_section)],
        "employer": [None if skipped else ["acme", "globex", "initech"][i % 3] for i, skipped in enumerate(skipped_section)],
        "workclass": ["?" if i % 25 == 0 else ["private", "public", "self"][i % 3] for i in range(n)],
        "marital": ["unknown" if i % 40 == 0 else ["single", "married"][i % 2] for i in range(n)],
        "fax_number": [None] * n,
    })


def _uniform(seed: int, n: int, low: float, high: float) -> list:
    """n evenly spread random values between low and high, identical on every run."""
    rng = random.Random(seed)
    return [low + (high - low) * rng.random() for _ in range(n)]


def shop_orders() -> pd.DataFrame:
    """Every kind of column pair for the correlation tool: straight-line, curved and extreme-driven number pairs,
    two-value and category columns, a near-duplicate, a leakage column, an ID, free text, a constant, and a sparse column.
    Each column's noise uses its own seed, so columns are only related where a formula links them."""
    n = 300
    regions = ["north", "east", "west", "south"]
    base_value = {"north": 120, "east": 80, "west": 60, "south": 40}
    segments = ["new", "regular", "vip"]

    region = [regions[i % 4] for i in range(n)]
    order_value = [round(base_value[r] + noise, 2) for r, noise in zip(region, _uniform(1, n, -30, 30))]
    delivery_days = [1 + int(u) for u in _uniform(2, n, 0, 9)]
    returned = ["yes" if d >= 8 or u < 0.08 else "no" for d, u in zip(delivery_days, _uniform(3, n, 0, 1))]
    segment = [segments[int(u)] for u in _uniform(4, n, 0, 3)]
    temp = [round(t, 1) for t in _uniform(5, n, -15, 35)]

    return pd.DataFrame({
        "order_id": [5001 + i for i in range(n)],
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "region": region,
        "order_value": order_value,
        "order_value_cad": [round(v * 1.37, 2) for v in order_value],  # the same amount in another currency
        "delivery_days": delivery_days,
        "rating": [
            round(5.2 - 0.3 * d + noise, 1) if i % 50 else None
            for i, (d, noise) in enumerate(zip(delivery_days, _uniform(6, n, -1.2, 1.2)))
        ],
        "returned": returned,
        # Issued because the order came back: almost a copy of 'returned' (leakage if 'returned' is the target)
        "refund_issued": [("no" if i == 7 else r) for i, r in enumerate(returned)],
        "customer_segment": segment,
        "payment_method": ["cash" if s == "new" or u < 0.2 else "card" for s, u in zip(segment, _uniform(7, n, 0, 1))],
        "outdoor_temp_c": temp,
        # Costly to ship in very cold and very hot weather: a U-shape
        "shipping_cost": [round(5 + 0.01 * (t - 10) ** 2 + noise, 2) for t, noise in zip(temp, _uniform(8, n, 0, 1.5))],
        # Unrelated, except one huge order with many support calls
        "basket_items": [200 if i == 0 else 1 + int(u) for i, u in enumerate(_uniform(9, n, 0, 7))],
        "support_calls": [60 if i == 0 else int(u) for i, u in enumerate(_uniform(10, n, 0, 5))],
        "loyalty_points": [round(u) if i % 15 == 0 else None for i, u in enumerate(_uniform(11, n, 0, 500))],  # only 20 rows filled in
        "notes": [f"order note {i}" for i in range(n)],
        "gift_wrap": ["no"] * n,
    })


def write_excel_if_changed(path: Path, sheets: dict) -> None:
    """Excel files store a "last saved" time, so rewriting identical data still changes the file's bytes.
    Only write when the content differs, so re-running this script doesn't show up as a change in git."""
    if path.exists():
        existing = pd.read_excel(path, sheet_name=None)
        if list(existing) == list(sheets) and all(existing[name].equals(df) for name, df in sheets.items()):
            return
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weather().to_csv(OUT_DIR / "weather_ddmm.csv", index=False)
    sales().to_csv(OUT_DIR / "sales.csv", index=False)
    (OUT_DIR / "messy_headers.csv").write_text(messy_headers_csv(), encoding="utf-8")
    latin1_rows().to_csv(OUT_DIR / "latin1_encoded.csv", index=False, encoding="latin-1")
    semicolon_rows().to_csv(OUT_DIR / "semicolon_separated.csv", index=False, sep=";")
    survey_with_gaps().to_csv(OUT_DIR / "survey_with_gaps.csv", index=False)
    shop_orders().to_csv(OUT_DIR / "shop_orders.csv", index=False)

    # Two sheets: the loader should read the first one and report that there were two
    write_excel_if_changed(OUT_DIR / "hospital.xlsx", {
        "admissions": hospital(),
        "readme": pd.DataFrame({"note": ["Synthetic data for testing"]}),
    })

    for path in sorted(OUT_DIR.glob("*.*")):
        if path.suffix != ".gitkeep" and path.name != ".gitkeep":
            print(f"wrote {path.relative_to(OUT_DIR.parent.parent)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
