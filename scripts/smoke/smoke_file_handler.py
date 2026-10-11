"""Smoke test for backend/utils/file_handler.py, using the committed sample files.

Run from the project root (regenerate the files first if they are missing):
    python -m scripts.generate_sample_data
    python -m scripts.smoke.smoke_file_handler
"""
from pathlib import Path

from backend.config import MAX_FILE_SIZE
from backend.utils.file_handler import FileLoadError, load_file
from scripts.smoke.common import SmokeRunner

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "sample_data"


def load(name: str):
    return load_file((SAMPLE_DIR / name).read_bytes(), name)


def error_message(file_bytes: bytes, filename: str) -> str:
    try:
        load_file(file_bytes, filename)
        return "no error raised"
    except FileLoadError as exc:
        return str(exc)


def main() -> None:
    runner = SmokeRunner("file_handler")

    runner.section("Day-first weather CSV")
    weather = load("weather_ddmm.csv")
    runner.check("rows x columns", (340, 5), weather.df.shape)
    runner.check("separator and encoding", (",", "utf-8"), (weather.delimiter, weather.encoding))

    runner.section("Semicolon-separated CSV (like UCI Bank Marketing)")
    semicolon = load("semicolon_separated.csv")
    runner.check("separator detected", ";", semicolon.delimiter)
    runner.check("columns split correctly", ["age", "job", "balance", "subscribed"], list(semicolon.df.columns))

    runner.section("Non-UTF-8 CSV (accented names)")
    latin1 = load("latin1_encoded.csv")
    runner.check("encoding guessed", "windows-1252", latin1.encoding)
    runner.check(
        "accents decoded (shown as escape codes so any terminal can display them)",
        ascii(["Montréal", "Québec", "Zürich", "São Paulo", "Malmö"]),
        ascii(latin1.df["city"].unique().tolist()),
    )

    runner.section("Messy headers")
    messy = load("messy_headers.csv")
    runner.check(
        "names trimmed, lowercased, made unique",
        ["amount", "amount_3", "column_3", "date", "amount_2", "region"],
        list(messy.df.columns),
    )
    runner.check("renames recorded for the report", {"amount": "amount_3", "Unnamed: 2": "column_3"}, messy.renamed_columns)

    runner.section("Excel workbook with two sheets")
    hospital = load("hospital.xlsx")
    runner.check("first sheet read, sheet count noted", ("admissions", 2), (hospital.sheet_name, hospital.sheet_count))
    runner.check("rows x columns", (170, 4), hospital.df.shape)

    runner.section("Every row kept on a large file")
    big = load_file(("value\n" + "\n".join(str(i) for i in range(150_000))).encode(), "big.csv")
    runner.check("150,000 rows in, 150,000 rows out (no sampling)", 150_000, len(big.df))

    runner.section("Rejected uploads")
    too_big = b"a\n" + b"1\n" * (MAX_FILE_SIZE // 2 + 1)
    runner.check(
        "over the 50 MB limit",
        f"The file is {len(too_big):,} bytes; the limit is {MAX_FILE_SIZE:,} bytes.",
        error_message(too_big, "huge.csv"),
    )
    runner.check("header but no rows", "The file has column headers but no data rows.", error_message(b"a,b\n", "x.csv"))
    runner.check(
        "wrong file type",
        "Unsupported file type 'pdf'. Allowed types: ['csv', 'xlsx', 'xls']",
        error_message(b"%PDF-1.7", "report.pdf"),
    )

    runner.finish()


if __name__ == "__main__":
    main()
