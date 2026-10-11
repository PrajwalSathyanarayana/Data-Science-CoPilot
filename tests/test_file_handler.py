"""Tests for backend/utils/file_handler.py — real sample files plus hand-built edge cases."""
import re
from pathlib import Path

import pytest

from backend.utils import file_handler
from backend.utils.file_handler import FileLoadError, load_file

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def load_sample(name: str):
    return load_file((SAMPLE_DIR / name).read_bytes(), name)


# --- the committed sample files ---

@pytest.mark.parametrize(
    "name, shape, columns, encoding, delimiter",
    [
        ("weather_ddmm.csv", (340, 5), ["observed_on", "station", "temp_c", "rainfall_mm", "storm_warning"], "utf-8", ","),
        ("sales.csv", (200, 4), ["order_date", "customer_id", "order_value", "region"], "utf-8", ","),
        ("semicolon_separated.csv", (40, 4), ["age", "job", "balance", "subscribed"], "utf-8", ";"),
        ("latin1_encoded.csv", (20, 2), ["city", "visits"], "windows-1252", ","),
        ("survey_with_gaps.csv", (400, 7),
         ["respondent_id", "age", "income", "employer", "workclass", "marital", "fax_number"], "utf-8", ","),
    ],
)
def test_sample_csv_files_load_correctly(name, shape, columns, encoding, delimiter):
    loaded = load_sample(name)

    assert loaded.df.shape == shape
    assert list(loaded.df.columns) == columns
    assert (loaded.file_type, loaded.encoding, loaded.delimiter) == ("csv", encoding, delimiter)
    assert loaded.sheet_name is None


def test_excel_reads_first_sheet_and_reports_sheet_count():
    loaded = load_sample("hospital.xlsx")

    assert loaded.df.shape == (170, 4)
    assert list(loaded.df.columns) == ["admitted", "age", "stay_days", "readmitted"]
    assert (loaded.file_type, loaded.sheet_name, loaded.sheet_count) == ("xlsx", "admissions", 2)
    assert loaded.encoding is None and loaded.delimiter is None


def test_non_utf8_text_is_decoded_correctly():
    cities = load_sample("latin1_encoded.csv").df["city"].unique().tolist()
    assert cities == ["Montréal", "Québec", "Zürich", "São Paulo", "Malmö"]


def test_messy_headers_are_cleaned_and_made_unique():
    loaded = load_sample("messy_headers.csv")

    # " Amount " and "amount" collide; "amount_2" already exists, so the clash becomes "amount_3"
    assert list(loaded.df.columns) == ["amount", "amount_3", "column_3", "date", "amount_2", "region"]
    assert loaded.renamed_columns == {"amount": "amount_3", "Unnamed: 2": "column_3"}
    assert loaded.original_columns == [" Amount ", "amount", "Unnamed: 2", "Date ", "amount_2", "Region"]


# --- every row is kept ---

def test_large_file_keeps_every_row():
    csv_bytes = ("value\n" + "\n".join(str(i) for i in range(150_000))).encode()
    loaded = load_file(csv_bytes, "big.csv")

    assert len(loaded.df) == 150_000
    assert loaded.df["value"].sum() == sum(range(150_000))


# --- CSV variations ---

@pytest.mark.parametrize(
    "content, delimiter",
    [
        ("a\tb\n1\t2\n3\t4\n", "\t"),
        ("a|b\n1|2\n3|4\n", "|"),
        ('name;note\nA;"x, y"\nB;"z, w"\n', ";"),  # commas inside quoted text must not fool detection
    ],
    ids=["tab", "pipe", "semicolon_with_quoted_commas"],
)
def test_other_separators_are_detected(content, delimiter):
    loaded = load_file(content.encode(), "data.csv")

    assert loaded.delimiter == delimiter
    assert loaded.df.shape == (2, 2)


def test_single_column_file_falls_back_to_comma():
    loaded = load_file(b"value\n1\n2\n3\n", "one.csv")
    assert (loaded.delimiter, loaded.df.shape) == (",", (3, 1))


def test_utf8_byte_order_mark_is_not_part_of_the_first_column_name():
    loaded = load_file("﻿name,score\nAda,9\n".encode("utf-8"), "bom.csv")
    assert list(loaded.df.columns) == ["name", "score"]


def test_uppercase_extension_is_accepted():
    assert load_file(b"a,b\n1,2\n", "DATA.CSV").df.shape == (1, 2)


# --- column name cleanup ---

def test_blank_header_names_get_placeholders():
    loaded = load_file(b",x,,x\n1,2,3,4\n", "blanks.csv")
    assert list(loaded.df.columns) == ["column_1", "x", "column_3", "x.1"]  # pandas itself renames exact duplicates


# --- rejected uploads ---

@pytest.mark.parametrize(
    "file_bytes, filename, message",
    [
        (b"", "empty.csv", "The file is empty"),
        (b"   \n\n", "blank.csv", "no data"),
        (b"a,b,c\n", "header_only.csv", "no data rows"),
        (b"a,b\n1,2\n", "notes.txt", "Unsupported file type 'txt'"),
        (b"a,b\n1,2\n", "no_extension", "Unsupported file type 'none'"),
        (b"this is not a spreadsheet", "fake.xlsx", "Could not read the file as Excel (xlsx)"),
        (b"this is not a spreadsheet", "fake.xls", "Could not read the file as Excel (xls)"),
        (b"a,b\n1,2\n3,4,5,6\n", "ragged.csv", "Could not read the file as CSV"),
    ],
    ids=["empty", "whitespace_only", "header_only", "wrong_type", "no_extension", "bad_xlsx", "bad_xls", "ragged_rows"],
)
def test_unusable_uploads_raise_clear_errors(file_bytes, filename, message):
    # re.escape: messages contain "(xlsx)", which regex would read as a group
    with pytest.raises(FileLoadError, match=re.escape(message)):
        load_file(file_bytes, filename)


def test_size_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(file_handler, "MAX_FILE_SIZE", 10)
    with pytest.raises(FileLoadError, match="the limit is 10 bytes"):
        load_file(b"a,b\n1,2\n3,4\n", "data.csv")


def test_file_load_error_is_a_value_error():
    # Code that already catches ValueError keeps working
    assert issubclass(FileLoadError, ValueError)
