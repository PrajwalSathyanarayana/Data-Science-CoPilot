"""Tests for backend/utils/file_handler.py — real sample files plus hand-built edge cases."""
import re
from pathlib import Path

import pandas as pd
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


# --- placeholders in number columns ---

def test_question_marks_in_a_number_column_are_read_as_missing():
    # delivery_days: 3 "?" among 7 numbers; pandas alone would read the whole column as text
    days = ["3", "?", "5", "7", "?", "2", "4", "?", "6", "1"]
    content = ("order,delivery_days\n" + "".join(f"{i},{v}\n" for i, v in enumerate(days))).encode()
    loaded = load_file(content, "orders.csv")
    assert str(loaded.df["delivery_days"].dtype) == "float64"
    assert loaded.df["delivery_days"].isna().sum() == 3
    assert loaded.df["delivery_days"].sum() == 28
    assert loaded.placeholders_read_as_missing == {"delivery_days": {"?": 3}}
    assert loaded.notes() == ["delivery_days: 3 placeholder value(s) read as missing ('?' x3); every other value is a number"]


def test_each_placeholder_is_kept_as_written():
    loaded = load_file(b"x\n1\nunknown\n 2 \n-\nUNKNOWN\n", "x.csv")
    assert loaded.df["x"].isna().sum() == 3 and loaded.df["x"].sum() == 3
    assert loaded.placeholders_read_as_missing == {"x": {"unknown": 1, "-": 1, "UNKNOWN": 1}}


@pytest.mark.parametrize(
    "content",
    [
        b"x\n1\n?\n12 kg net\n", # real text: not a number column
        b"x\n?\nunknown\n",      # only placeholders: nothing to keep as numbers
        b"x\na\nb\n?\n",         # a text column with a "?" answer
        b"x\nTrue\n?\n1\n",      # True is not a number here
    ],
    ids=["real_text", "only_placeholders", "text_column", "boolean_text"],
)
def test_columns_with_real_text_are_left_alone(content):
    loaded = load_file(content, "x.csv")
    assert loaded.placeholders_read_as_missing == {}
    assert not str(loaded.df["x"].dtype).startswith("float")


def test_excel_column_mixing_numbers_and_question_marks(tmp_path):
    path = tmp_path / "mixed.xlsx"
    pd.DataFrame({"rating": [4, "?", 5, 3.5, "?"]}).to_excel(path, index=False)
    loaded = load_file(path.read_bytes(), "mixed.xlsx")
    assert loaded.df["rating"].tolist()[0] == 4.0 and loaded.df["rating"].isna().sum() == 2
    assert loaded.placeholders_read_as_missing == {"rating": {"?": 2}}


# --- numbers written with units ---

UNITS_CSV = (
    "temperature_reading,trip,price,station\n"
    "26C,54km,$98.06,EDM-01\n32F,76mi,$1204.50,EDM-02\n105F,12 km,unknown,EDM-01\n"
    "113F,3.5mi,$5.00,EDM-02\n2C,10km,$7.25,EDM-01\n15C,8 km,$1.10,EDM-02\n"
).encode()


def test_one_unit_is_removed_and_noted():
    loaded = load_file(UNITS_CSV, "trips.csv")
    assert loaded.df["price"].tolist()[:2] == [98.06, 1204.5] and loaded.df["price"].isna().sum() == 1
    assert "price: '$' removed from 5 values (one unit in the column)" in loaded.notes()


def test_units_pint_reads_safely_are_converted_to_the_most_common():
    loaded = load_file(UNITS_CSV, "trips.csv")
    assert loaded.df["trip"].round(3).tolist() == [54.0, 122.31, 12.0, 5.633, 10.0, 8.0]
    assert "trip: converted to km (2 from mi); units read by pint as km = kilometer, mi = mile" in loaded.notes()


def test_the_user_can_choose_the_unit_to_keep():
    loaded = load_file(UNITS_CSV, "trips.csv", unit_choices={"Trip": "mi"})
    assert loaded.df["trip"].round(3).tolist() == [33.554, 76.0, 7.456, 3.5, 6.214, 4.971]


def test_temperature_letters_wait_for_the_planner_then_python_converts():
    loaded = load_file(UNITS_CSV, "trips.csv")
    assert loaded.mixed_units == {"temperature_reading": {"C": 3, "F": 3}}
    assert "temperature_reading: mixed units (C x3, F x3) not converted; kept as text" in loaded.notes()

    # A wrong answer is rejected and nothing changes; a right one is checked by pint and converted by Python
    assert loaded.resolve_units("temperature_reading", {"C": "degC", "F": "farad"}) == "'F' (farad) and 'C' (degree_Celsius) measure different things"
    assert loaded.resolve_units("temperature_reading", {"C": "degC"}) == "the unit names must cover exactly ['C', 'F']"
    assert loaded.resolve_units("temperature_reading", {"C": "degC", "F": "degF"}) is None
    assert loaded.df["temperature_reading"].round(2).tolist() == [26.0, 0.0, 40.56, 45.0, 2.0, 15.0]
    assert loaded.mixed_units == {}
    assert "temperature_reading: converted to C (3 from F); units read by planner as C = degC, F = degF" in loaded.notes()


def test_codes_stay_text():
    loaded = load_file(UNITS_CSV, "trips.csv")
    assert loaded.df["station"].tolist()[:2] == ["EDM-01", "EDM-02"]


@pytest.mark.parametrize(
    "choices, note",
    [
        ({"trip": "ft"}, "trip: unit setting ignored ('ft' is not one of its units (km x4, mi x2))"),
        ({"region": "km"}, "region: unit setting ignored (no column with this name)"),
        ({"station": "km"}, "station: unit setting ignored (the column is not numbers with units)"),
    ],
    ids=["unit_not_in_column", "no_such_column", "not_a_unit_column"],
)
def test_unusable_unit_choices_are_noted(choices, note):
    assert note in load_file(UNITS_CSV, "trips.csv", unit_choices=choices).notes()


def test_clean_files_report_no_conversion():
    assert load_sample("survey_with_gaps.csv").placeholders_read_as_missing == {}  # its "?" are in a text column


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
