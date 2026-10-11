"""Utility — reads an uploaded CSV/XLSX/XLS into a pandas DataFrame, keeping every row, and records what loading did."""
import csv
import io
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import chardet
import pandas as pd

from ..config import ALLOWED_FILE_TYPE, CSV_SNIFF_BYTES, ENCODING_SAMPLE_BYTES, MAX_FILE_SIZE

logger = logging.getLogger(__name__)

CSV_DELIMITERS = ",;\t|"
EXCEL_ENGINES = {"xlsx": "openpyxl", "xls": "xlrd"}
# pandas names a column with a blank header "Unnamed: 3"
PANDAS_BLANK_HEADER = re.compile(r"^unnamed: \d+$")


class FileLoadError(ValueError):
    """The upload can't be turned into a table. The message is safe to show to the user."""


@dataclass
class LoadedFile:
    """The loaded table plus notes about how it was read, for the report and for debugging."""

    df: pd.DataFrame
    file_type: str
    original_columns: List[str]
    renamed_columns: Dict[str, str] = field(default_factory=dict)  # original -> new, only beyond trim/lowercase
    encoding: Optional[str] = None      # CSV only
    delimiter: Optional[str] = None     # CSV only
    sheet_name: Optional[str] = None    # Excel only
    sheet_count: Optional[int] = None   # Excel only


def _file_type(filename: str) -> str:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_FILE_TYPE:
        raise FileLoadError(f"Unsupported file type '{extension or 'none'}'. Allowed types: {ALLOWED_FILE_TYPE}")
    return extension


def _decode(file_bytes: bytes) -> Tuple[str, str]:
    """Turn bytes into text. Returns the text and the encoding used."""
    # Most files are UTF-8. "utf-8-sig" also strips the marker Excel adds when saving "CSV UTF-8".
    try:
        return file_bytes.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        pass

    # Guess from the start of the file only: scanning all 50 MB with chardet can take minutes
    guess = chardet.detect(file_bytes[:ENCODING_SAMPLE_BYTES])["encoding"]
    try:
        return file_bytes.decode(guess), guess.lower()
    except (TypeError, LookupError, UnicodeDecodeError):
        # latin-1 maps every byte to a character, so it never fails (some symbols may look odd)
        return file_bytes.decode("latin-1"), "latin-1"


def _detect_delimiter(text: str) -> str:
    """Find the column separator from the start of the file. Falls back to a comma."""
    try:
        return csv.Sniffer().sniff(text[:CSV_SNIFF_BYTES], delimiters=CSV_DELIMITERS).delimiter
    except csv.Error:
        # Sniffer gives up on e.g. single-column files; a comma is the standard
        return ","


def _read_csv(file_bytes: bytes) -> Tuple[pd.DataFrame, str, str]:
    text, encoding = _decode(file_bytes)
    delimiter = _detect_delimiter(text)
    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
    except pd.errors.EmptyDataError as exc:
        raise FileLoadError("The file has no data.") from exc
    except Exception as exc:
        # Untrusted input: any parser failure becomes one clear, user-facing error
        raise FileLoadError(f"Could not read the file as CSV: {exc}") from exc
    return df, encoding, delimiter


def _read_excel(file_bytes: bytes, file_type: str) -> Tuple[pd.DataFrame, str, int]:
    try:
        workbook = pd.ExcelFile(io.BytesIO(file_bytes), engine=EXCEL_ENGINES[file_type])
        sheet_name = workbook.sheet_names[0]
        df = workbook.parse(sheet_name)
    except Exception as exc:
        # Damaged files, wrong formats, and password-protected workbooks all end up here
        raise FileLoadError(f"Could not read the file as Excel ({file_type}): {exc}") from exc
    return df, sheet_name, len(workbook.sheet_names)


def _clean_column_names(columns: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Trim and lowercase names; give blank names a placeholder and make every name unique.

    Returns the new names and {original: new} for names changed beyond trim/lowercase.
    """
    base_names = []
    for position, column in enumerate(columns):
        name = str(column).strip().lower()
        if not name or PANDAS_BLANK_HEADER.match(name):
            name = f"column_{position + 1}"
        base_names.append(name)

    # Suffixes must not collide with any name that exists anywhere in the file
    taken = set()
    reserved = set(base_names)
    new_names, renamed = [], {}
    for column, name in zip(columns, base_names):
        if name in taken:
            suffix = 2
            while f"{name}_{suffix}" in taken or f"{name}_{suffix}" in reserved:
                suffix += 1
            name = f"{name}_{suffix}"
        taken.add(name)
        new_names.append(name)
        if name != str(column).strip().lower():
            renamed[str(column)] = name
    return new_names, renamed


def load_file(file_bytes: bytes, filename: str) -> LoadedFile:
    """Read an uploaded CSV, XLSX, or XLS file into a DataFrame. Every row is kept.

    Raises:
        FileLoadError: unsupported type, empty, too large, unreadable, or no data rows.
    """
    file_type = _file_type(filename)

    # The route also checks the size; checking here too keeps this function safe on its own
    if not file_bytes:
        raise FileLoadError("The file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise FileLoadError(f"The file is {len(file_bytes):,} bytes; the limit is {MAX_FILE_SIZE:,} bytes.")

    encoding = delimiter = sheet_name = sheet_count = None
    if file_type == "csv":
        df, encoding, delimiter = _read_csv(file_bytes)
    else:
        df, sheet_name, sheet_count = _read_excel(file_bytes, file_type)

    if df.shape[1] == 0:
        raise FileLoadError("The file has no columns.")
    if df.shape[0] == 0:
        raise FileLoadError("The file has column headers but no data rows.")

    original_columns = [str(column) for column in df.columns]
    df.columns, renamed = _clean_column_names(list(df.columns))
    if renamed:
        logger.info("Renamed %d column(s) while loading %s: %s", len(renamed), filename, renamed)

    return LoadedFile(
        df=df,
        file_type=file_type,
        original_columns=original_columns,
        renamed_columns=renamed,
        encoding=encoding,
        delimiter=delimiter,
        sheet_name=sheet_name,
        sheet_count=sheet_count,
    )
