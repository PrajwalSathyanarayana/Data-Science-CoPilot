"""Utility — reads an uploaded CSV/XLSX/XLS into a pandas DataFrame, keeping every row, and records what loading did."""
import csv
import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import chardet
import pandas as pd

from ..config import ALLOWED_FILE_TYPE, CSV_SNIFF_BYTES, ENCODING_SAMPLE_BYTES, MAX_FILE_SIZE
from .units import TextNumbers, UnitError, conversion_factors, most_common_unit, pint_meanings, read_text_numbers

logger = logging.getLogger(__name__)

CSV_DELIMITERS = ",;\t|"
EXCEL_ENGINES = {"xlsx": "openpyxl", "xls": "xlrd"}
# pandas names a column with a blank header "Unnamed: 3"
PANDAS_BLANK_HEADER = re.compile(r"^unnamed: \d+$")
UNIT_DECIMALS = 10  # converted units are rounded to this many decimal places


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
    # Number columns whose placeholders ("?", "unknown", ...) were read as missing: {column: {value as written: count}}
    placeholders_read_as_missing: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Number columns written with units ("$98.06", "54 km"): {column: {"kept", "counts", "factors", "meanings", "by"}}
    units_read: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Columns with several units nobody could safely read yet (still text): {column: {unit as written: count}}
    mixed_units: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # The user's choice of unit to keep per column (upload config `units`), and choices that couldn't be used
    unit_choices: Dict[str, str] = field(default_factory=dict)
    ignored_unit_choices: Dict[str, str] = field(default_factory=dict)

    def notes(self) -> List[str]:
        """Plain-English notes on changes made while loading, for the report and the LLM context."""
        notes = []
        for column, found in self.placeholders_read_as_missing.items():
            values = ", ".join(f"{value.strip() or '(blank)'!r} x{count:,}" for value, count in found.items())
            notes.append(f"{column}: {sum(found.values()):,} placeholder value(s) read as missing ({values}); every other value is a number")
        for column, read in self.units_read.items():
            notes.append(f"{column}: {_describe_units(read)}")
        for column, counts in self.mixed_units.items():
            notes.append(f"{column}: mixed units ({_unit_list(counts)}) not converted; kept as text")
        for column, reason in self.ignored_unit_choices.items():
            notes.append(f"{column}: unit setting ignored ({reason})")
        return notes

    def resolve_units(self, column: str, meanings: Mapping[str, str]) -> Optional[str]:
        """Convert a mixed-units column using unit names from the planner ({"C": "degC", "F": "degF"}).

        Python checks the names with pint and does the arithmetic. Returns None when converted, else why not.
        """
        counts = self.mixed_units.get(column)
        if counts is None:
            return f"{column!r} has no unresolved units"
        if set(meanings) != set(counts):
            return f"the unit names must cover exactly {sorted(counts)}"
        target = self.unit_choices.get(column) if self.unit_choices.get(column) in counts else most_common_unit(counts)
        try:
            factors = conversion_factors(meanings, target)
        except UnitError as exc:
            return str(exc)
        position = list(self.df.columns).index(column)
        parsed = read_text_numbers(self.df.iloc[:, position])
        if parsed is None:  # cannot happen for a column that was recorded as mixed units
            return "the column no longer reads as numbers"
        _apply_units(self.df, position, parsed, factors)
        self.units_read[column] = _units_record(parsed, target, factors, dict(meanings), "planner")
        if parsed.placeholders:
            self.placeholders_read_as_missing[column] = parsed.placeholders
        del self.mixed_units[column]
        return None


def _unit_list(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{label or '(no unit)'} x{count:,}" for label, count in counts.items())


def _describe_units(read: Mapping[str, Any]) -> str:
    counts, kept = read["counts"], read["kept"]
    if len(counts) == 1:
        if not kept:
            return "read as numbers (commas between thousands removed)"
        return f"{kept!r} removed from {counts[kept]:,} values (one unit in the column)"
    converted = ", ".join(f"{counts[label]:,} from {label}" for label in counts if label != kept)
    meanings = ", ".join(f"{label} = {name}" for label, name in read["meanings"].items())
    return f"converted to {kept} ({converted}); units read by {read['by']} as {meanings}"


def _apply_units(df: pd.DataFrame, position: int, parsed: TextNumbers, factors: Mapping[str, Tuple[float, float]]) -> None:
    """Replace the text column with value * scale + offset per row (the same arithmetic the notebook repeats).

    Converted values are rounded to UNIT_DECIMALS places, dropping float noise (32F -> -3.7e-13 C becomes 0.0).
    """
    scale = parsed.units.map({label: f[0] for label, f in factors.items()})
    offset = parsed.units.map({label: f[1] for label, f in factors.items()})
    values = (parsed.numbers * scale + offset).astype("float64")
    if len(factors) > 1:
        values = values.round(UNIT_DECIMALS) + 0.0  # + 0.0 turns -0.0 into 0.0
    df.isetitem(position, values)


def _units_record(parsed: TextNumbers, kept: str, factors, meanings, by: str) -> Dict[str, Any]:
    return {
        "kept": kept,
        "counts": dict(parsed.unit_counts),
        "factors": {label: [scale, offset] for label, (scale, offset) in factors.items()},
        "meanings": meanings,
        "by": by,
    }


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


def _read_number_columns(loaded: LoadedFile) -> None:
    """Read text columns that hold numbers: placeholders become missing, one unit is removed, mixed units converted.

    pandas reads a whole column as text when a single cell is not a number, so "?" in 15 of 300 rows, or "$"
    on every value, would hide the column from the number checks. Columns with real text are left alone.
    Mixed units pint can't read safely stay text in `mixed_units`, for the planner to name.
    """
    df = loaded.df
    for position in range(df.shape[1]):
        name = str(df.columns[position])
        parsed = read_text_numbers(df.iloc[:, position])
        choice = loaded.unit_choices.get(name)
        if parsed is None:
            if choice is not None:
                loaded.ignored_unit_choices[name] = "the column is not numbers with units"
            continue
        counts = parsed.unit_counts

        if choice is not None and choice not in counts:
            loaded.ignored_unit_choices[name] = f"{choice!r} is not one of its units ({_unit_list(counts)})"
            choice = None
        meanings = pint_meanings(counts) if len(counts) > 1 else None
        if len(counts) > 1 and meanings is None:
            loaded.mixed_units[name] = dict(counts)  # stays text: its placeholders are recorded only once converted
            continue

        if parsed.placeholders:
            loaded.placeholders_read_as_missing[name] = parsed.placeholders
        if parsed.plain:
            df.isetitem(position, parsed.numbers)
        elif len(counts) == 1:
            label = next(iter(counts))
            _apply_units(df, position, parsed, {label: (1.0, 0.0)})
            loaded.units_read[name] = _units_record(parsed, label, {label: (1.0, 0.0)}, {}, "one unit")
        else:
            target = choice or most_common_unit(counts)
            factors = conversion_factors(meanings, target)
            _apply_units(df, position, parsed, factors)
            loaded.units_read[name] = _units_record(parsed, target, factors, meanings, "pint")

    names = {str(c) for c in df.columns}
    for name in loaded.unit_choices:
        if name not in names:
            loaded.ignored_unit_choices[name] = "no column with this name"


def load_file(file_bytes: bytes, filename: str, unit_choices: Optional[Mapping[str, str]] = None) -> LoadedFile:
    """Read an uploaded CSV, XLSX, or XLS file into a DataFrame. Every row is kept.

    Raises:
        FileLoadError: unsupported type, empty, too large, unreadable, or no data rows.

    Text columns holding numbers are read as numbers: placeholders become missing, units are removed or
    converted (see `notes()`). `unit_choices` ({column: unit as written}) picks the unit kept for mixed units.
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

    loaded = LoadedFile(
        df=df,
        file_type=file_type,
        original_columns=original_columns,
        renamed_columns=renamed,
        encoding=encoding,
        delimiter=delimiter,
        sheet_name=sheet_name,
        sheet_count=sheet_count,
        unit_choices={str(k).strip().lower(): str(v) for k, v in (unit_choices or {}).items()},
    )
    _read_number_columns(loaded)
    if loaded.notes():
        logger.info("Loading notes for %s: %s", filename, loaded.notes())
    return loaded
