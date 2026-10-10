"""Agent — uses Claude to build an ordered AnalysisPlan of tool calls, with autonomous column detection."""
import logging
import re
import warnings
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import pandas as pd
from pandas.tseries.api import guess_datetime_format

from ..config import (
    DATE_PARSE_MIN_RATIO,
    DATE_PARSE_SAMPLE_SIZE,
    IMBALANCE_THRESHOLD,
    SKEW_MIN_VALUES,
    SKEW_THRESHOLD,
)
from ..models import ColumnFlag, FlagKind

logger = logging.getLogger(__name__)

# A standalone 4-digit number, like the 2024 in "05/01/2024"
FOUR_DIGIT_YEAR = re.compile(r"(?<!\d)\d{4}(?!\d)")

# pandas can't guess formats with 2-digit years ("05/01/24"), so these are tried explicitly.
# Each pair is (month-first, day-first) for the same layout.
TWO_DIGIT_YEAR_FORMATS = [
    ("%m/%d/%y", "%d/%m/%y"),
    ("%m-%d-%y", "%d-%m-%y"),
    ("%m.%d.%y", "%d.%m.%y"),
]


# ---------------------------------------------------------------------------
# Autonomous column detection — pure pandas, no LLM
# ---------------------------------------------------------------------------

def _is_text_column(series: pd.Series) -> bool:
    # pandas 3 stores text as a "string" dtype; older files may still produce "object"
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def _guess_format(example: str, dayfirst: bool) -> Optional[str]:
    """pandas' best guess at a date format for one value, e.g. "%d/%m/%Y". None if it isn't a date."""
    # pandas warns when its guess disagrees with dayfirst; that's expected here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return guess_datetime_format(example, dayfirst=dayfirst)


def _two_digit_year_formats(example: str) -> Tuple[Optional[str], Optional[str]]:
    """The (month-first, day-first) pair whose layout fits the example, or (None, None)."""
    for month_first, day_first in TWO_DIGIT_YEAR_FORMATS:
        for fmt in (month_first, day_first):
            try:
                datetime.strptime(example, fmt)
                return month_first, day_first
            except ValueError:
                continue
    return None, None


def _candidate_formats(example: str, dayfirst: Optional[bool]) -> List[str]:
    """Date formats worth trying for a column, based on one example value."""
    month_first = _guess_format(example, dayfirst=False)
    day_first = _guess_format(example, dayfirst=True)
    if month_first is None and day_first is None:
        month_first, day_first = _two_digit_year_formats(example)

    # Year-first dates (2024-01-05) are never written year-day-month, so day order doesn't apply.
    # Forcing day-first on them would silently turn 2024-01-05 into 1 May.
    if month_first and month_first.startswith("%Y"):
        return [month_first]

    if dayfirst is True:
        candidates = [day_first]
    elif dayfirst is False:
        candidates = [month_first]
    else:
        candidates = [month_first, day_first]
    return list(dict.fromkeys(fmt for fmt in candidates if fmt))


def _parse_dates(values: pd.Series, dayfirst: Optional[bool]) -> Tuple[pd.Series, str]:
    """Parse text into dates. Returns the parsed values and how it went: "ok", "ambiguous", or "mixed".

    With dayfirst=None, both orders are tried with one consistent format each, and the one
    that parses more rows wins. The wrong order leaves rows unparsed (13/11 is not a valid
    month-first date), so the right one shows up. If both parse equally well but give
    different dates, every day was 12 or less and the order can't be known: month-first
    is used and the result is "ambiguous". "mixed" means no single format fit.
    """
    attempts = [
        pd.to_datetime(values, errors="coerce", format=fmt)
        for fmt in _candidate_formats(values.iloc[0], dayfirst)
    ]
    good = [parsed for parsed in attempts if parsed.notna().mean() >= DATE_PARSE_MIN_RATIO]

    # No single format fits, e.g. a column mixing "2024-01-05" and "Jan 6 2024": parse value by value
    if not good:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=bool(dayfirst)), "mixed"

    # max() keeps the first on a tie, so month-first wins when both orders fit equally
    best = max(good, key=lambda parsed: parsed.notna().mean())
    ambiguous = (
        len(good) == 2
        and good[0].notna().mean() == good[1].notna().mean()
        and not good[0].equals(good[1])
    )
    return best, "ambiguous" if ambiguous else "ok"


def _date_detail(series: pd.Series, dayfirst: Optional[bool]) -> Optional[str]:
    """Return a description if the column holds dates, otherwise None."""
    # Columns pandas already recognised as dates
    if pd.api.types.is_datetime64_any_dtype(series):
        values = series.dropna()
        if values.empty:
            return None
        return f"date column ranging from {values.min().date()} to {values.max().date()}"

    if not _is_text_column(series):
        return None

    values = series.dropna().astype(str)
    if values.empty:
        return None
    sample = values.sample(n=min(len(values), DATE_PARSE_SAMPLE_SIZE), random_state=42)

    # Plain numbers like "2024" or "10001" parse as dates but are usually years, IDs, or codes
    if pd.to_numeric(sample, errors="coerce").notna().mean() >= DATE_PARSE_MIN_RATIO:
        return None

    # A date needs a year. Times ("10:30") and "Jan 5" parse too, but pandas invents the missing
    # parts from today's date, which would put a false range in the report.
    has_four_digit_year = sample.str.contains(FOUR_DIGIT_YEAR).mean() >= DATE_PARSE_MIN_RATIO
    if not has_four_digit_year and _two_digit_year_formats(sample.iloc[0]) == (None, None):
        return None

    setting_ignored = False
    try:
        # The sample only decides IF it's a date column
        sample_parsed, _ = _parse_dates(sample, dayfirst)
        if sample_parsed.notna().mean() < DATE_PARSE_MIN_RATIO:
            return None

        # Facts in the report come from every row (about 1 second per million rows)
        full, status = _parse_dates(values, dayfirst)

        # The user's day/month setting didn't fit these dates, but the data itself might settle it
        if dayfirst is not None and status == "mixed":
            auto, auto_status = _parse_dates(values, None)
            if auto_status != "mixed":
                full, status, setting_ignored = auto, auto_status, True
    except (ValueError, TypeError):
        return None

    detail = (
        f"text column where {full.notna().mean():.0%} of values parse as dates, "
        f"ranging from {full.min().date()} to {full.max().date()}"
    )
    if setting_ignored:
        detail += "; the day/month order in the upload settings did not fit these dates, so it was detected from the data"
    if status == "ambiguous":
        detail += "; day/month order is ambiguous (every day is 12 or less), assumed month-first"
    return detail


def _skew_detail(series: pd.Series) -> Optional[str]:
    """Return a description if a numeric column is long-tailed (highly skewed), otherwise None.

    Decided only by the values, never the column name, so it works for revenue, rainfall,
    web traffic, or hospital stays alike.
    """
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return None

    values = series.dropna()
    # Too few values give a meaningless skew. Two-value columns (0/1 flags) belong to the imbalance check.
    if len(values) < SKEW_MIN_VALUES or values.nunique() <= 2:
        return None

    skew = values.skew()
    if pd.isna(skew) or abs(skew) <= SKEW_THRESHOLD:
        return None

    tail = "right" if skew > 0 else "left"
    return (
        f"long {tail} tail (skew {skew:.2f}): mean {values.mean():,.2f} vs median {values.median():,.2f}, "
        f"range {values.min():,.2f} to {values.max():,.2f}"
    )


def _imbalance_detail(series: pd.Series, is_target: bool) -> Optional[str]:
    """Return a description if a two-class column has a small minority class, otherwise None."""
    counts = series.value_counts(dropna=True)
    if len(counts) != 2:
        return None

    minority_count = int(counts.min())
    total = int(counts.sum())
    share = minority_count / total
    if share >= IMBALANCE_THRESHOLD:
        return None

    role = "target column" if is_target else "possible target"
    return (
        f"{role}: minority class '{counts.idxmin()}' is {share:.1%} of rows "
        f"({minority_count:,} of {total:,})"
    )


def _run_check(kind: FlagKind, column: str, check: Callable[..., Optional[str]], *args) -> Optional[ColumnFlag]:
    """Run one check on one column. Returns a flag if it found something, otherwise None.

    Safety net: detection only adds hints, so one odd column must never stop the whole analysis.
    Expected problems (e.g. text that isn't a date) are handled quietly inside each check.
    Anything unexpected is logged as a warning with the full error, then that check is skipped
    for that column only.
    """
    try:
        detail = check(*args)
    except Exception:
        logger.warning("Column detection: %s check failed on column '%s', skipping it", kind, column, exc_info=True)
        return None
    return ColumnFlag(kind=kind, column=column, detail=detail) if detail else None


def detect_interesting_columns(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    date_dayfirst: Optional[bool] = None,
) -> List[ColumnFlag]:
    """Find columns worth the planner's attention: dates, long-tailed numbers, and class imbalance.

    Every check looks at the values only, never at column names, so it works on any dataset.

    If `target_column` is given, only that column is checked for imbalance. Otherwise every
    column with exactly two values is checked, since any of them could be the target.

    `date_dayfirst` comes from the user's upload config: True for dd/mm/yyyy files, False for
    mm/dd/yyyy, None (default) to work it out from the data. Year-first dates are unaffected.
    """
    if target_column is not None and target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' is not a column in the dataset")

    flags: List[ColumnFlag] = []
    for position in range(df.shape[1]):
        # Read by position, not name: two columns can end up with the same name
        # once names are cleaned up (e.g. "Amount" and "amount ")
        series = df.iloc[:, position]
        name = str(df.columns[position])
        is_target = target_column is not None and df.columns[position] == target_column

        flags.extend(filter(None, [
            _run_check("date_column", name, _date_detail, series, date_dayfirst),
            _run_check("skewed_distribution", name, _skew_detail, series),
            # With a target, only the target is checked; without one, any two-class column could be it
            _run_check("class_imbalance", name, _imbalance_detail, series, is_target)
            if target_column is None or is_target else None,
        ]))

    logger.info("Column detection found %d flag(s) across %d columns", len(flags), df.shape[1])
    return flags
