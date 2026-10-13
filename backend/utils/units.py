"""Utility — reads numbers written as text ("$98.06", "1,234", "26C", "54 km") and works out how to convert mixed units.

Units are never listed in advance: the unit is whatever text surrounds the number. A column with one unit
just loses it. A column with several units is converted with pint when pint can read every unit safely;
otherwise the units go to the planner's LLM call, whose answer is checked here before Python converts.
"""
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Mapping, Optional, Tuple

import pandas as pd

from ..config import MISSING_PLACEHOLDERS

logger = logging.getLogger(__name__)

PLACEHOLDER_SET = {value.lower() for value in MISSING_PLACEHOLDERS}
# Text before the number, the number (US thousands commas allowed), text after it. No digits or spaces inside
# the unit, so dates (2024-01-05), ranges (10-20), times (10:30) and European numbers (1.234,5) never match.
UNIT_NUMBER = r"^\s*(?P<prefix>[^\d\s+\-.,]*)(?P<gap>\s*)(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+)\s*(?P<suffix>[^\d\s]*)\s*$"
UNIT_NUMBER_RE = re.compile(UNIT_NUMBER)
UNIT_MAX_LENGTH = 10  # longer "units" are words, not units
PINT_PREFIX_MAX_SYMBOL = 2  # pint may add a metric prefix only to short symbols: km, kg, kWh - not kmph (kilo-mph)


class UnitError(ValueError):
    """A unit reading that can't be used safely. The message says why, for the loading notes."""


@dataclass
class TextNumbers:
    """A text column read as numbers. `units` is per row ("" = no unit, NaN = gap or placeholder)."""

    numbers: pd.Series
    units: pd.Series
    unit_counts: Dict[str, int]       # unit as written -> rows, in order of first appearance
    placeholders: Dict[str, int]      # placeholder as written -> rows
    plain: bool                       # every value was an ordinary number (pd.to_numeric), no unit text at all


def _is_placeholder(text: pd.Series) -> pd.Series:
    stripped = text.str.strip()
    return (stripped == "") | stripped.str.lower().isin(PLACEHOLDER_SET)


def _looks_like_code(parts: pd.DataFrame) -> pd.Series:
    """Values written like codes, not like a number with a unit. Judged by how they are written, not by a list."""
    prefix_has_letter = parts["prefix"].str.contains(r"[^\W\d_]", regex=True)
    letters_attached = prefix_has_letter & (parts["gap"] == "")            # A12, EDM01 (but "USD 50" is a unit)
    sign_after_prefix = (parts["prefix"] != "") & parts["number"].str.match(r"[+-]")  # EDM-01
    leading_zero = parts["number"].str.match(r"[+-]?0\d")                 # 01, 007 (but 0.5 is a number)
    return letters_attached | sign_after_prefix | leading_zero


def read_text_numbers(series: pd.Series) -> Optional[TextNumbers]:
    """Read a text column as numbers, or return None when it holds real text ("north", "12-20", "A12 and B4").

    Decided on distinct values only, so a 75,000-row column costs as much as its few distinct values.
    """
    if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
        return None
    # str() also covers Excel columns that mix real numbers with text cells
    text = series.dropna().astype(str)
    distinct = pd.Series(text.unique())
    is_placeholder = _is_placeholder(distinct)
    values = distinct[~is_placeholder]
    if values.empty:
        return None

    # Ordinary numbers apart from placeholders ("?" in a number column)
    plain_numbers = pd.to_numeric(values, errors="coerce")
    plain = bool(plain_numbers.notna().all())
    if plain:
        number_of = dict(zip(values, plain_numbers.astype("float64")))
        unit_of = dict.fromkeys(values, "")
    else:
        parts = values.str.extract(UNIT_NUMBER_RE)
        if parts["number"].isna().any():
            return None  # some value isn't "number with a unit": real text
        if _looks_like_code(parts).any():
            return None  # station or product codes (EDM-01, A12, 007), not measurements
        units = parts["prefix"] + parts["suffix"]
        if units.str.len().max() > UNIT_MAX_LENGTH:
            return None
        numbers = pd.to_numeric(parts["number"].str.replace(",", "", regex=False))
        number_of = dict(zip(values, numbers.astype("float64")))
        unit_of = dict(zip(values, units))

    row_units = text.map(unit_of)
    counts = row_units.value_counts()
    found = text[text.isin(distinct[is_placeholder])]
    return TextNumbers(
        numbers=pd.Series(text.map(number_of), dtype="float64").reindex(series.index),
        units=row_units.reindex(series.index),
        unit_counts={str(unit): int(counts[unit]) for unit in row_units.dropna().unique()},
        placeholders={str(v): int(c) for v, c in found.value_counts().items()},
        plain=plain,
    )


# --- converting between units ---

@lru_cache(maxsize=1)
def _registry():
    """pint's unit registry, created once and only when a column has mixed units (about 1 s with its cache)."""
    import pint

    return pint.UnitRegistry(cache_folder=":auto:")


def _read_unit(label: str):
    """pint's reading of a unit as written in the data, only when it is safe to trust without context."""
    registry = _registry()
    try:
        readings = registry.parse_unit_name(label)
    except Exception:
        return None
    if not readings:
        return None
    # "min" can be minute or milli-inch: trust the reading with no prefix; a prefix only on a short symbol (km, not kmph)
    unprefixed = [r for r in readings if not r[0]]
    if unprefixed:
        prefix, base, _ = unprefixed[0]
    elif len(readings) == 1 and len(registry.get_symbol(readings[0][1])) <= PINT_PREFIX_MAX_SYMBOL:
        prefix, base, _ = readings[0]
    else:
        return None
    unit = registry.parse_units(prefix + base)
    # Without a physical quantity (%, bytes) "same kind of quantity" proves nothing
    return unit if dict(unit.dimensionality) else None


def pint_meanings(unit_counts: Mapping[str, int]) -> Optional[Dict[str, str]]:
    """{unit as written: pint unit} when pint reads every unit safely and they measure the same quantity, else None."""
    if "" in unit_counts or len(unit_counts) < 2:
        return None  # a bare number next to "kg" could be anything
    meanings = {}
    for label in unit_counts:
        unit = _read_unit(label)
        if unit is None:
            return None
        meanings[label] = str(unit)
    try:
        conversion_factors(meanings, next(iter(unit_counts)))
    except UnitError:
        return None  # e.g. C (coulomb) and F (farad): pint read them, as the wrong thing
    return meanings


def conversion_factors(meanings: Mapping[str, str], target: str) -> Dict[str, Tuple[float, float]]:
    """{unit as written: (scale, offset)} so that value_in_target = value * scale + offset.

    `meanings` maps each unit as written to a pint unit name ("F" -> "degF"); every unit must measure the same
    quantity. Raises UnitError otherwise. Scale and offset come from converting 0 and 1, so the notebook can
    repeat the conversion with plain arithmetic.
    """
    registry = _registry()
    if target not in meanings:
        raise UnitError(f"the unit to keep ({target!r}) is not one of the column's units")
    try:
        units = {label: registry.parse_units(name) for label, name in meanings.items()}
    except Exception as exc:
        raise UnitError(f"unknown unit name: {exc}") from exc
    goal = units[target]
    if not dict(goal.dimensionality):
        raise UnitError("units without a physical quantity (%, counts) can't be checked")
    factors = {}
    for label, unit in units.items():
        if unit.dimensionality != goal.dimensionality:
            raise UnitError(f"{label!r} ({unit}) and {target!r} ({goal}) measure different things")
        zero = registry.Quantity(0.0, unit).to(goal).magnitude
        one = registry.Quantity(1.0, unit).to(goal).magnitude
        factors[label] = (float(one - zero), float(zero))
    return factors


def most_common_unit(unit_counts: Mapping[str, int]) -> str:
    """The unit kept by default: the most common one; a tie goes to the one that appears first in the column."""
    return max(unit_counts, key=lambda label: unit_counts[label])  # max keeps the first of equal counts
