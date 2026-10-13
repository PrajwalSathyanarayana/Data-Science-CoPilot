"""Tests for backend/utils/units.py: numbers written as text, and when mixed units can be converted safely."""
import pandas as pd
import pytest

from backend.utils.units import UnitError, conversion_factors, most_common_unit, pint_meanings, read_text_numbers


def read(values):
    return read_text_numbers(pd.Series(values))


# --- reading numbers out of text ---

@pytest.mark.parametrize(
    "values, numbers, units",
    [
        (["$98.06", "$1,204.50"], [98.06, 1204.5], {"$": 2}),
        (["54km", "76mi", "12 km"], [54.0, 76.0, 12.0], {"km": 2, "mi": 1}),
        (["26C", "32F", "-5C"], [26.0, 32.0, -5.0], {"C": 2, "F": 1}),
        (["USD 50", "USD 1,200"], [50.0, 1200.0], {"USD": 2}),
        (["45%", "0.5%"], [45.0, 0.5], {"%": 2}),
        (["1,234", "5"], [1234.0, 5.0], {"": 2}),
        (["9 days", "10 days"], [9.0, 10.0], {"days": 2}),
    ],
    ids=["currency", "distance", "temperature", "currency_code", "percent", "thousands", "word_unit"],
)
def test_any_unit_is_read_from_the_data(values, numbers, units):
    parsed = read(values)
    assert parsed.numbers.tolist() == numbers
    assert parsed.unit_counts == units


def test_placeholders_become_gaps_and_are_counted():
    parsed = read(["$5", "?", "$7", "unknown", "?"])
    assert parsed.numbers.isna().sum() == 3
    assert parsed.placeholders == {"?": 2, "unknown": 1}


@pytest.mark.parametrize(
    "values",
    [
        ["north", "south"],
        ["10-20", "5"],             # a range
        ["2024-01-05", "2024-02-01"],  # dates
        ["10:30", "11:45"],         # times
        ["1.234,5", "2,5"],         # European numbers: not guessed
        ["12 kg net", "3 kg"],      # a unit plus words
        ["EDM-01", "EDM-02"],       # station codes
        ["A12", "B7"],              # codes with letters attached
        ["?", "unknown"],           # only placeholders
    ],
    ids=["words", "range", "dates", "times", "european", "unit_and_words", "codes_with_dash", "codes", "only_placeholders"],
)
def test_text_that_is_not_a_number_with_a_unit_is_left_alone(values):
    assert read(values) is None


# --- converting mixed units ---

def test_pint_converts_units_it_reads_safely():
    assert pint_meanings({"km": 2, "mi": 1}) == {"km": "kilometer", "mi": "mile"}


@pytest.mark.parametrize(
    "units",
    [
        {"C": 3, "F": 3},           # pint reads coulomb and farad
        {"kmph": 1, "mph": 1},      # pint reads kmph as kilo-mph
        {"%": 1, "B": 1},           # no physical quantity behind either
        {"": 2, "kg": 1},           # a bare number next to kg could be anything
        {"$": 1, "€": 1},           # pint has no currencies
    ],
    ids=["temperature_letters", "kilo_mph", "no_quantity", "bare_number", "currencies"],
)
def test_units_pint_cannot_read_safely_are_left_for_the_planner(units):
    assert pint_meanings(units) is None


def test_conversion_factors_turn_fahrenheit_into_celsius():
    factors = conversion_factors({"C": "degC", "F": "degF"}, "C")
    scale, offset = factors["F"]
    assert factors["C"] == (1.0, 0.0)
    assert round(105 * scale + offset, 2) == 40.56 and round(32 * scale + offset, 9) == 0


@pytest.mark.parametrize(
    "meanings, target, message",
    [
        ({"C": "degC", "F": "farad"}, "C", "measure different things"),
        ({"C": "degC", "F": "not_a_unit"}, "C", "unknown unit name"),
        ({"C": "degC", "F": "degF"}, "K", "not one of the column's units"),
    ],
    ids=["wrong_quantity", "unknown_name", "target_not_in_column"],
)
def test_bad_unit_names_are_rejected(meanings, target, message):
    with pytest.raises(UnitError, match=message):
        conversion_factors(meanings, target)


def test_most_common_unit_wins_and_ties_go_to_the_first():
    assert most_common_unit({"F": 114, "C": 226}) == "C"
    assert most_common_unit({"C": 3, "F": 3}) == "C"
