"""Tests for backend/utils/json_safe.py — pandas/numpy values become strict JSON."""
import datetime
import json

import numpy as np
import pandas as pd
import pytest

from backend.utils.json_safe import to_json_safe


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.int64(7), 7),
        (np.float32(1.5), 1.5),
        (np.bool_(True), True),
        (float("nan"), None),
        (np.inf, None),
        (-np.inf, None),
        (pd.NaT, None),
        (pd.NA, None),
        (pd.Timestamp("2024-01-05"), "2024-01-05T00:00:00"),
        (datetime.date(2024, 1, 5), "2024-01-05"),
        (pd.Timedelta(days=2), "2 days 00:00:00"),
        (np.dtype("float64"), "float64"),
        (1 + 2j, "(1+2j)"),
        ("text", "text"),
        (None, None),
    ],
    ids=[
        "np_int", "np_float", "np_bool", "nan", "inf", "neg_inf", "nat", "pd_na", "timestamp",
        "date", "timedelta", "dtype", "complex", "str", "none",
    ],
)
def test_single_values(value, expected):
    assert to_json_safe(value) == expected


def test_bool_stays_bool_not_int():
    # True is also an int in Python; it must not turn into 1
    assert to_json_safe(True) is True


def test_nested_containers_are_converted_recursively():
    value = {"a": [np.int64(1), {"b": np.nan}], 3: (pd.Timestamp("2024-01-01"),), "s": {np.float64(2.5)}}
    assert to_json_safe(value) == {"a": [1, {"b": None}], "3": ["2024-01-01T00:00:00"], "s": [2.5]}


def test_dataframe_and_series():
    df = pd.DataFrame({"x": [1.0, np.nan], "d": pd.to_datetime(["2024-01-01", None])})
    assert to_json_safe(df) == {"x": {"0": 1.0, "1": None}, "d": {"0": "2024-01-01T00:00:00", "1": None}}
    assert to_json_safe(pd.Series([np.int64(3)], index=["k"])) == {"k": 3}


# pandas itself warns when describing a column that contains infinity; that input is the point of this test
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_result_is_strict_json():
    value = {"stats": pd.Series([1.0, np.nan, np.inf]).describe(), "dtypes": pd.DataFrame({"a": [1]}).dtypes}
    json.dumps(to_json_safe(value), allow_nan=False)  # raises if anything unsafe is left
