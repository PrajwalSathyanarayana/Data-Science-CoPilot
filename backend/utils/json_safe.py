"""Utility — converts pandas/numpy results into plain JSON values, so tool payloads can be saved and sent by the API."""
import datetime
import math
from typing import Any

import numpy as np
import pandas as pd


def to_json_safe(value: Any) -> Any:
    """Return a copy of `value` made only of dict, list, str, int, float, bool, and None.

    - numpy numbers become Python numbers; NaN and infinity become None (JSON has no NaN)
    - dates and times become ISO text ("2024-01-05T00:00:00"); missing dates (NaT) become None
    - dtypes become their name ("float64")
    - DataFrames become {column: {row: value}}, Series become {row: value}
    - anything else unknown becomes its text form
    """
    # bool must be checked before int: in Python, True is also an int
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None

    # Missing-value markers: NaT, pd.NA
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta, datetime.timedelta, np.timedelta64, np.datetime64)):
        return None if pd.isna(value) else str(value)

    # Containers, converted recursively
    if isinstance(value, pd.DataFrame):
        return {str(column): to_json_safe(value[column].to_dict()) for column in value.columns}
    if isinstance(value, pd.Series):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [to_json_safe(item) for item in value]

    # dtypes, complex numbers, Periods, and anything else: their readable text form
    return str(value)
