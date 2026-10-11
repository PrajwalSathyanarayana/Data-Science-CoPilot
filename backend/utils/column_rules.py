"""Utility — value-based rules about columns shared by several tools (never based on column names)."""
import numpy as np


def looks_like_id(distinct: np.ndarray, count: int) -> bool:
    """Every value different, whole numbers, evenly spaced (1, 2, 3... or 10001, 10002...): a row number or ID.

    `distinct` is the sorted unique values (np.unique), `count` the number of non-missing values.
    """
    if distinct.size != count or distinct.size < 3 or not np.all(distinct == np.round(distinct)):
        return False
    steps = np.diff(distinct)
    return bool(np.all(steps == steps[0]))
