"""Utility — readable number formatting shared by tool summaries."""
import math
from typing import Any


def format_number(value: Any) -> str:
    """Readable number for a summary: never rounds small values down to zero."""
    if value is None:
        return "n/a"
    # Values that aren't plain numbers (e.g. complex numbers, stored as text) are shown as they are
    if not isinstance(value, (int, float)):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    if abs(value) >= 1e15:
        return f"{value:.4g}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.4g}"  # 4 significant digits: 0.00012, 3.142, 512


def format_pct(pct: float) -> str:
    """'24.1%', but small shares keep a meaningful digit: '0.02%' rather than '0.0%'."""
    return f"{pct:.1f}%" if pct >= 1 or pct == 0 else f"{pct:.2g}%"
