"""Tool — computes shape, dtypes, basic descriptive stats, and sample rows from a DataFrame."""
import pandas as pd
from ..models import ToolResult

def inspect_dataset(df: pd.DataFrame) -> ToolResult:
    try:
        payload = {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "dtypes": df.dtypes.to_dict(),
            "describe": df.describe(include = "all").to_dict(),
            "sample": df.sample(3, random_state = 42).to_dict(),
        }

        lines = [f"Dataset: {payload['shape']['rows']} rows x {payload['shape']['columns']} columns"]

        for col, dtype in payload['dtypes'].items():
            desc = payload['describe'].get(col, {})
            if 'mean' in desc:
                lines.append(f"{col} ({dtype}): mean={desc['mean']:.2f}, std = {desc['std']:.2f}")
            else:
                lines.append(f"{col} ({dtype}): {df[col].nunique()} unique values")

        summary = "\n".join(lines)
        
        return ToolResult(
            tool_name = "dataset_inspector",
            success = True,
            payload = payload,
            summary = summary,
            error = None,
        )
    except Exception as e:
        return ToolResult(
            success = False,
            tool_name = "dataset_inspector",
            payload = {},
            summary = "",
            error = str(e),
        )