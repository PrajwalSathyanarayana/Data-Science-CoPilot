"""Tool — counts and ranks missing values per column, filters to columns with >1% missing."""
import pandas as pd
from ..models import ToolResult

def analyze_missing_values(df: pd.DataFrame) -> ToolResult:
    try:
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_pct = missing_pct[missing_pct>1].sort_values(ascending = False)
        missing_count = missing_count[missing_pct.index].sort_values(ascending = False)

        payload = {
            "missing_count" : missing_count.to_dict(),
            "missing_pct": missing_pct.to_dict(),
            "total_rows": len(df),
        }

        lines = [f"Missing value report: {len(missing_pct)} of {len(df.columns)} columns have >1% missing"]

        for col,pct in missing_pct.items():
            lines.append(f"{col}: {payload['missing_count'][col]} missing {pct:.2f}%")

        summary = "\n".join(lines)

        return ToolResult(
            tool_name = "missing_value_analyzer",
            success = True,
            payload = payload,
            summary = summary,
            error = None
        )

    except Exception as e:
        return ToolResult(
            success = False,
            tool_name = "missing_value_analyzer",
            payload = {},
            summary = "",
            error = str(e),
        )