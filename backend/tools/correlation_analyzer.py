"""Tool — computes the correlation matrix and returns the top-10 pairs by absolute value."""
import pandas as pd
from ..models import ToolResult

def analyze_correlation(df: pd.DataFrame) -> ToolResult:
    try:
        corr_matrix = df.select_dtypes(include='number').corr()

        corr_unstacked = corr_matrix.unstack()
        corr_unstacked = corr_unstacked[
            corr_unstacked.index.get_level_values(0)
                                        < corr_unstacked.index.get_level_values(1)]
        
        corr_unstacked = corr_unstacked.abs().sort_values(ascending = False).head(10)

        payload = {
            "corr_matrix": corr_matrix.to_dict(),
            "corr_unstacked": corr_unstacked.to_dict(),
        }

        lines = [f"Top correlated column pairs (by absolute value):"]

        for (col1, col2), val in corr_unstacked.items():
            lines.append(f"{col1} vs {col2}: {val:.2f}")

        summary = "\n".join(lines)


        return ToolResult(
            tool_name = "correlation_analyzer",
            success = True,
            payload = payload,
            summary = summary,
            error = None,
        )
    except Exception as e:
        return ToolResult(
            tool_name = "correlation_analyzer",
            success = False,
            payload = {},
            summary = "",
            error = str(e),
        )