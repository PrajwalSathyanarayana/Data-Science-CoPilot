"""Tool — detects outliers via IQR/Z-score and returns per-column counts and percentages."""
import pandas as pd
from ..models import ToolResult

def analyze_outlier(df: pd.DataFrame, method: str = "iqr") -> ToolResult:
    try:
        outliers = df.select_dtypes(include = 'number')

        if method == 'iqr':
            q1 = outliers.quantile(0.25)
            q3 = outliers.quantile(0.75)
            iqr = q3 - q1
            mask = ((outliers < (q1 - 1.5 * iqr)) | (outliers > (q3 + 1.5* iqr)))

        elif method == 'zscore':
            z_score = (outliers - outliers.mean()) / (outliers.std())
            mask = z_score.abs() > 3

        outlier_count = mask.sum()
        outlier_pct = outlier_count / len(df) * 100

        payload = {
            'outlier_count': outlier_count.to_dict(),
            'outlier_pct': outlier_pct.to_dict(),
            'method': method,
            'total_rows': len(df),
        }

        lines = [f"Outlier Report: ({method}): {(outlier_count > 0).sum()} columns have outliers"]

        for col, count in outlier_count.items():
            if count > 0:
                lines.append(f"{col}: {count} outliers {outlier_pct[col]:.2f}%")
            
        summary = "\n".join(lines)

        return ToolResult(
            tool_name = "analyze_outlier",
            success = True,
            payload = payload,
            summary = summary,
            error = None,
        )
    
    except Exception as e:
        return ToolResult(
            tool_name = "analyze_outlier",
            success = False,
            payload = {},
            summary = "",
            error = str(e),
        )