"""Tool — generates Plotly figures (distributions, heatmaps, scatter plots) serialized as JSON."""
import pandas as pd
import plotly.express as px
from ..models import ToolResult

def analyze_visualizations(df: pd.DataFrame, tool_results: dict) -> ToolResult:
    try:
        numeric_cols = df.select_dtypes(include = 'number').columns
        figures = {}

        for col in numeric_cols:
            fig = px.histogram(df, x = col, title = f"Distribution of {col}")
            figures[f"histogram_{col}"] = fig.to_json()

        corr_data = tool_results.get("correlation_analyzer", {}).get("corr_matrix")
        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            fig = px.imshow(corr_df, title = "Correlation Heatmap", 
                            color_continuous_scale = "RdBu_r", zmin = -1, zmax = 1)
            figures["heatmap_correlation"] = fig.to_json()
        
        for col in numeric_cols:
            fig = px.box(df, y = col, title = f"Box Plot of {col}")
            figures[f"boxplot_{col}"] = fig.to_json()

        missing_data = tool_results.get("missing_value_analyzer", {}).get("missing_pct")
        if missing_data:
            missing_df = pd.DataFrame(list(missing_data.items()), columns = ['column','missing_pct'])
            fig = px.bar(missing_df, x = 'column',
                         y = 'missing_pct', title = 'Missing Value % per Column')
            figures["bar_missing_values"] = fig.to_json()
        
        payload = {
            'figures': figures
        }

        summary = f"Generated {len(figures)} visualizations: {', '.join(figures.keys())}"

        return ToolResult(
            tool_name = "visualize_data",
            success = True,
            payload = payload,
            summary = summary,
            error = None,
        )


    except Exception as e:
        return ToolResult(
            tool_name = "visualize_data",
            success = False,
            payload = {},
            summary = '',
            error = str(e),
        )