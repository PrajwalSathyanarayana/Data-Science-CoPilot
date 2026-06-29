"""Tool — assembles a reproducible Jupyter notebook (.ipynb) from the analysis plan and results."""
import nbformat, os, pandas as pd
from ..models import ToolResult
from ..config import REPORTS_DIR

def generate_notebook(df:pd.DataFrame, tool_results: dict, insights: str, report_id: str, dataset_name: str) -> ToolResult:
    try:
        nb = nbformat.v4.new_notebook() # nb is notebook object
        cells = [] # cells - list wherein all the cells are appended, and then assigned to the notebook at the end.

        cells.append(nbformat.v4.new_markdown_cell(f"# DS Copilot Analysis\n**Dataset:** {dataset_name}"))

        cells.append(nbformat.v4.new_markdown_cell("**Instructions:** Place your dataset file "
        "in the same directory as this notebook before running.\n"f"Expected file: `{dataset_name}`"))

        cells.append(nbformat.v4.new_code_cell("import pandas as pd\n" \
                                               "import numpy as np\n" \
                                               "import plotly.express as px\n" \
                                               "import warnings\n" \
                                               "warnings.filterwarnings('ignore')"))

        cells.append(nbformat.v4.new_code_cell(f"df = pd.read_csv('{dataset_name}')\n" \
                                               "df.columns = [col.strip().lower() for col in df.columns]\n" \
                                               "print(df.shape)\ndf.head()"))

        cells.append(nbformat.v4.new_markdown_cell("## Exploratory Data Analysis"))
        cells.append(nbformat.v4.new_code_cell("print('Shape:', df.shape)\n" \
                                               "print('\\nDtypes:')\n" \
                                               "print(df.dtypes)\n" \
                                               "print('\\nDescribe:')\n" \
                                               "df.describe(include='all')"))

        cells.append(nbformat.v4.new_markdown_cell("### Missing Values"))
        cells.append(nbformat.v4.new_code_cell("missing = df.isnull().sum()\n" \
                                               "missing_pct = (missing / len(df)) * 100\n" \
                                                "pd.DataFrame({'count': missing, 'percentage': missing_pct})[missing > 0].sort_values('percentage', ascending=False)"))

        cells.append(nbformat.v4.new_markdown_cell("### Correlation Analysis"))
        cells.append(nbformat.v4.new_code_cell("corr = df.select_dtypes(include='number').corr()\n" \
                                                "px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title='Correlation Heatmap')"))

        cells.append(nbformat.v4.new_markdown_cell("### Outlier Analysis"))
        cells.append(nbformat.v4.new_code_cell("for col in df.select_dtypes(include='number').columns:\n    " \
                                                "px.box(df, y=col, title=f'Box Plot of {col}').show()"))  

        cells.append(nbformat.v4.new_markdown_cell(f"## Insights\n{insights}"))

        nb.cells = cells
        output_dir = os.path.join(REPORTS_DIR, report_id)
        os.makedirs(output_dir, exist_ok = True)
        output_path = os.path.join(output_dir, "notebook.ipynb")  

        with open(output_path, "w") as f:
            nbformat.write(nb, f)
        
        return ToolResult(
            tool_name = "generate_notebook",
            success = True,
            payload = {"notebook_path": output_path},
            summary = f"Notebook saved to {output_path}",
            error = None,
        )
    
    except Exception as e:
        return ToolResult(
            tool_name = "generate_notebook",
            success = False,
            payload = {},
            summary = '',
            error = str(e),
        )