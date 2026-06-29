# DS Copilot: AI Powered Data Science Assistant

DS Copilot is an AI-powered tool that turns any CSV or Excel file into a full data science report, no coding required.

Upload a dataset and the system automatically runs exploratory data analysis, detects patterns and anomalies, generates interactive visualizations, proposes hypotheses, and recommends machine learning approaches. A final report is produced as a downloadable PDF and Jupyter notebook.

Under the hood, Python and pandas handle all the computation. Claude (Anthropic's LLM) receives only compressed statistical summaries and translates them into plain-language business insights.

## Intended Audience

- **Data analysts and scientists** who want a fast first-pass analysis before diving deeper
- **Business analysts** who work with structured data but don't write code
- **Students and learners** exploring a new dataset and looking for a starting point
- **Anyone** who has a CSV or Excel file and wants to understand what's in it quickly

## Stack

- **Backend:** FastAPI (Python)
- **Frontend:** React/Node.js (in progress)
- **LLM:** Claude API (`claude-sonnet-4-20250514` for analysis, `claude-haiku-4-5-20251001` for critic)
- **Data:** pandas, numpy, scipy, scikit-learn
- **Visualization:** Plotly (interactive), Matplotlib/Seaborn (export)
- **Notebook export:** nbformat
- **Validation:** Pydantic v2

## Current Status

Active development:

**Completed:**
- FastAPI backend with `/api/analyze`, `/api/health`, `/api/download` endpoints
- File handler — CSV/XLSX validation, encoding detection, normalisation
- All 5 EDA tools: dataset inspector, missing value analyzer, correlation analyzer, outlier analyzer, visualization analyzer
- Notebook generator — produces a downloadable `.ipynb` from analysis results
- Pydantic models: `ToolResult`

**In progress:**
- LLM pipeline — `llm_client`, `token_manager`, `planner`, `insight_generator`
- Pydantic models: `AnalysisPlan`, `FinalReport`
- React/Node.js frontend

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd Data-Science-CoPilot

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Running the Backend

```bash
python -m uvicorn backend.app:app --reload --port 5000
```

API docs (Swagger UI: use this to test endpoints without a frontend) available at `http://127.0.0.1:5000/docs`