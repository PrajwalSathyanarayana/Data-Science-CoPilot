# DS Copilot — AI-Powered Data Science Assistant

DS Copilot is an AI-powered tool that turns any CSV or Excel file into a full data science report — no coding required.

Upload a dataset and the system automatically runs exploratory data analysis, detects patterns and anomalies, generates interactive visualizations, proposes hypotheses, and recommends machine learning approaches. A final report is produced as a downloadable PDF and Jupyter notebook.

Under the hood, Python and pandas handle all the computation. Claude (Anthropic's LLM) receives only compressed statistical summaries and translates them into plain-language business insights.

## Intended Audience

- **Data analysts and scientists** who want a fast first-pass analysis before diving deeper
- **Business analysts** who work with structured data but don't write code
- **Students and learners** exploring a new dataset and looking for a starting point
- **Anyone** who has a CSV or Excel file and wants to understand what's in it quickly

## Current Status

Under active development. Core backend (FastAPI) and frontend (Streamlit) are being built out.