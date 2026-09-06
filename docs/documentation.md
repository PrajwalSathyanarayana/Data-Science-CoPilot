# DS Copilot — File Documentation

A file-by-file reference for what each module does, why key decisions were made, and what to watch out for.

---

## `backend/config.py`

**Purpose:** Single source of truth for all runtime constants and environment variables. Every other module imports from here instead of reading `.env` directly.

**What it does:**
- Loads `.env` using `python-dotenv`
- Validates that `ANTHROPIC_API_KEY` is set at startup — fails fast with a clear error rather than crashing mid-request
- Exposes typed constants (int, str, list) so the rest of the app never deals with raw string env vars

**Key constants and why they exist:**

| Constant | Value | Why |
|---|---|---|
| `ANALYSIS_MODEL` | `claude-sonnet-4-20250514` | More capable model for analysis, insights, hypothesis |
| `CRITIC_MODEL` | `claude-haiku-4-5-20251001` | Cheaper/faster model sufficient for review-only tasks |
| `LLM_MAX_TOKENS` | 1000 | Hard cap on all LLM responses — prevents runaway costs |
| `PLANNER_MAX_TOKENS` | 800 | Token budget for planner input compression |
| `INSIGHT_GENERATOR_MAX_TOKENS` | 3000 | Insight generator needs more context than other agents |
| `CRITIC_AGENT_MAX_TOKENS` | 2000 | Critic sees insights + stats — needs moderate context |
| `HYPOTHESIS_MAX_TOKENS` | 1000 | Hypothesis agent gets a tight summary only |
| `MAX_AGENT_STEPS` | 10 | Prevents infinite agent loops |
| `MAX_AGENT_MEMORY_TOKENS` | 6000 | Forces agent to finish if accumulated memory grows too large |
| `LARGE_FILE_THRESHOLD` | 100,000 rows | Above this, stratified sample to 50k for LLM context |
| `SAMPLE_SIZE` | 50,000 rows | Target sample size for large file LLM context |
| `REPORTS_DIR` | `../reports/` | Built dynamically using `__file__` — works regardless of launch directory |
| `ALLOWED_ORIGINS` | from `.env` | CORS allowed origins — defaults to `http://localhost:8501` (Streamlit) |

**Patterns used:**
- `int(os.getenv("VAR", "default"))` — always cast env vars to the right type; defaults are strings because `os.getenv` returns strings
- `os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")` — go up one level from `backend/` to project root, then join `reports/`

---

## `backend/app.py`

**Purpose:** FastAPI app factory — creates the app instance, configures middleware, and registers all routers. Contains no business logic.

**What it does:**
- Creates the `FastAPI()` app instance
- Adds CORS middleware using `ALLOWED_ORIGINS` from config
- Registers three routers under the `/api` prefix: `analyze`, `health`, `download`

**Why routers and not inline routes:**
Each route file defines its own `APIRouter()`. `app.py` just wires them together with `app.include_router()`. This keeps concerns separated — `app.py` never grows beyond wiring code.

**CORS note:**
`allow_credentials=True` is set but should be revisited before production deployment. `allow_credentials=True` cannot be combined with wildcard `allow_origins=["*"]` — browsers reject it. Since `ALLOWED_ORIGINS` comes from `.env`, set it to the real frontend URL in production.

---

## `backend/routes/health.py`

**Purpose:** Single `GET /api/health` endpoint that confirms the service is running.

**What it does:**
- Returns `{"status": "ok", "version": "0.1.0"}`
- Used by load balancers, monitoring tools, and the frontend to check backend availability

**Pattern:** Defines `router = APIRouter()` — `app.py` imports this and registers it. Every route file follows this same pattern.

---

## `backend/routes/analyze.py`

**Purpose:** `POST /api/analyze` — the main endpoint. Accepts a file upload and config, validates both, and will trigger the full analysis pipeline.

**What it does:**
1. Receives a `multipart/form-data` request with `file` (CSV/XLSX) and `config` (JSON string)
2. Validates file extension against `ALLOWED_FILE_TYPE`
3. Parses `config` string into a dict using `json.loads()` — raises 400 if malformed
4. Reads file bytes to check size against `MAX_FILE_SIZE` — raises 413 if too large
5. Resets file cursor with `await file.seek(0)` after size check so downstream handlers can re-read the file
6. Returns a `FinalReport` (currently placeholder — full pipeline wired in Day 3)

**Key patterns:**
- `file: UploadFile = File(...)` and `config: str = Form(...)` in the function signature — FastAPI's dependency injection handles extraction automatically; no manual `request.form()` needed
- `config.get("target_column")` (not `config["target_column"]`) — `target_column` is optional; `.get()` returns `None` safely instead of raising `KeyError`
- `await file.seek(0)` after `await file.read()` — reading moves the cursor to end; seek resets it so `file_handler.py` can read from the start

**Why `config` comes in as a string:**
FastAPI cannot mix Pydantic body models with file uploads in the same request. Form fields must use `Form()`, which delivers raw strings — so `json.loads()` is needed to parse it manually.

---

## `backend/routes/download.py`

**Purpose:** `GET /api/download/{report_id}/{filename}` — serves saved report files as binary downloads.

**What it does:**
1. Builds the file path using `os.path.join(REPORTS_DIR, report_id, filename)`
2. Checks if the file exists with `os.path.exists()`
3. If it exists — returns it as a `FileResponse` with `application/octet-stream` media type (forces browser download)
4. If it doesn't exist — raises `HTTPException(404)`

**Key patterns:**
- `os.path.join()` instead of f-string path building — handles OS path separators correctly on both Windows (development) and Linux (deployment)
- Explicit `os.path.exists()` check before `FileResponse` — `FileResponse` does not raise an exception for missing files on its own
- Path parameters `{report_id}` and `{filename}` declared directly in the function signature as `str` — FastAPI extracts them from the URL automatically

---

## `backend/utils/file_handler.py`

**Purpose:** Validates, reads, and normalises CSV/XLSX uploads into a pandas DataFrame.

**What it does:**
1. Validates the file extension against `ALLOWED_FILE_TYPE` — raises `ValueError` immediately if unsupported, before touching the file contents
2. Detects character encoding with `chardet` (falls back to `utf-8` if detection fails) — handles files exported from non-English systems that aren't UTF-8
3. Reads the file bytes into a DataFrame via `pd.read_csv()` or `pd.read_excel()` depending on extension, wrapping bytes in `io.BytesIO` so no temp file is written to disk
4. Normalises column names — strips whitespace and lowercases every column so downstream tools never need to worry about casing or padding
5. Samples down to `SAMPLE_SIZE` rows if the file exceeds `LARGE_FILE_THRESHOLD` — uses `random_state=42` for reproducibility; note this is a fallback simple sample, not stratified (stratified sampling happens in the planner when `target_column` is known)

**Key patterns:**
- `io.BytesIO(file_bytes)` — pandas `read_csv`/`read_excel` accept file-like objects; wrapping bytes avoids writing a temp file and keeps everything in memory
- `chardet.detect(file_bytes)` before reading — encoding must be known before pandas opens the stream; detecting after would require reading the file twice
- `df.columns = [col.strip().lower() for col in df.columns]` — normalisation happens once here so every tool downstream can assume clean column names
- `any(filename.lower().endswith(ext) for ext in ALLOWED_FILE_TYPE)` — lowercasing the filename before checking handles `.CSV`, `.Xlsx`, etc.

---

## `backend/models/tool_result.py`

**Purpose:** Pydantic model that standardises the return value of every tool function.

**Fields:**

| Field | Type | Description |
|---|---|---|
| `tool_name` | `str` | Name of the tool that produced this result |
| `success` | `bool` | Whether the tool ran without error |
| `payload` | `Dict` | Full computed result — used for report rendering and visualisations |
| `summary` | `str` | ≤500 token compressed version — the only part passed to LLM prompts |
| `error` | `Optional[str]` | Exception message if `success=False`, else `None` |

**Key patterns:**
- `error: Optional[str] = None` — default avoids having to pass `error=None` on every successful return
- `payload` and `summary` are always populated even on failure (empty dict and empty string) — prevents downstream code from needing to guard against missing keys

---

## `backend/tools/dataset_inspector.py`

**Purpose:** Computes shape, dtypes, descriptive stats, and sample rows from a DataFrame and returns a `ToolResult`.

**What it does:**
1. Builds `payload` with four keys: `shape` (rows/columns dict), `dtypes` (col → dtype string), `describe` (full `df.describe(include="all")` as dict), `sample` (3 random rows as dict)
2. Builds a compressed `summary` — one line per column: numeric columns get mean ± std, non-numeric columns get unique value count
3. Returns a successful `ToolResult` on completion, or a failure `ToolResult` with `str(e)` if anything raises

**Key patterns:**
- `df.describe(include="all")` — passing `include="all"` ensures non-numeric columns appear in describe output (with `count`, `unique`, `top`, `freq` instead of mean/std)
- `payload["describe"].get(col, {})` — `.get()` with default `{}` handles columns that describe might not include; then `"mean" in desc` distinguishes numeric vs non-numeric
- Summary is built from `payload` data, not by re-querying `df` — avoids redundant computation (except `df[col].nunique()` for non-numeric, which isn't in describe)
- `random_state=42` on `df.sample()` — makes sample rows reproducible across runs

---

## `backend/tools/missing_value_analyzer.py`

**Purpose:** Counts and ranks missing values per column, filters to only columns with >1% missing, and returns a `ToolResult`.

**What it does:**
1. Computes `missing_count` per column using `df.isnull().sum()`
2. Computes `missing_pct` as `(missing_count / len(df)) * 100`
3. Filters `missing_pct` to columns where percentage > 1% and sorts descending
4. Filters `missing_count` to the same surviving columns using `missing_pct.index`
5. Builds `payload` with `missing_count` (dict), `missing_pct` (dict), and `total_rows` (int)
6. Builds a compact `summary` — header line with counts, then one line per affected column showing count and percentage
7. Returns successful `ToolResult`, or failure `ToolResult` with `str(e)` on error

**Key patterns:**
- Filter `missing_pct` first, then use `missing_pct.index` to align `missing_count` to the same columns — avoids any mismatch between the two Series
- `>1%` threshold is intentional per CLAUDE.md — columns with ≤1% missing are noise, not worth LLM attention
- `total_rows` stored in payload so report renderer can show "X missing out of N rows" without needing the original df
- Summary only lists affected columns — a 50-column dataset with 2 missing columns produces a 3-line summary, not 50 lines

---

## `backend/tools/correlation_analyzer.py`

**Purpose:** Computes the correlation matrix for numeric columns and returns the top-10 pairs by absolute correlation value.

**What it does:**
1. Selects only numeric columns using `df.select_dtypes(include='number')` before calling `.corr()` — avoids errors on string/categorical columns
2. Unstacks the correlation matrix into a Series of (col1, col2) → value pairs
3. Removes duplicate pairs and the diagonal — keeps only pairs where `col1 < col2` alphabetically (since correlation of A↔B equals B↔A, and diagonal is always 1.0)
4. Sorts by absolute value descending and keeps top 10
5. Builds `payload` with the full `corr_matrix` (for visualisations) and the filtered `corr_unstacked` (top-10 pairs)
6. Builds `summary` using a loop over `corr_unstacked.items()` — one line per pair

**Key patterns:**
- `corr_matrix.unstack()` converts the 2D matrix into a flat Series with a MultiIndex of (col1, col2) — makes it easy to sort and filter pairs as a single sequence
- `corr_unstacked.index.get_level_values(0) < corr_unstacked.index.get_level_values(1)` — the clean way to drop duplicates and diagonal without masking or numpy tricks; compares column name strings alphabetically
- `.abs().sort_values(ascending=False).head(10)` — absolute value ensures strong negative correlations (-0.95) rank equally to strong positive ones (0.95)
- `for (col1, col2), val in corr_unstacked.items()` — tuple unpacking on a MultiIndex Series; each key is a `(col1, col2)` tuple, unpacked directly in the `for` statement

---

## `backend/tools/outlier_analyzer.py`

**Purpose:** Detects outliers in numeric columns via IQR or Z-score and returns per-column counts and percentages.

**What it does:**
1. Selects only numeric columns using `df.select_dtypes(include='number')`
2. Computes a boolean `mask` DataFrame using the chosen method:
   - **IQR**: flags values below `Q1 - 1.5×IQR` or above `Q3 + 1.5×IQR`
   - **Z-score**: flags values where `|(value - mean) / std| > 3`
3. Computes `outlier_count` per column using `mask.sum()` (True = 1, False = 0)
4. Computes `outlier_pct` as `outlier_count / len(df) * 100`
5. Builds `payload` with count dict, pct dict, method used, and total rows
6. Builds `summary` — header line with column count, then one line per column that actually has outliers (count > 0)
7. Returns successful `ToolResult`, or failure `ToolResult` with `str(e)` on error

**Key patterns:**
- `mask` is a boolean DataFrame — `True` where a value is an outlier; `.sum()` across columns counts outliers without any explicit loop
- IQR method doesn't assume normality — better for skewed distributions (income, prices); Z-score assumes normality — better for symmetric distributions (height, test scores)
- Z-score threshold of 3 is the standard — only ~0.3% of values in a normal distribution fall outside ±3 std deviations
- `if count > 0` in the summary loop — skips columns with no outliers to keep the summary compact; a 50-column dataset with 2 outlier columns produces a 3-line summary

---

## `backend/tools/visualization_analyzer.py`

**Purpose:** Generates Plotly figures (histograms, heatmap, box plots, missing value bar chart) serialized as JSON for frontend rendering.

**What it does:**
1. Selects numeric columns and generates a histogram per column using `px.histogram`
2. Pulls `corr_matrix` from `tool_results["correlation_analyzer"]` if available and generates a correlation heatmap using `px.imshow`
3. Generates a box plot per numeric column using `px.box`
4. Pulls `missing_pct` from `tool_results["missing_value_analyzer"]` if available and generates a bar chart using `px.bar`
5. Stores all figures as JSON strings in `payload["figures"]` keyed by chart name (e.g. `histogram_age`, `boxplot_salary`)
6. Returns summary listing count and names of all generated figures

**Key patterns:**
- `fig.to_json()` serialises the Plotly figure for HTTP transport — figure objects live in Python memory and can't be sent over the API directly; JSON is deserialisable back to a figure with `pio.from_json()` on the frontend
- `tool_results.get("correlation_analyzer", {}).get("corr_matrix")` — double `.get()` with empty dict fallback means the chart is simply skipped if the tool wasn't run, no crash
- Heatmap uses `color_continuous_scale="RdBu_r"` with `zmin=-1, zmax=1` — red for negative, blue for positive correlations, anchored to the full correlation range
- Takes `tool_results: dict` as a second argument to reuse already-computed data rather than re-querying the df

---

## `backend/tools/notebook_generator.py`

**Purpose:** Assembles a reproducible Jupyter notebook (.ipynb) from the analysis results and saves it to `reports/<report_id>/notebook.ipynb`.

**What it does:**
1. Creates a new notebook object with `nbformat.v4.new_notebook()`
2. Appends cells in order: title markdown → instructions markdown → imports code → data loading code → EDA markdown + code → missing values → correlation → outlier analysis → insights markdown
3. Assigns the cells list to `nb.cells`
4. Creates the output directory using `REPORTS_DIR` from config and writes the notebook with `nbformat.write()`
5. Returns a `ToolResult` with the saved file path in payload

**Key patterns:**
- `nbformat.v4.new_markdown_cell()` and `nbformat.v4.new_code_cell()` — the two cell types used; cell content is plain strings with `\n` for line breaks
- `REPORTS_DIR` from config used for output path — avoids relative `".."` paths that break depending on launch directory
- Data loading cell uses `pd.read_csv(dataset_name)` — user must place their file next to the notebook; instructions markdown cell explains this
- Notebook code cells replicate the same pandas/plotly logic as the backend tools — intentional, so the notebook is self-contained and runnable without the backend

**Known limitation (Day 5 fix):**
- Data loading depends on the user placing the CSV next to the notebook — not fully reproducible out of the box. Day 5 will add an embedded sample data cell (50 rows as a `pd.DataFrame({...})` literal) so the notebook runs without any external file.

---

*Last updated: After completing routes/ (config, app, health, analyze, download) + file_handler.py + tool_result.py + dataset_inspector.py + missing_value_analyzer.py + correlation_analyzer.py + outlier_analyzer.py + visualization_analyzer.py + notebook_generator.py*
