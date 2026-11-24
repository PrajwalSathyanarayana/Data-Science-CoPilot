"""FastAPI/app configuration — loads environment variables and defines runtime constants."""
from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_ENV = os.getenv("ANTHROPIC_API_KEY", default="")
if not ANTHROPIC_ENV:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set. Please set it in the .env file or as an environment variable.")
MAX_FILE_SIZE = int(os.getenv("MAX_UPLOAD_BYTES", default = "52428800"))
APP_ENV = os.getenv("APP_ENV", default = "")
APP_PORT = int(os.getenv("APP_PORT", default = "5000"))

ANALYSIS_MODEL = "claude-opus-5-5" # planner, insight generator, hypothesis agents
CRITIC_MODEL = "claude-haiku-5-5" # critic: review-only, so the cheapest current model

# Per-agent LLM settings. max_tokens caps the OUTPUT, and thinking tokens count toward it,
# so it must be far larger than the visible answer. effort controls how much the model thinks:
# "low" | "medium" | "high" | "xhigh" | "max". Tune these with measured cost and quality.
AGENT_LLM_SETTINGS = {
    "planner": {"model": ANALYSIS_MODEL, "effort": "medium", "max_tokens": 16000},
    "insight_generator": {"model": ANALYSIS_MODEL, "effort": "medium", "max_tokens": 16000},
    "hypothesis": {"model": ANALYSIS_MODEL, "effort": "medium", "max_tokens": 16000},
    "critic": {"model": CRITIC_MODEL, "effort": "low", "max_tokens": 16000},
}
LLM_MAX_RETRIES = 3 # SDK retries rate limits, server errors and network errors with backoff
LLM_TIMEOUT_SECONDS = 120.0 # per attempt

# Input budgets, in real tokens: the most each agent's input may hold before it is trimmed. They are ceilings
# (typical prompts are far smaller and cost the same) and a fuse: a bug or a hostile file can't blow up the cost
# of a call. Set to 4x the first budgets: those assumed 4 characters per token, about 2x off the real count,
# so 4x restates them in real tokens with room to spare. call_llm warns if a real call goes over.
PLANNER_MAX_TOKENS = 14000 # whole planner input: system prompt + context (overview, target check, flags, units, notes, example rows)
INSIGHT_GENERATOR_MAX_TOKENS = 40000 # up to 5 tool summaries + flags + plan
CRITIC_AGENT_MAX_TOKENS = 44000 # insights + tool summaries
HYPOTHESIS_MAX_TOKENS = 12000 # insights + plan + flags
# Each agent's budget, so call_llm can compare the real input count from the API against it after every call
AGENT_INPUT_BUDGETS = {
    "planner": PLANNER_MAX_TOKENS,
    "insight_generator": INSIGHT_GENERATOR_MAX_TOKENS,
    "critic": CRITIC_AGENT_MAX_TOKENS,
    "hypothesis": HYPOTHESIS_MAX_TOKENS,
}

# Tool output
TOOL_SUMMARY_MAX_TOKENS = 3750 # a tool's summary (the part sent to the LLM) must stay under this; about 6,750 characters
INSPECTOR_SAMPLE_ROWS = 3 # example rows kept by dataset_inspector (fewer if the file is smaller)

# Checks that compare columns in pairs (missing together, correlations). Pairs grow with the square of
# the column count (75 columns = 2,775 pairs; 2,000 columns = about 2 million), so these checks look at
# most this many columns. Checks of one column at a time always use every column. Every row is always used.
PAIR_CHECK_MAX_COLUMNS = 75

# missing_value_analyzer
MISSING_HIGHLIGHT_PCT = 1.0 # columns above this % missing get their own summary line; the rest are counted
# Text values that often stand for "no data". In a column where every other value is a number, the loader
# reads them as missing (so the column stays numeric); in text columns they are only counted, never changed.
# (pandas already turns "", "NA", "N/A", "null", "None", "NaN" into real missing values when loading.)
MISSING_PLACEHOLDERS = ["?", "-", "--", "---", ".", "unknown", "unk", "missing", "not available", "n.a.", "undefined", "tbd"]
CO_MISSING_MIN_OVERLAP = 0.8 # a pair is reported when 80%+ of rows missing either column are missing both
CO_MISSING_MIN_ROWS = 5 # ...and at least this many rows are missing both

# correlation_analyzer
CORR_MIN_ROWS = 30 # a pair is measured only when at least this many rows have both values (fewer gives chance "relationships")
CORR_MAX_CATEGORIES = 50 # text columns with more different values than this are not compared (IDs, free text)
CORR_MIN_STRENGTH = 0.3 # relationships listed in the summary must be at least this strong (all are in the payload)
CORR_NEAR_PERFECT = 0.95 # pairs this strong are reported as near-duplicates, or possible leakage when the target is involved
CORR_METHOD_GAP = 0.2 # Pearson and Spearman this far apart: "not a straight line" or "driven by extreme values"
CORR_BANDS = 10 # the curve check splits a number column into this many equal-sized bands
CORR_CURVE_MIN = 0.5 # a pair is reported as a curve when the banded score is at least this...
CORR_CURVE_MARGIN = 0.2 # ...and at least this much higher than both Pearson and Spearman
CORR_SUMMARY_TOP = 10 # most pairs listed per summary section

# outlier_analyzer: one column at a time (box-plot rule), then unusual combinations across columns
OUTLIER_IQR_K = 1.5 # box-plot rule: outside Q1 - 1.5 x IQR or Q3 + 1.5 x IQR. Long-tailed positive columns use the log scale
OUTLIER_MIN_VALUES = 20 # columns with fewer values to check are not judged
# Log scale when quartile skew is above this. Quartile skew uses only the middle 50% of values, so typos can't
# trigger it (two ages of 999 push ordinary skew to 20); 0.1 matches the planner's skew > 1 on typical long-tailed data
OUTLIER_LOG_QUARTILE_SKEW = 0.1
OUTLIER_FEW_VALUES = 10 # columns with this many different values or fewer (ratings, small counts) only flag values that are...
OUTLIER_RARE_PCT = 1.0 # ...outside the normal range AND rarer than this % of checked values (a 1-star review is not an error; a 200-item basket is)
OUTLIER_DOMINANT_SHARE = 0.5 # when one value fills this share of a column (e.g. mostly 0), the other values are checked among themselves
OUTLIER_EXTREMES_SHOWN = 3 # most extreme values listed per column in the summary
OUTLIER_ROWS_KEPT = 100 # flagged row numbers kept per column (and for the row check) for the report; never sent to the LLM
ROW_CHECK_MAX_COLUMNS = 20 # most number columns used together by the unusual-combination check
ROW_CHECK_FIT_ROWS = 5000 # the normal "shape" is learned from this many random rows (fixed seed); every row is then checked
ROW_CHECK_MIN_ROWS = 100 # the row check needs at least this many complete rows, and 10 per column used
ROW_CHECK_MAX_MISSING_PCT = 50.0 # columns missing more than this % are left out of the row check (they'd leave too few complete rows)
ROW_CHECK_CUTOFF = 0.999 # a row is unusual when its distance is beyond this share of what normal rows reach
ROW_CHECK_WARN_PCT = 5.0 # far above the ~0.1% expected: the columns likely follow a curve or groups the check can't model

# visualization_analyzer: charts are counted by pandas, so their size doesn't grow with the number of rows
CHART_MAX_TOTAL = 60 # most charts per report (overview, target, relationships, outliers, time, distributions - in that order)
CHART_MAX_DISTRIBUTIONS = 30 # one distribution chart per column, for at most this many columns
CHART_MAX_BOXES = 10 # box plots for the columns most affected by outliers
CHART_HIST_BINS = 40 # bars per histogram
CHART_MAX_BARS = 20 # most groups or values shown in a bar chart; the rest are grouped as "other"
CHART_SCATTER_SAMPLE = 2000 # scatter plots of bigger files show a random sample of this many rows (fixed seed)...
# ...plus every row the outlier tool flagged, so the chart always shows what the summary reports
CHART_MAX_TIME_POINTS = 100 # time charts use day, week, month, quarter or year, whichever gives at most this many points
CHART_MAX_DATE_COLUMNS = 3 # most date columns given a time chart

AGENT_STEPS_MAX = 10 # most tool runs per report; the orchestrator finishes early instead of starting more
AGENT_MEMORY_MAX = 24000 # most tokens of tool summaries collected per report; past it, no new tools start
# How long to wait for one tool, in seconds. About 10x the slowest tool we measured (correlations took 6.24 s
# on 75,000 rows x 61 columns). A slower tool is marked as failed and the report goes on without it.
TOOL_TIMEOUT_SECONDS = 60.0

SAFETY_MARGIN = 0.20 # safety margin below each agent's input budget, since token counts are estimated
CHARS_PER_TOKEN = 1.8 # measured with count_tokens: 2.4-2.6 on text-heavy prompts, 1.89 on number-heavy ones; just below the worst case

# Order in which tool summaries are kept when an agent's budget is tight (most important first).
# Tools not listed here are kept last, in the order they ran.
TOOL_PRIORITY = [
    "dataset_inspector",
    "missing_value_analyzer",
    "outlier_analyzer",
    "correlation_analyzer",
    "visualization_analyzer",
    "notebook_generator",
]

# The analysis tools the orchestrator always runs, all 3 every time. dataset_inspector runs before
# the planner; visualization_analyzer runs after these (its charts follow their findings); notebook_generator
# runs after the agents. Tool choice by the planner returns in Phase 3.
PLANNER_TOOLS = [
    "missing_value_analyzer",
    "outlier_analyzer",
    "correlation_analyzer",
]

# Autonomous column detection (runs before the planner)
DATE_PARSE_SAMPLE_SIZE = 100 # how many non-empty values to try parsing as dates per text column
DATE_PARSE_MIN_RATIO = 0.9 # share of sampled values that must parse for the column to count as a date
SKEW_THRESHOLD = 1.0 # numeric column is flagged as long-tailed when abs(skewness) is above this (1.0 = "highly skewed")
SKEW_MIN_VALUES = 20 # skewness is unreliable on very few values, so smaller columns are not checked
IMBALANCE_THRESHOLD = 0.15 # binary column is flagged when its smaller class is below this share of rows

# Planner (create_plan). Problem type: Python's rule settles the clear cases; the LLM only picks for "unclear" ones
PROBLEM_UNCLEAR_MAX_VALUES = 10 # whole-number columns with this many different values or fewer could be classes or amounts
TARGET_MAX_CLASSES = 20 # text columns with more different values than this can't be a target (IDs, free text)
PLANNER_FLAGS_SHOWN = 15 # flags shown in full and numbered for ranking; the rest are grouped on one line per kind
PLANNER_FOCUS_MAX = 8 # most focus columns kept (the LLM lists the most important first; extras are cut, no retry)
PLANNER_RATIONALE_MAX_CHARS = 600 # longer rationales are cut (it is shown in the report)
PLANNER_REASON_MAX_CHARS = 200 # the one-line reason for an unclear problem type
PLANNER_RETRY_ERRORS_SHOWN = 5 # most validation errors listed in the correction prompt
PROMPT_CELL_MAX_CHARS = 30 # example-row cells are cut to this length in prompts (long text, injection attempts)
PROMPT_COLUMN_NAME_MAX_CHARS = 60 # column names are cut to this length where the planner context lists them


ALLOWED_FILE_TYPE = ['csv', 'xlsx', 'xls']

# File loading. Every row is always kept: all statistics are computed on the full data.
CSV_SNIFF_BYTES = 65536 # how much of a CSV is inspected to detect its separator (, ; tab |)
ENCODING_SAMPLE_BYTES = 100000 # how much of a non-UTF-8 file is scanned to guess its encoding

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", default = "http://localhost:8501").split(',')

