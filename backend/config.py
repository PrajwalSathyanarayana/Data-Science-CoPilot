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

# Input budgets: the most tokens of tool summaries token_manager may send to each agent
PLANNER_MAX_TOKENS = 800
INSIGHT_GENERATOR_MAX_TOKENS = 3000
CRITIC_AGENT_MAX_TOKENS = 2000
HYPOTHESIS_MAX_TOKENS = 1000

# Tool output
TOOL_SUMMARY_MAX_TOKENS = 500 # a tool's summary (the part sent to the LLM) must stay under this
INSPECTOR_SAMPLE_ROWS = 3 # example rows kept by dataset_inspector (fewer if the file is smaller)

AGENT_STEPS_MAX = 10
AGENT_MEMORY_MAX = 6000 # accumulated maximum tokens for agent's memory.

SAFETY_MARGIN = 0.20 # safety margin below each agent's input budget, since token counts are estimated
CHARS_PER_TOKEN = 4 # rough rule of thumb for English text with Claude's tokenizer

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

# Tools the planner is allowed to schedule. dataset_inspector always runs before the planner,
# and notebook_generator runs after the agents, so neither is the planner's choice.
PLANNER_TOOLS = [
    "missing_value_analyzer",
    "outlier_analyzer",
    "correlation_analyzer",
    "visualization_analyzer",
]

# Autonomous column detection (runs before the planner)
DATE_PARSE_SAMPLE_SIZE = 100 # how many non-empty values to try parsing as dates per text column
DATE_PARSE_MIN_RATIO = 0.9 # share of sampled values that must parse for the column to count as a date
SKEW_THRESHOLD = 1.0 # numeric column is flagged as long-tailed when abs(skewness) is above this (1.0 = "highly skewed")
SKEW_MIN_VALUES = 20 # skewness is unreliable on very few values, so smaller columns are not checked
IMBALANCE_THRESHOLD = 0.15 # binary column is flagged when its smaller class is below this share of rows


ALLOWED_FILE_TYPE = ['csv', 'xlsx', 'xls']

# File loading. Every row is always kept: all statistics are computed on the full data.
CSV_SNIFF_BYTES = 65536 # how much of a CSV is inspected to detect its separator (, ; tab |)
ENCODING_SAMPLE_BYTES = 100000 # how much of a non-UTF-8 file is scanned to guess its encoding

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", default = "http://localhost:8501").split(',')

