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

ANALYSIS_MODEL = "claude-sonnet-4-20250514"
CRITIC_MODEL = "claude-haiku-4-5-20251001"

LLM_MAX_TOKENS = 1000
PLANNER_MAX_TOKENS = 800
INSIGHT_GENERATOR_MAX_TOKENS = 3000
CRITIC_AGENT_MAX_TOKENS = 2000
HYPOTHESIS_MAX_TOKENS = 1000

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


ALLOWED_FILE_TYPE = ['csv', 'xlsx', 'xls']

LARGE_FILE_THRESHOLD = 100000 # more than 100k rows, then sample down to 50k before passing it to LLM
SAMPLE_SIZE = 50000 

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", default = "http://localhost:8501").split(',')

