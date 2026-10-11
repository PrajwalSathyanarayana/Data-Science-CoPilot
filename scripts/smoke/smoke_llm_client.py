"""Smoke test for backend/utils/llm_client.py. Makes TWO REAL API calls (costs a few cents).

Needs ANTHROPIC_API_KEY in .env. Run from the project root:
    python -m scripts.smoke.smoke_llm_client
"""
from backend.config import AGENT_LLM_SETTINGS
from backend.utils.llm_client import LLMCallError, call_llm
from scripts.smoke.common import SmokeRunner

SYSTEM_PROMPT = "You are a test endpoint. Reply with exactly the word OK and nothing else."


def check_agent(runner: SmokeRunner, agent: str) -> None:
    settings = AGENT_LLM_SETTINGS[agent]
    runner.section(f"{agent}: {settings['model']} at effort={settings['effort']}")
    try:
        result = call_llm(SYSTEM_PROMPT, "Health check.", agent=agent)
    except LLMCallError as exc:
        runner.check("call succeeded", "no error", f"{type(exc).__name__}: {exc}")
        return

    # Exact token counts and latency vary per call, so these are shown for you to read, not compared
    print(f"         text:          {result.text!r}")
    print(f"         input_tokens:  {result.input_tokens}")
    print(f"         output_tokens: {result.output_tokens} (includes thinking)")
    print(f"         latency_ms:    {result.latency_ms}")

    runner.check("finished normally", "end_turn", result.stop_reason)
    runner.check("answered OK", True, "OK" in result.text.upper())
    runner.check("answered by the configured model", settings["model"], result.model)
    runner.check("no fallback needed", False, result.fallback_used)
    runner.check("token counts recorded", True, result.input_tokens > 0 and result.output_tokens > 0)


def main() -> None:
    runner = SmokeRunner("llm_client (real API calls)")
    check_agent(runner, "planner")  # Opus 5.5, via the server-side fallback path
    check_agent(runner, "critic")   # Haiku 5.5, via the plain path
    runner.finish()


if __name__ == "__main__":
    main()
