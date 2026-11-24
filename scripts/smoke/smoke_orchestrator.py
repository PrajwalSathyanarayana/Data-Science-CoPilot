"""Smoke test for run_analysis() in backend/agents/orchestrator.py, on real files.

Run from the project root (scripted fake LLM, no API cost):
    python -m scripts.smoke.smoke_orchestrator
"""
import logging

import numpy as np
import pandas as pd

from backend.agents import orchestrator
from backend.agents.orchestrator import run_analysis
from backend.config import AGENT_MEMORY_MAX, TOOL_TIMEOUT_SECONDS
from backend.utils.file_handler import LoadedFile, load_file
from backend.utils.token_manager import estimate_tokens
from scripts.smoke.common import SmokeRunner
from tests.llm_fakes import ScriptedClient, answer, make_response

ALL_TOOLS = ["dataset_inspector", "missing_value_analyzer", "outlier_analyzer", "correlation_analyzer", "visualization_analyzer"]
WEATHER_ANSWER = dict(
    rationale="Storm warnings are rare (4.1%), so that imbalance matters most.",
    focus_columns=["storm_warning", "rainfall_mm", "observed_on"],
    suggested_target="storm_warning",
    flag_order=[1, 3, 2],
)


def weather() -> LoadedFile:
    with open("tests/sample_data/weather_ddmm.csv", "rb") as handle:
        return load_file(handle.read(), "weather_ddmm.csv")


def mixed_temperatures() -> LoadedFile:
    celsius = np.random.default_rng(1).lognormal(2.5, 0.6, 300).round(1)
    cells = [f"{v}C" if i % 3 else f"{round(v * 9 / 5 + 32, 1)}F" for i, v in enumerate(celsius)]
    rows = "\n".join(f"{cell},{'abc'[i % 3]}" for i, cell in enumerate(cells))
    return load_file(("temperature_reading,city\n" + rows).encode(), "temps.csv")


def wide_file() -> LoadedFile:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"spend_{i:02d}": rng.lognormal(3, 1, 75000).round(2) for i in range(60)})
    df.loc[rng.random(75000) < 0.05, "spend_03"] = np.nan
    df["churned"] = np.where(rng.random(75000) < 0.06, "yes", "no")
    return LoadedFile(df=df, file_type="csv", original_columns=list(df.columns))


def main() -> None:
    logging.disable(logging.WARNING)
    runner = SmokeRunner("orchestrator (run_analysis)")

    runner.section("weather_ddmm.csv: planner picks the target, every tool runs")
    run = run_analysis(weather(), client=ScriptedClient([answer(**WEATHER_ANSWER)]))
    runner.check("tools in priority order", ALL_TOOLS, [r.tool_name for r in run.tool_results])
    runner.check("all succeeded", True, all(r.success for r in run.tool_results))
    runner.check("target, problem type", ("storm_warning", "classification"), (run.plan.target_column, run.plan.problem_type))
    runner.check("summary tokens per tool", [190, 30, 190, 227, 236], [estimate_tokens(r.summary) for r in run.tool_results])
    runner.check("finished early", False, run.finished_early)
    runner.check("no refresh (nothing converted)", False, "dataset_inspector_refresh" in run.step_timings_ms)

    runner.section("planner refuses: fallback plan, report goes on")
    run = run_analysis(weather(), client=ScriptedClient([make_response("", stop_reason="refusal")]))
    runner.check("plan source", "fallback", run.plan.plan_source)
    runner.check("tools that ran", ALL_TOOLS, [r.tool_name for r in run.tool_results])

    runner.section("300 temperatures in C and F: the planner names the units, then a refresh")
    loaded = mixed_temperatures()
    runner.check("mixed units before the plan", {"temperature_reading": {"F": 100, "C": 200}}, loaded.mixed_units)
    temps_answer = answer(rationale="Temperatures mix C and F.", focus_columns=["temperature_reading"],
                          unit_meanings={"temperature_reading": {"C": "degC", "F": "degF"}}, flag_order=[])
    run = run_analysis(loaded, client=ScriptedClient([temps_answer]))
    overview_line = run.tool_results[0].summary.splitlines()[1]
    runner.check("overview line after the refresh",
                 "- temperature_reading (numeric): mean 13.43, median 11.79, min 2.389, max 78.28", overview_line)
    runner.check("new flag for the converted column", [("skewed_distribution", "temperature_reading")],
                 [(f.kind, f.column) for f in run.plan.flags])

    runner.section(f"memory guard (limit lowered from {AGENT_MEMORY_MAX:,} to 500 for this check)")
    orchestrator.AGENT_MEMORY_MAX = 500
    try:
        run = run_analysis(weather(), client=ScriptedClient([answer(**WEATHER_ANSWER)]))
    finally:
        orchestrator.AGENT_MEMORY_MAX = AGENT_MEMORY_MAX
    runner.check("tools kept", ALL_TOOLS[:4], [r.tool_name for r in run.tool_results])
    runner.check("reason", "tool summaries reached 637 of 500 tokens, so the charts did not run", run.finish_reason)

    runner.section("wide file, 75,000 rows x 61 columns (timings vary by machine)")
    wide_answer = answer(rationale="Churn is rare.", focus_columns=["churned"], suggested_target="churned", flag_order=[])
    run = run_analysis(wide_file(), client=ScriptedClient([wide_answer]))
    runner.check("all succeeded", True, all(r.success for r in run.tool_results))
    timings = run.step_timings_ms
    slowest = max(timings[name] for name in orchestrator.ANALYSIS_TOOLS)
    runner.check("3 tools together took no longer than the slowest + 1 s (ran in parallel)",
                 True, timings["analysis_tools"] <= slowest + 1000)
    runner.check(f"every tool under the {TOOL_TIMEOUT_SECONDS:g} s limit", True,
                 all(timings[name] < TOOL_TIMEOUT_SECONDS * 1000 for name in ALL_TOOLS))
    runner.check("total summary tokens under the memory limit", True,
                 sum(estimate_tokens(r.summary) for r in run.tool_results) < AGENT_MEMORY_MAX)
    print("\n  timings (ms): " + ", ".join(f"{k} {v:,}" for k, v in timings.items()))
    print("  summary tokens: " + ", ".join(f"{r.tool_name} {estimate_tokens(r.summary):,}" for r in run.tool_results))

    runner.finish()


if __name__ == "__main__":
    main()
