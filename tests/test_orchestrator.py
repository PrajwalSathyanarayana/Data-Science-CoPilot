"""Tests for backend/agents/orchestrator.py — scripted fake LLM, so no real API calls and no cost."""
import threading
import time

import numpy as np
import pytest
from pydantic import ValidationError

from backend.agents import orchestrator
from backend.agents.orchestrator import run_analysis
from backend.config import TOOL_PRIORITY
from backend.models import AnalysisPlan, AnalysisRun, ToolResult
from backend.utils.file_handler import load_file
from tests.llm_fakes import ScriptedClient, answer, make_response

WEATHER_ANSWER = dict(
    rationale="Storm warnings are rare (4.1%), so the imbalance matters most; rainfall is long-tailed.",
    focus_columns=["storm_warning", "rainfall_mm", "observed_on"],
    suggested_target="storm_warning",
    flag_order=[1, 3, 2],
)
TEMPS_ANSWER = dict(
    rationale="Temperatures mix Celsius and Fahrenheit.",
    focus_columns=["temperature_reading"],
    unit_meanings={"temperature_reading": {"C": "degC", "F": "degF"}},
    flag_order=[],
)
ALL_TOOLS = ["dataset_inspector", "missing_value_analyzer", "outlier_analyzer", "correlation_analyzer", "visualization_analyzer"]


def weather():
    with open("tests/sample_data/weather_ddmm.csv", "rb") as handle:
        return load_file(handle.read(), "weather_ddmm.csv")


def mixed_temperatures():
    """300 long-tailed temperatures: 200 written in C, 100 in F. Only the planner can name C and F."""
    celsius = np.random.default_rng(1).lognormal(2.5, 0.6, 300).round(1)
    cells = [f"{v}C" if i % 3 else f"{round(v * 9 / 5 + 32, 1)}F" for i, v in enumerate(celsius)]
    rows = "\n".join(f"{cell},{'abc'[i % 3]}" for i, cell in enumerate(cells))
    return load_file(("temperature_reading,city\n" + rows).encode(), "temps.csv")


def weather_run(**kwargs) -> AnalysisRun:
    return run_analysis(weather(), client=ScriptedClient([answer(**WEATHER_ANSWER)]), **kwargs)


def fake_tool(name, delay=0.0, summary="ok", error=None, calls=None):
    """An analysis tool that waits `delay` seconds, then succeeds (or raises `error`)."""
    def tool(df, target, focus):
        if calls is not None:
            calls.append((name, target, list(focus)))
        time.sleep(delay)
        if error is not None:
            raise error
        return ToolResult(tool_name=name, success=True, payload={}, summary=summary)
    return tool


# --- a normal run ---

def test_weather_run_has_every_tool_in_priority_order():
    run = weather_run()

    assert [r.tool_name for r in run.tool_results] == ALL_TOOLS
    assert all(r.success for r in run.tool_results)
    assert run.plan.plan_source == "planner" and run.plan.target_column == "storm_warning"
    assert len(run.llm_calls) == 1
    assert run.finished_early is False and run.finish_reason is None


def test_timings_are_recorded_for_every_step_and_tool():
    timings = weather_run().step_timings_ms
    expected = {"detection", "planner", "analysis_tools", "total", *ALL_TOOLS}
    assert expected <= set(timings)
    assert all(ms >= 0 for ms in timings.values())


def test_order_does_not_depend_on_which_thread_finished_first(monkeypatch):
    # The first tool in TOOL_PRIORITY finishes last
    for delay, name in zip((0.3, 0.1, 0.0), ["missing_value_analyzer", "outlier_analyzer", "correlation_analyzer"]):
        monkeypatch.setitem(orchestrator.ANALYSIS_TOOLS, name, fake_tool(name, delay))
    names = [r.tool_name for r in weather_run().tool_results]
    assert names == sorted(names, key=TOOL_PRIORITY.index)


def test_analysis_tools_run_in_parallel(monkeypatch):
    for name in list(orchestrator.ANALYSIS_TOOLS):
        monkeypatch.setitem(orchestrator.ANALYSIS_TOOLS, name, fake_tool(name, delay=0.3))
    assert weather_run().step_timings_ms["analysis_tools"] < 800  # one after another would be 900+


def test_tools_get_the_plans_target_and_focus_columns(monkeypatch):
    calls = []
    for name in list(orchestrator.ANALYSIS_TOOLS):
        monkeypatch.setitem(orchestrator.ANALYSIS_TOOLS, name, fake_tool(name, calls=calls))
    weather_run()
    assert {(target, tuple(focus)) for _, target, focus in calls} == {
        ("storm_warning", ("storm_warning", "rainfall_mm", "observed_on"))
    }


def test_unknown_target_is_rejected_before_any_work():
    client = ScriptedClient([])
    with pytest.raises(ValueError, match="'nope' is not a column"):
        run_analysis(weather(), target_column="nope", client=client)
    assert client.requests == []


def test_planner_fallback_still_runs_every_tool():
    run = run_analysis(weather(), client=ScriptedClient([make_response("", stop_reason="refusal")]))
    assert run.plan.plan_source == "fallback"
    assert [r.tool_name for r in run.tool_results] == ALL_TOOLS


# --- tool failures ---

def test_slow_tool_is_left_out_and_the_report_goes_on(monkeypatch):
    release = threading.Event()

    def stuck(df, target, focus):
        release.wait(5)  # released at the end of the test, so no thread is left behind
        return ToolResult(tool_name="correlation_analyzer", success=True, payload={}, summary="late")

    monkeypatch.setattr(orchestrator, "TOOL_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setitem(orchestrator.ANALYSIS_TOOLS, "correlation_analyzer", stuck)
    try:
        start = time.perf_counter()
        run = weather_run()
        waited = time.perf_counter() - start
    finally:
        release.set()

    results = {r.tool_name: r for r in run.tool_results}
    assert results["correlation_analyzer"].success is False
    assert "took longer than 0.2 seconds" in results["correlation_analyzer"].error
    assert results["outlier_analyzer"].success and results["visualization_analyzer"].success
    assert waited < 3  # did not wait for the stuck tool
    assert run.finished_early is False  # a failed tool is not a safety stop


def test_tool_that_raises_becomes_a_failed_result(monkeypatch):
    monkeypatch.setitem(orchestrator.ANALYSIS_TOOLS, "outlier_analyzer",
                        fake_tool("outlier_analyzer", error=RuntimeError("boom")))
    results = {r.tool_name: r for r in weather_run().tool_results}
    assert results["outlier_analyzer"].success is False
    assert results["outlier_analyzer"].error == "stopped with an error: boom"
    assert results["visualization_analyzer"].success


# --- safety limits ---

def test_memory_guard_stops_new_tools_and_says_why(monkeypatch):
    # weather with the plan's target: inspector 190 + missing 30 + outliers 190 + correlations 227 = 637; charts would add 236
    monkeypatch.setattr(orchestrator, "AGENT_MEMORY_MAX", 500)
    run = weather_run()

    assert [r.tool_name for r in run.tool_results] == ALL_TOOLS[:4]  # nothing computed is thrown away
    assert run.finished_early is True
    assert run.finish_reason == "tool summaries reached 637 of 500 tokens, so the charts did not run"


def test_memory_guard_after_the_inspector_skips_the_analysis_tools(monkeypatch):
    monkeypatch.setattr(orchestrator, "AGENT_MEMORY_MAX", 100)
    run = weather_run()
    assert [r.tool_name for r in run.tool_results] == ["dataset_inspector"]
    assert run.finish_reason.endswith("so the analysis tools and the charts did not run")
    assert run.plan.plan_source == "planner"  # the plan is still made


def test_step_limit_stops_before_tools_that_would_pass_it(monkeypatch):
    monkeypatch.setattr(orchestrator, "AGENT_STEPS_MAX", 2)  # inspector = 1; the 3 tools would make 4
    run = weather_run()
    assert [r.tool_name for r in run.tool_results] == ["dataset_inspector"]
    assert run.finish_reason == "the limit of 2 tool runs was reached, so the analysis tools and the charts did not run"


def test_finish_reason_must_match_finished_early():
    plan = weather_run().plan
    with pytest.raises(ValidationError, match="finish_reason must be set exactly when finished_early is True"):
        AnalysisRun(plan=plan, finished_early=True)
    with pytest.raises(ValidationError):
        AnalysisRun(plan=plan, finish_reason="why")


# --- redo the overview after the planner's unit names convert a column ---

def test_converted_column_gets_a_fresh_overview_and_flags():
    loaded = mixed_temperatures()
    assert loaded.mixed_units == {"temperature_reading": {"F": 100, "C": 200}}

    run = run_analysis(loaded, client=ScriptedClient([answer(**TEMPS_ANSWER)]))

    overview = run.tool_results[0].summary
    assert "temperature_reading (numeric): mean 13.43, median 11.79" in overview
    assert "temperature_reading (text)" not in overview
    # Text can't be skewed, so this flag only exists after the refresh
    assert [(f.kind, f.column) for f in run.plan.flags] == [("skewed_distribution", "temperature_reading")]
    assert {"dataset_inspector_refresh", "detection_refresh"} <= set(run.step_timings_ms)
    assert "temperature_reading: converted to C (100 from F); units read by planner as C = degC, F = degF" in run.loading_notes


def test_nothing_is_refreshed_when_nothing_was_converted():
    timings = weather_run().step_timings_ms
    assert "dataset_inspector_refresh" not in timings and "detection_refresh" not in timings


def test_refresh_keeps_other_flags_and_puts_new_ones_last():
    loaded = mixed_temperatures()
    run = run_analysis(loaded, client=ScriptedClient([answer(**TEMPS_ANSWER)]))
    plan: AnalysisPlan = run.plan
    kept = plan.flags[0].model_copy(update={"column": "city", "kind": "date_column", "value": None})
    plan.flags = [kept]

    orchestrator._refresh_flags(loaded.df, plan, ["temperature_reading"], None, None)
    assert [(f.kind, f.column) for f in plan.flags] == [("date_column", "city"), ("skewed_distribution", "temperature_reading")]
