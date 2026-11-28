"""Agent — orchestrator: takes one uploaded file through every analysis step and collects the results.

The steps, in order:
1. Overview of the data (dataset_inspector)
2. Spot columns worth attention (dates, long tails, rare classes)
3. Ask the planner for a target, focus columns and unit names
4. If the planner's unit names turned text columns into numbers, redo the overview for the new data
5. Run the missing-value, outlier and correlation tools at the same time
6. Draw the charts

Safety limits, so one upload can never run away:
- at most AGENT_STEPS_MAX tool runs
- once the tool summaries get too long (AGENT_MEMORY_MAX tokens), no new tool starts
- each tool gets TOOL_TIMEOUT_SECONDS; a slower tool is marked as failed and the report goes on without it

Running the 3 tools at the same time is safe because they only read the data, never change it. Their results
are always put in the same fixed order, no matter which one finished first.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import anthropic
import pandas as pd

from ..config import AGENT_MEMORY_MAX, AGENT_STEPS_MAX, PLANNER_TOOLS, TOOL_TIMEOUT_SECONDS
from ..models import AnalysisPlan, AnalysisRun, ColumnFlag, ToolResult
from ..tools.correlation_analyzer import analyze_correlations
from ..tools.dataset_inspector import inspect_dataset
from ..tools.missing_value_analyzer import analyze_missing_values
from ..tools.outlier_analyzer import analyze_outliers
from ..tools.visualization_analyzer import analyze_visualizations
from ..utils.file_handler import LoadedFile
from ..utils.token_manager import estimate_tokens, sort_by_priority
from .planner import create_plan, detect_interesting_columns

logger = logging.getLogger(__name__)

INSPECTOR = "dataset_inspector"
CHARTS = "visualization_analyzer"

# The 3 analysis tools by name. Each is called the same way: (data, target column, focus columns)
AnalysisTool = Callable[[pd.DataFrame, Optional[str], Sequence[str]], ToolResult]
ANALYSIS_TOOLS: Dict[str, AnalysisTool] = {
    "missing_value_analyzer": lambda df, _target, _focus: analyze_missing_values(df),
    "outlier_analyzer": lambda df, target, focus: analyze_outliers(df, target_column=target, focus_columns=focus),
    "correlation_analyzer": lambda df, target, focus: analyze_correlations(df, target_column=target, focus_columns=focus),
}


def _failed(tool_name: str, error: str) -> ToolResult:
    return ToolResult(tool_name=tool_name, success=False, payload={}, summary="", error=error)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _timed(job: Callable[[], ToolResult]) -> Tuple[ToolResult, int]:
    start = time.perf_counter()
    result = job()
    return result, _elapsed_ms(start)


def _run_tools(jobs: Mapping[str, Callable[[], ToolResult]]) -> Tuple[List[ToolResult], Dict[str, int]]:
    """Run the jobs at the same time and wait at most TOOL_TIMEOUT_SECONDS for them.

    Returns each job's result (in the order given) and how long each took in milliseconds.
    A job that is too slow or crashes becomes a failed result instead of stopping the report.
    Python can't stop a running thread, so a slow job keeps going in the background and its
    answer is ignored; that's why we don't wait for it when closing the pool.
    """
    pool = ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="tool")
    results: List[ToolResult] = []
    timings: Dict[str, int] = {}
    try:
        futures = {name: pool.submit(_timed, job) for name, job in jobs.items()}
        deadline = time.monotonic() + TOOL_TIMEOUT_SECONDS
        for name, future in futures.items():
            try:
                result, timings[name] = future.result(timeout=max(0.0, deadline - time.monotonic()))
            except FutureTimeout:
                logger.warning("Tool %s took longer than %g s; left out of this report", name, TOOL_TIMEOUT_SECONDS)
                result = _failed(name, f"took longer than {TOOL_TIMEOUT_SECONDS:g} seconds, so it was left out of this report")
            except Exception as exc:  # tools handle their own errors; this is a backup so one tool can't stop the report
                logger.exception("Tool %s stopped with an error", name)
                result = _failed(name, f"stopped with an error: {exc}")
            results.append(result)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return results, timings


class _Limits:
    """Keeps count of tool runs and summary length, and remembers why the run stopped early, if it did."""

    def __init__(self) -> None:
        self.steps = 0
        self.finish_reason: Optional[str] = None

    def can_start(self, count: int, what: str) -> bool:
        if self.finish_reason is not None:
            return False
        if self.steps + count > AGENT_STEPS_MAX:
            self.finish_reason = f"the limit of {AGENT_STEPS_MAX} tool runs was reached, so {what} did not run"
            logger.warning("Run finished early: %s", self.finish_reason)
            return False
        self.steps += count
        return True

    def check_memory(self, results: Sequence[ToolResult], next_steps: str) -> None:
        """Called after each step: once the summaries are too long, no new tool starts."""
        used = sum(estimate_tokens(result.summary) for result in results)
        if self.finish_reason is None and used > AGENT_MEMORY_MAX:
            self.finish_reason = f"tool summaries reached {used:,} of {AGENT_MEMORY_MAX:,} tokens, so {next_steps} did not run"
            logger.warning("Run finished early: %s", self.finish_reason)


def _refresh_flags(
    df: pd.DataFrame,
    plan: AnalysisPlan,
    converted: Sequence[str],
    target_column: Optional[str],
    date_dayfirst: Optional[bool],
) -> None:
    """Check the newly converted columns again and replace their old flags with new ones.

    Only these columns changed, so all other flags stay as they were. The planner never saw the new
    flags, so they go at the end of the list.
    """
    positions = [i for i, name in enumerate(df.columns) if name in converted]
    target = target_column if target_column in converted else None
    new_flags = detect_interesting_columns(df.iloc[:, positions], target, date_dayfirst)
    if target_column is not None and target is None:
        # When the user picked a target, only that column is checked for rare classes
        new_flags = [flag for flag in new_flags if flag.kind != "class_imbalance"]
    kept: List[ColumnFlag] = [flag for flag in plan.flags if flag.column not in converted]
    plan.flags = kept + new_flags


def run_analysis(
    loaded: LoadedFile,
    target_column: Optional[str] = None,
    date_dayfirst: Optional[bool] = None,
    client: Optional[anthropic.Anthropic] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> AnalysisRun:
    """Take one uploaded file through every step and return everything it produced.

    If the planner fails, a simpler plan made by Python is used instead. If a tool fails, it is kept
    as a failed result and the other steps still run. `progress`, if given, is called with each step's
    name in plain words, for the progress display.

    Raises:
        ValueError: `target_column` is not a column of the dataset.
    """
    df = loaded.df
    if target_column is not None and target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' is not a column in the dataset")

    run_start = time.perf_counter()
    limits = _Limits()
    timings: Dict[str, int] = {}
    results: Dict[str, ToolResult] = {}
    report = progress or (lambda step: None)

    # 1. Overview of the data
    report("looking at the data")
    if limits.can_start(1, "the overview"):
        (results[INSPECTOR],), tool_times = _run_tools({INSPECTOR: lambda: inspect_dataset(df)})
        timings.update(tool_times)
        limits.check_memory(list(results.values()), "the analysis tools and the charts")

    # 2 and 3. Spot columns worth attention, then ask the planner
    start = time.perf_counter()
    flags = detect_interesting_columns(df, target_column, date_dayfirst)
    timings["detection"] = _elapsed_ms(start)

    report("planning the analysis")
    start = time.perf_counter()
    plan_result = create_plan(loaded, results.get(INSPECTOR), flags, target_column=target_column, client=client)
    timings["planner"] = _elapsed_ms(start)
    plan = plan_result.plan

    # 4. Some text columns became numbers, so the overview and their flags are out of date: redo them
    converted = list(plan.unit_meanings)
    if converted:
        if INSPECTOR in results and limits.can_start(1, "the refreshed overview"):
            (refreshed,), tool_times = _run_tools({INSPECTOR: lambda: inspect_dataset(df)})
            results[INSPECTOR] = refreshed
            timings["dataset_inspector_refresh"] = tool_times.get(INSPECTOR, 0)
        start = time.perf_counter()
        _refresh_flags(df, plan, converted, target_column, date_dayfirst)
        timings["detection_refresh"] = _elapsed_ms(start)
        logger.info("Refreshed the overview and flags for %d converted column(s)", len(converted))
    limits.check_memory(list(results.values()), "the analysis tools and the charts")

    # 5. The 3 analysis tools at the same time, using the plan's target and focus columns
    if limits.can_start(len(PLANNER_TOOLS), "the analysis tools and the charts"):
        report("running the analysis tools")
        target, focus = plan.target_column, list(plan.focus_columns)
        jobs = {name: (lambda tool=ANALYSIS_TOOLS[name]: tool(df, target, focus)) for name in PLANNER_TOOLS}
        start = time.perf_counter()
        tool_results, tool_times = _run_tools(jobs)
        timings["analysis_tools"] = _elapsed_ms(start)
        timings.update(tool_times)
        results.update({result.tool_name: result for result in tool_results})
        limits.check_memory(sort_by_priority(list(results.values())), "the charts")

    # 6. Charts, based on what the successful tools found
    if limits.can_start(1, "the charts"):
        report("drawing the charts")
        payloads = {name: result.payload for name, result in results.items() if result.success}
        (results[CHARTS],), tool_times = _run_tools({
            CHARTS: lambda: analyze_visualizations(
                df,
                tool_payloads=payloads,
                target_column=plan.target_column,
                focus_columns=plan.focus_columns,
                column_flags=plan.flags,
                date_dayfirst=date_dayfirst,
            )
        })
        timings.update(tool_times)
        limits.check_memory(list(results.values()), "the remaining steps")

    timings["total"] = _elapsed_ms(run_start)
    ordered = sort_by_priority(list(results.values()))
    failed = [result.tool_name for result in ordered if not result.success]
    logger.info(
        "Analysis run: %d tool(s), %d failed %s, plan from %s, %d LLM call(s), %d ms%s",
        len(ordered), len(failed), failed, plan.plan_source, len(plan_result.llm_calls), timings["total"],
        f", finished early ({limits.finish_reason})" if limits.finish_reason else "",
    )
    return AnalysisRun(
        plan=plan,
        tool_results=ordered,
        llm_calls=plan_result.llm_calls,
        loading_notes=loaded.notes(),
        unit_notes=plan_result.unit_notes,
        finished_early=limits.finish_reason is not None,
        finish_reason=limits.finish_reason,
        step_timings_ms=timings,
    )
