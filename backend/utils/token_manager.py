"""Utility — compresses tool summaries to fit each agent's token budget before an LLM call."""
import logging
import math
from typing import List

from ..config import CHARS_PER_TOKEN, SAFETY_MARGIN, TOOL_PRIORITY
from ..models import CompressedContext, ToolResult

logger = logging.getLogger(__name__)

SECTION_SEPARATOR = "\n\n"


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token). Rounds up so we never under-count."""
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _sort_by_priority(tool_results: List[ToolResult]) -> List[ToolResult]:
    """Most important tools first. Unknown tools go last, keeping the order they ran in."""
    rank = {name: i for i, name in enumerate(TOOL_PRIORITY)}
    return sorted(tool_results, key=lambda r: rank.get(r.tool_name, len(rank)))


def _format_section(result: ToolResult) -> str:
    """Turn one tool result into a titled block. Failed tools get a one-line note so the LLM knows they failed."""
    if result.success:
        body = result.summary.strip()
    else:
        body = f"Tool failed: {result.error or 'unknown error'}"
    return f"## {result.tool_name}\n{body}"


def _trim_to_budget(section: str, max_tokens: int) -> str:
    """Keep whole lines from the top until the budget runs out.

    Tools put their most important lines first, so the top of a summary is what we want to keep.
    Returns an empty string if only the title would fit, since a title alone tells the LLM nothing.
    """
    kept_lines: List[str] = []
    used = 0
    for line in section.splitlines():
        cost = estimate_tokens(line + "\n")
        if used + cost > max_tokens:
            break
        kept_lines.append(line)
        used += cost

    return "\n".join(kept_lines) if len(kept_lines) > 1 else ""


def fit_summaries_to_budget(tool_results: List[ToolResult], max_tokens: int) -> CompressedContext:
    """Build prompt-ready text from tool summaries without going over an agent's token budget.

    Summaries are added in priority order. The first one that doesn't fully fit is trimmed
    line by line to fill the remaining space, and everything after it is dropped.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")

    # Leave headroom because token counts are estimates
    budget = int(max_tokens * (1 - SAFETY_MARGIN))
    separator_cost = estimate_tokens(SECTION_SEPARATOR)

    ordered = _sort_by_priority(tool_results)
    all_sections = [_format_section(r) for r in ordered]
    tokens_before = estimate_tokens(SECTION_SEPARATOR.join(all_sections))

    kept: List[str] = []
    included: List[str] = []
    truncated: List[str] = []
    dropped: List[str] = []
    used = 0
    budget_full = False

    for result, section in zip(ordered, all_sections):
        if budget_full:
            dropped.append(result.tool_name)
            continue

        sep_cost = separator_cost if kept else 0
        remaining = budget - used - sep_cost
        cost = estimate_tokens(section)

        # Whole summary fits
        if cost <= remaining:
            kept.append(section)
            included.append(result.tool_name)
            used += cost + sep_cost
            continue

        # Doesn't fit: keep what we can of this one, then stop adding more
        budget_full = True
        trimmed = _trim_to_budget(section, remaining)
        if trimmed:
            kept.append(trimmed)
            truncated.append(result.tool_name)
            used += estimate_tokens(trimmed) + sep_cost
        else:
            dropped.append(result.tool_name)

    text = SECTION_SEPARATOR.join(kept)
    context = CompressedContext(
        text=text,
        budget=budget,
        tokens_before=tokens_before,
        tokens_after=estimate_tokens(text),
        included_tools=included,
        truncated_tools=truncated,
        dropped_tools=dropped,
    )

    if truncated or dropped:
        logger.info(
            "Compressed tool summaries %d -> %d tokens (budget %d); truncated=%s dropped=%s",
            context.tokens_before, context.tokens_after, budget, truncated, dropped,
        )
    return context
