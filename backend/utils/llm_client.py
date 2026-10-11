"""Utility — thin wrapper around the Claude API: per-agent model settings, refusal handling, and per-call metrics."""
import logging
import time
from functools import lru_cache
from typing import Any, Optional

import anthropic

from ..config import AGENT_LLM_SETTINGS, LLM_MAX_RETRIES, LLM_TIMEOUT_SECONDS
from ..models import LLMResponse

logger = logging.getLogger(__name__)

# Models that support Anthropic's server-side refusal fallback. On these, a declined request is
# re-run on a suitable model inside the same call. Claude Haiku 5.5 has no server-side fallback.
SERVER_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMCallError(Exception):
    """The call failed: API error after the SDK's retries, or an unusable response."""


class LLMRefusalError(LLMCallError):
    """The model (and any fallback model) declined the request. Callers decide whether to skip or fail."""


@lru_cache(maxsize=1)
def _default_client() -> anthropic.Anthropic:
    # Created on first use, not at import, so importing this module never needs network or credentials.
    # The SDK retries rate limits (429), server errors (5xx) and network errors with backoff.
    return anthropic.Anthropic(max_retries=LLM_MAX_RETRIES, timeout=LLM_TIMEOUT_SECONDS)


def _send(client: anthropic.Anthropic, model: str, request: dict) -> Any:
    """Send the request, adding the server-side fallback where the model supports it."""
    if model in SERVER_FALLBACK_MODELS:
        return client.beta.messages.create(**request, betas=[SERVER_FALLBACK_BETA], fallbacks="default")
    return client.messages.create(**request)


def call_llm(
    system_prompt: str,
    user_message: str,
    agent: str,
    client: Optional[anthropic.Anthropic] = None,
) -> LLMResponse:
    """Call Claude with the settings configured for `agent` and return the answer text plus metrics.

    Raises:
        ValueError: `agent` has no entry in AGENT_LLM_SETTINGS.
        LLMRefusalError: the request was declined, even after any fallback.
        LLMCallError: the API failed after retries, or the answer was cut off at max_tokens.
    """
    if agent not in AGENT_LLM_SETTINGS:
        raise ValueError(f"No LLM settings for agent '{agent}'. Known agents: {sorted(AGENT_LLM_SETTINGS)}")
    settings = AGENT_LLM_SETTINGS[agent]
    model = settings["model"]

    # Thinking is always on for these models, so no `thinking` parameter; effort sets how much
    request = {
        "model": model,
        "max_tokens": settings["max_tokens"],
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
        "output_config": {"effort": settings["effort"]},
    }

    started = time.perf_counter()
    try:
        response = _send(client or _default_client(), model, request)
    except anthropic.APIConnectionError as exc:
        # Network failures and timeouts, after the SDK's own retries
        raise LLMCallError(f"[{agent}] could not reach the Claude API: {exc}") from exc
    except anthropic.APIStatusError as exc:
        # Permanent errors (bad request, auth, unknown model) fail at once; transient ones were already retried
        raise LLMCallError(f"[{agent}] Claude API returned {exc.status_code}: {exc.message}") from exc
    latency_ms = round((time.perf_counter() - started) * 1000)

    # Check why it stopped BEFORE reading content: a refusal can have empty or partial content
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None)
        raise LLMRefusalError(f"[{agent}] {model} declined the request (category: {category})")
    if response.stop_reason == "max_tokens":
        raise LLMCallError(
            f"[{agent}] answer was cut off at max_tokens={settings['max_tokens']}; raise it in AGENT_LLM_SETTINGS"
        )

    # The first block is the model's thinking, so collect the text blocks by type, never content[0]
    text = "".join(block.text for block in response.content if block.type == "text")

    # A fallback model answered if the usage breakdown includes a fallback step
    iterations = getattr(response.usage, "iterations", None) or []
    fallback_used = any(getattr(step, "type", None) == "fallback_message" for step in iterations)

    result = LLMResponse(
        text=text,
        agent=agent,
        model=response.model,
        stop_reason=response.stop_reason,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=latency_ms,
        fallback_used=fallback_used,
    )
    logger.info(
        "LLM call agent=%s model=%s in=%d out=%d latency_ms=%d fallback=%s",
        agent, result.model, result.input_tokens, result.output_tokens, latency_ms, fallback_used,
    )
    return result
