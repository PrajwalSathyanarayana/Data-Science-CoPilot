"""Tests for backend/utils/llm_client.py — uses a fake client, so no real API calls and no cost."""
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from backend.config import AGENT_LLM_SETTINGS, ANALYSIS_MODEL, CRITIC_MODEL, LLM_MAX_RETRIES, LLM_TIMEOUT_SECONDS
from backend.utils import llm_client
from backend.utils.llm_client import LLMCallError, LLMRefusalError, call_llm

API_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def make_response(
    texts=("Hello",),
    stop_reason="end_turn",
    model=ANALYSIS_MODEL,
    input_tokens=120,
    output_tokens=450,
    iterations=None,
    stop_details=None,
):
    """A response shaped like the SDK's: a thinking block first, then the text blocks."""
    content = [SimpleNamespace(type="thinking", thinking="")]
    content += [SimpleNamespace(type="text", text=t) for t in texts]
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=stop_details,
        model=model,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens, iterations=iterations),
    )


class FakeClient:
    """Records every request and returns (or raises) whatever the test sets up."""

    def __init__(self, result):
        self.result = result
        self.calls = []
        self.messages = SimpleNamespace(create=self._create("plain"))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create("beta")))

    def _create(self, path):
        def create(**kwargs):
            self.calls.append((path, kwargs))
            if isinstance(self.result, Exception):
                raise self.result
            return self.result
        return create


# --- normal answers ---

def test_text_is_read_from_text_blocks_not_the_first_block():
    client = FakeClient(make_response(texts=("Revenue is ", "skewed.")))
    result = call_llm("system", "question", agent="planner", client=client)

    assert result.text == "Revenue is skewed."


def test_metrics_are_recorded():
    client = FakeClient(make_response(input_tokens=321, output_tokens=987))
    result = call_llm("system", "question", agent="planner", client=client)

    assert (result.agent, result.model, result.stop_reason) == ("planner", ANALYSIS_MODEL, "end_turn")
    assert (result.input_tokens, result.output_tokens) == (321, 987)
    assert result.latency_ms >= 0
    assert result.fallback_used is False


def test_request_uses_the_agents_configured_settings():
    client = FakeClient(make_response())
    call_llm("the system prompt", "the question", agent="planner", client=client)

    _, request = client.calls[0]
    settings = AGENT_LLM_SETTINGS["planner"]
    assert request["model"] == settings["model"]
    assert request["max_tokens"] == settings["max_tokens"]
    assert request["output_config"] == {"effort": settings["effort"]}
    assert request["system"] == "the system prompt"
    assert request["messages"] == [{"role": "user", "content": "the question"}]
    assert "thinking" not in request  # always on for these models; sending "disabled" would be a 400


# --- which API path each model uses ---

def test_opus_uses_server_side_fallback():
    client = FakeClient(make_response())
    call_llm("system", "question", agent="planner", client=client)

    path, request = client.calls[0]
    assert path == "beta"
    assert request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]


def test_haiku_critic_uses_plain_call_without_fallback():
    # Haiku 5.5 has no server-side fallback; sending one would be pointless or a 400
    client = FakeClient(make_response(model=CRITIC_MODEL))
    call_llm("system", "question", agent="critic", client=client)

    path, request = client.calls[0]
    assert path == "plain"
    assert request["model"] == CRITIC_MODEL
    assert "fallbacks" not in request and "betas" not in request


def test_fallback_answer_is_reported():
    iterations = [SimpleNamespace(type="message"), SimpleNamespace(type="fallback_message")]
    client = FakeClient(make_response(model="claude-opus-5", iterations=iterations))
    result = call_llm("system", "question", agent="planner", client=client)

    assert result.fallback_used is True
    assert result.model == "claude-opus-5"  # the model that actually answered


# --- unusable answers ---

def test_refusal_raises_refusal_error_with_category():
    response = make_response(texts=(), stop_reason="refusal", stop_details=SimpleNamespace(category="bio"))
    with pytest.raises(LLMRefusalError, match="category: bio"):
        call_llm("system", "question", agent="critic", client=FakeClient(response))


def test_refusal_error_is_also_an_llm_call_error():
    # Callers that only catch LLMCallError still catch refusals
    assert issubclass(LLMRefusalError, LLMCallError)


def test_cut_off_answer_raises_instead_of_returning_partial_text():
    response = make_response(texts=('{"steps": ["missing_val',), stop_reason="max_tokens")
    with pytest.raises(LLMCallError, match="cut off at max_tokens"):
        call_llm("system", "question", agent="planner", client=FakeClient(response))


# --- errors ---

@pytest.mark.parametrize(
    "error, message",
    [
        (anthropic.BadRequestError("bad model", response=httpx2.Response(400, request=API_REQUEST), body=None),
         "returned 400"),
        (anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=API_REQUEST), body=None),
         "returned 429"),
        (anthropic.APIConnectionError(request=API_REQUEST), "could not reach"),
        (anthropic.APITimeoutError(request=API_REQUEST), "could not reach"),
    ],
    ids=["bad_request", "rate_limited_after_retries", "network", "timeout"],
)
def test_api_errors_become_llm_call_error_with_cause(error, message):
    with pytest.raises(LLMCallError, match=message) as raised:
        call_llm("system", "question", agent="planner", client=FakeClient(error))
    assert raised.value.__cause__ is error  # original error kept for debugging


def test_unknown_agent_raises():
    with pytest.raises(ValueError, match="No LLM settings for agent 'summariser'"):
        call_llm("system", "question", agent="summariser", client=FakeClient(make_response()))


# --- default client ---

def test_default_client_uses_configured_retries_and_timeout(monkeypatch):
    created = {}

    def fake_anthropic(**kwargs):
        created.update(kwargs)
        return FakeClient(make_response())

    llm_client._default_client.cache_clear()
    monkeypatch.setattr(llm_client.anthropic, "Anthropic", fake_anthropic)
    try:
        call_llm("system", "question", agent="planner")
    finally:
        llm_client._default_client.cache_clear()

    assert created == {"max_retries": LLM_MAX_RETRIES, "timeout": LLM_TIMEOUT_SECONDS}
