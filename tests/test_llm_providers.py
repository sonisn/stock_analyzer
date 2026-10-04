"""The model layer (llm.py): settings per provider, fallback, the output
ceiling, validation retries and the cost ledger. No network: the provider
model is scripted (tests/llm_fakes.py).

A wrong provider mapping breaks a whole provider silently (a 400 on a
field it doesn't take), so each provider's request fields are asserted
directly.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.usage import RequestUsage

from stock_analyzer import llm
from stock_analyzer.llm import (
    CallSettings,
    InvalidOutputError,
    ModelAgent,
    OutputTruncatedError,
    deterministic_settings,
    fallback_builder,
    model_settings_for,
    reasoning_settings,
    run_with_fallback,
)
from stock_analyzer.usage import TRACKER

from .llm_fakes import script


class _Out(BaseModel):
    x: int


# --- request fields per provider ----------------------------------------------


def test_claude_reasoning_sends_thinking_effort_and_no_temperature():
    s = model_settings_for("claude", reasoning_settings("high", max_tokens=1234, temperature=0.3))
    # Pydantic AI turns `thinking` into adaptive thinking at that effort.
    assert s == {"max_tokens": 1234, "thinking": "high"}


def test_openai_reasoning_sends_no_temperature():
    # Reasoning-tier OpenAI models reject a caller-set temperature.
    s = model_settings_for("openai", reasoning_settings("medium", max_tokens=999))
    assert s == {"max_tokens": 999, "thinking": "medium"}


def test_gemini_reasoning_keeps_its_temperature():
    s = model_settings_for("gemini", reasoning_settings("high", max_tokens=555, temperature=0.7))
    assert s == {"max_tokens": 555, "thinking": "high", "temperature": 0.7}


def test_deterministic_calls_send_temperature_zero_except_to_claude():
    # Claude gets the 8k ceiling a plain answer always had, not the model max.
    assert model_settings_for("claude", deterministic_settings()) == {"max_tokens": 8192}
    assert model_settings_for("openai", deterministic_settings()) == {"temperature": 0}
    assert model_settings_for("gemini", deterministic_settings()) == {"temperature": 0}


def test_prompt_caching_is_a_claude_setting():
    cached = deterministic_settings(cache_instructions=True)
    assert model_settings_for("claude", cached)["anthropic_cache_instructions"] is True
    assert "anthropic_cache_instructions" not in model_settings_for("gemini", cached)


def test_an_unknown_effort_is_refused():
    with pytest.raises(ValueError, match="effort"):
        reasoning_settings("extreme")


def test_an_unknown_provider_is_refused():
    with pytest.raises(ValueError, match="provider"):
        ModelAgent("T", "mistral", "m")  # ty: ignore[invalid-argument-type]


def test_every_provider_builds_without_a_network_call():
    for provider in ("claude", "gemini", "openai"):
        agent = ModelAgent("T", provider, "some-model", settings=reasoning_settings("low"))
        assert agent.provider == provider


# --- running -------------------------------------------------------------------


def test_a_structured_answer_comes_back_validated(monkeypatch):
    seen = script(monkeypatch, [('{"x": 3}', "stop")])
    agent = ModelAgent("T", "claude", "claude-sonnet-5", output_schema=_Out)
    result = agent.run("prompt")
    assert result.content == _Out(x=3)
    assert result.finish_reason == "stop"
    assert seen.calls == 1


def test_a_text_agent_returns_text(monkeypatch):
    script(monkeypatch, [("plain words", "stop")])
    assert ModelAgent("T", "claude", "m").run("p").content == "plain words"


def test_an_invalid_answer_goes_back_to_the_model_once(monkeypatch):
    seen = script(monkeypatch, [("not json", "stop"), ('{"x": 5}', "stop")])
    agent = ModelAgent("T", "claude", "m", output_schema=_Out)
    assert agent.run("p").content == _Out(x=5)
    assert seen.calls == 2


def test_an_answer_still_invalid_after_the_retry_raises_with_its_text(monkeypatch):
    script(monkeypatch, [("nope", "stop"), ("still nope", "stop")])
    agent = ModelAgent("T", "claude", "m", output_schema=_Out)
    with pytest.raises(InvalidOutputError) as e:
        agent.run("p")
    assert e.value.raw_text == "still nope"


# --- the output ceiling -------------------------------------------------------------


def test_a_structured_answer_cut_off_at_the_ceiling_raises_and_is_not_retried(monkeypatch):
    """A cut-off JSON document used to come back as a plain string and read
    downstream as an empty result. Retrying under the same ceiling would
    pay for the same cut-off again."""
    seen = script(monkeypatch, [('{"x": 1, "prose": "sell MR', "length")])
    agent = ModelAgent(
        "T", "claude", "m", output_schema=_Out, settings=CallSettings(max_tokens=16000)
    )
    with pytest.raises(OutputTruncatedError) as e:
        agent.run("p")
    assert e.value.raw_text.startswith('{"x": 1')
    assert e.value.max_output_tokens == 16000
    assert seen.calls == 1


def test_a_cut_off_text_answer_is_returned_with_its_stop_reason(monkeypatch):
    script(monkeypatch, [("half a sente", "length")])
    result = ModelAgent("T", "claude", "m").run("p")
    assert (result.content, result.finish_reason) == ("half a sente", "length")


# --- the cost ledger ------------------------------------------------------------------


def test_every_response_is_recorded_with_cached_input_split_out(monkeypatch):
    TRACKER.reset()
    usage = RequestUsage(input_tokens=1000, cache_read_tokens=800, output_tokens=50)
    script(monkeypatch, [("bad", "stop"), ('{"x": 1}', "stop")], usage=usage)
    ModelAgent("Sizer", "claude", "claude-opus-5", output_schema=_Out).run("p")
    (row,) = TRACKER.rows()
    # Both responses count, the rejected one included.
    assert (row.stage, row.model, row.calls) == ("Sizer", "claude-opus-5", 2)
    assert (row.input_tokens, row.cache_read_tokens, row.output_tokens) == (400, 1600, 100)
    TRACKER.reset()


# --- fallback --------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self, provider, *, error=None, result="ok"):
        self.provider = provider
        self.name = "Ranker"
        self.model_id = "m"
        self._error = error
        self._result = result
        self.calls = 0

    def run(self, prompt, **_kw):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._result


def _outage():
    return ModelHTTPError(status_code=502, model_name="m", body="bad gateway")


def test_run_with_fallback_uses_primary_when_it_succeeds():
    primary = _FakeAgent("claude")
    assert run_with_fallback(primary, lambda: _FakeAgent("gemini"), "prompt") == "ok"
    assert primary.calls == 1


def test_run_with_fallback_retries_on_fallback_after_an_outage():
    primary = _FakeAgent("claude", error=_outage())
    fallback = _FakeAgent("gemini", result="fallback-ok")
    assert run_with_fallback(primary, lambda: fallback, "prompt") == "fallback-ok"
    assert (primary.calls, fallback.calls) == (1, 1)


def test_run_with_fallback_reraises_when_no_fallback_given():
    with pytest.raises(ModelHTTPError):
        run_with_fallback(_FakeAgent("claude", error=_outage()), None, "prompt")


@pytest.mark.parametrize(
    "error",
    [
        ModelHTTPError(status_code=400, model_name="m", body="prompt is too long"),
        OutputTruncatedError("Ranker", 100, ""),
        InvalidOutputError("Ranker", "bad", ""),
    ],
)
def test_errors_the_fallback_would_repeat_are_not_retried(error):
    fallback = _FakeAgent("gemini")
    with pytest.raises(type(error)):
        run_with_fallback(_FakeAgent("claude", error=error), lambda: fallback, "prompt")
    assert fallback.calls == 0


def test_fallback_builder_skips_a_missing_or_same_provider_fallback():
    built = []

    def build(provider, model):
        built.append((provider, model))
        return "agent"

    assert fallback_builder(None, "claude", build) is None
    assert fallback_builder(("claude", "claude-sonnet-5"), "claude", build) is None
    make = fallback_builder(("gemini", "gemini-pro-latest"), "claude", build)
    assert built == []  # built only when the primary fails
    assert make() == "agent" and built == [("gemini", "gemini-pro-latest")]


def test_each_thread_gets_its_own_model_client():
    """An SDK client's connection pool belongs to the event loop that opened
    it, and each thread runs its own loop."""
    import threading

    here = llm.thread_model("claude", "claude-haiku-4-5")
    assert llm.thread_model("claude", "claude-haiku-4-5") is here
    other = []
    t = threading.Thread(
        target=lambda: other.append(llm.thread_model("claude", "claude-haiku-4-5"))
    )
    t.start()
    t.join()
    assert other[0] is not here
