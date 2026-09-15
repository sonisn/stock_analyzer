"""Provider-specific reasoning kwargs and the fallback wrapper.

Ranker/RedTeam/Sizer used to build one Claude-only `model_kwargs` dict
unconditionally; adding Gemini/OpenAI rounds means each provider's agno
model class needs its own kwargs (different field names, different
supported knobs). A wrong branch here breaks a whole provider silently
(TypeError on an unexpected kwarg), so each is asserted directly.
"""

from __future__ import annotations

from stock_analyzer.llm import AgnoAgent, reasoning_model_kwargs, run_with_fallback


def test_claude_kwargs_use_adaptive_thinking_and_pin_temperature():
    kwargs = reasoning_model_kwargs("claude", "high", temperature=0.3, max_tokens=1234)
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}
    assert kwargs["max_tokens"] == 1234
    # Adaptive thinking requires temperature=1 regardless of what's passed.
    assert kwargs["temperature"] == 1


def test_openai_kwargs_use_reasoning_effort_and_max_completion_tokens():
    # Reasoning-tier OpenAI models reject `max_tokens` (400 "Unsupported
    # parameter") and require `max_completion_tokens` instead — confirmed
    # against a real API call.
    kwargs = reasoning_model_kwargs("openai", "medium", max_tokens=999)
    assert kwargs == {"reasoning_effort": "medium", "max_completion_tokens": 999}
    assert "temperature" not in kwargs
    assert "thinking" not in kwargs
    assert "max_tokens" not in kwargs


def test_gemini_kwargs_use_thinking_level_and_max_output_tokens():
    kwargs = reasoning_model_kwargs("gemini", "high", temperature=0.7, max_tokens=555)
    assert kwargs["thinking_level"] == "high"
    assert kwargs["max_output_tokens"] == 555
    assert kwargs["temperature"] == 0.7
    # Gemini's field name differs from Claude/OpenAI's max_tokens.
    assert "max_tokens" not in kwargs


def test_gemini_low_effort_maps_to_low_thinking_level():
    assert reasoning_model_kwargs("gemini", "low")["thinking_level"] == "low"


# --- run_with_fallback -------------------------------------------------


class _FakeAgent:
    def __init__(self, provider, name, *, fails=False, result="ok"):
        self.provider = provider
        self.name = name
        self.model_id = "m"
        self._fails = fails
        self._result = result
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        if self._fails:
            from agno.exceptions import ModelProviderError

            raise ModelProviderError("boom", status_code=502)
        return self._result


def test_run_with_fallback_uses_primary_when_it_succeeds():
    primary = _FakeAgent("claude", "Ranker")
    result = run_with_fallback(primary, lambda: _FakeAgent("gemini", "Ranker"), "prompt")
    assert result == "ok"
    assert primary.calls == 1


def test_run_with_fallback_retries_on_fallback_after_provider_error():
    primary = _FakeAgent("claude", "Ranker", fails=True)
    fallback = _FakeAgent("gemini", "Ranker", result="fallback-ok")
    result = run_with_fallback(primary, lambda: fallback, "prompt")
    assert result == "fallback-ok"
    assert primary.calls == 1
    assert fallback.calls == 1


def test_run_with_fallback_reraises_when_no_fallback_given():
    import pytest
    from agno.exceptions import ModelProviderError

    primary = _FakeAgent("claude", "Ranker", fails=True)
    with pytest.raises(ModelProviderError):
        run_with_fallback(primary, None, "prompt")


def test_agno_agent_accepts_openai_provider():
    # Construction only — no network call. Confirms the registry wiring
    # (Provider Literal + _MODEL_REGISTRY) actually includes "openai".
    agent = AgnoAgent("Test", "openai", "gpt-5.4-mini", model_kwargs={"reasoning_effort": "low"})
    assert agent.provider == "openai"


# --- AgnoAgent.run() raising on agno 3.0's swallowed-error responses ---


def test_agno_agent_run_raises_when_agno_returns_status_error():
    """agno 3.0's Agent.run() stopped raising once it exhausts its own
    internal retries — it now returns a RunOutput with status=ERROR and
    content=str(exception) instead. Confirmed against a real 404
    (nonexistent Gemini model id) in production: without this check,
    that string gets fed straight into `RankerOutput.model_validate_json`
    and fails with a confusing "Field required: picks, full_text" error
    instead of the actual provider error. AgnoAgent.run() must convert
    that back into a raised ModelProviderError so every existing
    except-based error path (run_with_fallback, and every stage's own
    `except Exception` wrapper) keeps working."""
    import pytest
    from agno.exceptions import ModelProviderError
    from agno.run.base import RunStatus

    class _FakeRunOutput:
        status = RunStatus.error
        content = '{"error": {"code": 404, "message": "model not found"}}'

    agent = AgnoAgent("Test", "claude", "claude-haiku-4-5")
    agent.agent.run = lambda *a, **k: _FakeRunOutput()

    with pytest.raises(ModelProviderError, match="model not found"):
        agent.run("prompt")


def test_agno_agent_run_passes_through_on_success():
    from agno.run.base import RunStatus

    class _FakeRunOutput:
        status = RunStatus.completed
        content = "all good"

    agent = AgnoAgent("Test", "claude", "claude-haiku-4-5")
    agent.agent.run = lambda *a, **k: _FakeRunOutput()

    result = agent.run("prompt")
    assert result.content == "all good"
