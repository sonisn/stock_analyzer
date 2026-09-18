"""LLM provider abstraction — wraps `agno.Agent` for Claude, Gemini, and OpenAI."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from agno.agent import Agent
from agno.exceptions import (
    ModelAuthenticationError,
    ModelProviderError,
    RemoteServerUnavailableError,
)
from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.run.base import RunStatus

from .logging import get_logger
from .usage import BUDGET, TRACKER

logger = get_logger(__name__)

Provider = Literal["claude", "gemini", "openai"]

_MODEL_REGISTRY: dict[Provider, type] = {
    "claude": Claude,
    "gemini": Gemini,
    "openai": OpenAIChat,
}

# Errors worth retrying on a fallback provider: auth failures, rate limits,
# 5xx/overload responses, and the remote endpoint being unreachable. Anything
# else (bad input, context-window overflow, schema validation) would fail
# identically on the fallback provider too, so it isn't worth the extra call.
_FALLBACK_ERRORS: tuple[type[Exception], ...] = (
    ModelAuthenticationError,
    ModelProviderError,
    RemoteServerUnavailableError,
)


# Output ceiling assumed for the cost cap when a call sets no max-tokens
# field (agno's Claude default is 8192).
_DEFAULT_OUTPUT_ALLOWANCE = 8192


class AgnoAgent:
    """Factory wrapper around `agno.agent.Agent` supporting multiple providers."""

    def __init__(
        self,
        name: str,
        provider: Provider,
        model: str,
        *,
        model_kwargs: dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        if provider not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported provider {provider!r}. Expected one of {sorted(_MODEL_REGISTRY)}."
            )

        self.name = name
        self.provider: Provider = provider
        self.model_id = model
        # For the cost cap's worst-case estimate: system prompt size and the
        # most output the model is allowed to produce on one call.
        kw = model_kwargs or {}
        self._max_output_tokens = int(
            kw.get("max_tokens")
            or kw.get("max_completion_tokens")
            or kw.get("max_output_tokens")
            or _DEFAULT_OUTPUT_ALLOWANCE
        )
        self._instruction_chars = len(str(agent_kwargs.get("instructions") or ""))

        model_cls = _MODEL_REGISTRY[provider]
        self._model = model_cls(id=model, **(model_kwargs or {}))
        self.agent = Agent(name=name, model=self._model, **agent_kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        prompt_chars = self._instruction_chars + sum(
            len(a) for a in (*args, *kwargs.values()) if isinstance(a, str)
        )
        with BUDGET.hold(self.name, self.model_id, prompt_chars, self._max_output_tokens):
            result = self.agent.run(*args, **kwargs)
            TRACKER.record(self.name, self.model_id, getattr(result, "metrics", None))
        # agno 3.0's Agent.run() no longer raises once it has exhausted its
        # own internal retries (Model.retries, default 0 — so effectively
        # on the very first non-retryable provider error, e.g. an HTTP 404
        # for a bad model id). Instead it swallows the exception and
        # returns a RunOutput with status=RunStatus.error and content set
        # to str(exception) — which would otherwise fail Pydantic
        # validation against whatever output_schema was expected, instead
        # of surfacing as the provider error it actually is. Every caller
        # in this codebase goes through this method, so raising here once
        # is what makes every existing `except Exception` wrapper (and
        # run_with_fallback below) work the way it did under agno 2.x.
        if getattr(result, "status", None) == RunStatus.error:
            message = str(getattr(result, "content", None) or "Agent.run() returned status=ERROR")
            raise ModelProviderError(message=message, model_name=self.name, model_id=self.model_id)
        return result

    def print_response(self, *args: Any, **kwargs: Any) -> Any:
        return self.agent.print_response(*args, **kwargs)


def reasoning_model_kwargs(
    provider: Provider,
    effort: str,
    *,
    temperature: float = 1,
    max_tokens: int = 16000,
) -> dict[str, Any]:
    """Provider-specific kwargs for a high-effort reasoning call.

    Each provider's agno model class names its reasoning/thinking knob (and
    its max-output-tokens field) differently, and not every provider accepts
    an arbitrary `temperature` alongside reasoning effort — so this can't be
    one dict passed unconditionally to all three, the way `Ranker`/`RedTeam`
    used to (Claude-only) before other providers were wired in.
    """
    if provider == "claude":
        # Adaptive thinking pins temperature at 1; the API rejects any other
        # value, so `temperature` is intentionally not threaded through here.
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
            "max_tokens": max_tokens,
            "temperature": 1,
        }
    if provider == "openai":
        # Reasoning-tier OpenAI models reject a caller-set temperature, so it
        # is omitted rather than passed and rejected by the API. Reasoning-
        # tier models (this is always a reasoning call — see
        # `reasoning_effort` above) also reject plain `max_tokens`: OpenAI
        # requires `max_completion_tokens` instead once a model reasons,
        # since that budget covers hidden reasoning tokens too, not just
        # the visible output. `max_tokens` fails with a 400
        # "Unsupported parameter" — confirmed against a real call.
        return {
            "reasoning_effort": effort,
            "max_completion_tokens": max_tokens,
        }
    if provider == "gemini":
        return {
            "thinking_level": "low" if effort == "low" else "high",
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
    raise ValueError(f"Unsupported provider {provider!r}.")


def deterministic_model_kwargs(provider: Provider) -> dict[str, Any]:
    """Kwargs for a plain (non-thinking) low-temperature call.

    Claude 5-generation models (confirmed on claude-sonnet-5) reject an
    explicit `temperature` outside of adaptive-thinking mode — a 400
    "`temperature` is deprecated for this model" — so it's omitted
    entirely for claude and the model runs at its own default instead.
    (claude-haiku-4-5 still accepts temperature=0 fine as of this writing,
    but the safer default going forward is to not pass it for any claude
    model, since Anthropic is clearly phasing this out model-by-model and
    there's no cheap way to know in advance which model id will reject it
    next.) Gemini/OpenAI still accept and want an explicit temperature=0
    for deterministic, non-reasoning calls.
    """
    if provider == "claude":
        return {}
    return {"temperature": 0}


def run_with_fallback(
    primary: AgnoAgent,
    build_fallback: Callable[[], AgnoAgent] | None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run `primary.run(...)`, retrying once on a fallback provider.

    Only retries on auth/rate-limit/provider/connectivity errors — anything
    else (bad input, schema mismatch) would fail identically on the fallback
    provider, so it isn't retried.
    """
    try:
        return primary.run(*args, **kwargs)
    except _FALLBACK_ERRORS as e:
        if build_fallback is None:
            raise
        logger.warning(
            "%s call failed on %s/%s (%s) — retrying on fallback provider",
            primary.name,
            primary.provider,
            primary.model_id,
            e,
        )
        fallback = build_fallback()
        return fallback.run(*args, **kwargs)
