"""LLM layer: one `ModelAgent` over Pydantic AI for Claude, Gemini and OpenAI.

Every deciding and helper call on these three providers goes through
`ModelAgent.run`. OpenRouter's open models run on Pydantic AI too, through
`openrouter.py`, which adds their host allowlist, daily cap and billed-cost
ledger. What this layer adds on top of Pydantic AI:

  - `CallSettings`, one provider-neutral description of a call (output
    ceiling, thinking effort, prompt caching, retries), translated here
    into each provider's request fields;
  - the run's cost cap (`usage.BUDGET`) and per-stage token ledger
    (`usage.TRACKER`), recorded per model response so retried and failed
    calls are counted too;
  - `OutputTruncatedError` the moment a response stops at its output
    ceiling: a cut-off structured answer is never a result, and it is
    never retried (another attempt under the same ceiling is the same
    bill for the same cut-off);
  - one retry on a fallback provider for outages (`run_with_fallback`);
  - `check`: a stage's own rules about its answer (picks drawn from the
    candidates, a sell only of something held...). An answer that breaks
    one goes back to the model once, with the problems listed; one that
    still breaks it is returned anyway and logged, so the caller's
    deterministic safeguards decide — a rule never costs a stage its answer.

Structured answers use the provider's native JSON-schema output, not a
forced tool call: Claude answers a forced tool call without thinking,
which would quietly turn every high-effort stage into a no-thinking one.
A structured answer that fails validation is sent back to the model once
with the validation errors (`output_retries`) before the call fails.
"""

from __future__ import annotations

import asyncio
import os
import warnings
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    AgentRetries,
    ModelRetry,
    NativeOutput,
    RunContext,
    Tool,
    UnexpectedModelBehavior,
    capture_run_messages,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.settings import ModelSettings

from .logging import get_logger
from .providers import Provider
from .usage import BUDGET, TRACKER

logger = get_logger(__name__)

# OpenAI reasoning models take no temperature; Pydantic AI drops it from the
# request (correctly) and warns on every call, which only fills the logs.
warnings.filterwarnings("ignore", message=r"Sampling parameters .* are not supported")

PROVIDERS: tuple[Provider, ...] = ("claude", "gemini", "openai")

Effort = Literal["low", "medium", "high"]

# A stage's rules about its own answer: the problems found, or [] if none.
Check = Callable[[Any], list[str]]

# Output ceiling assumed for the cost cap when a call sets none.
_DEFAULT_OUTPUT_ALLOWANCE = 8192

# HTTP statuses that say the request itself is wrong (bad input, context
# overflow, schema rejected). Another provider would get the same request,
# so these never fall back.
_REQUEST_ERROR_STATUSES = frozenset({400, 413, 422})


@dataclass(frozen=True)
class CallSettings:
    """How a stage calls its model, whatever the provider.

    `thinking` is the reasoning effort (adaptive thinking on Claude,
    reasoning effort on OpenAI, thinking level on Gemini); None leaves the
    model at its default. `temperature` is sent only where the model takes
    one: never to Claude (current models removed the sampling parameters),
    never to an OpenAI model that is reasoning.
    """

    max_tokens: int | None = None
    thinking: Effort | None = None
    temperature: float | None = None
    # Claude only: cache the (long, shared) system prompt across a fan-out.
    cache_instructions: bool = False
    # Transport retries on 429/5xx inside the SDK, with its backoff.
    http_retries: int = 2
    # Times a structured answer that fails validation goes back to the model.
    output_retries: int = 1


def reasoning_settings(
    effort: Effort | str, *, max_tokens: int = 16000, temperature: float = 1
) -> CallSettings:
    """A high-effort reasoning call (Ranker, Red team, Sizer, Rebalancer...)."""
    return CallSettings(
        max_tokens=max_tokens,
        thinking=_effort(effort),
        temperature=temperature,
    )


def deterministic_settings(
    *,
    max_tokens: int | None = None,
    cache_instructions: bool = False,
    http_retries: int = 2,
) -> CallSettings:
    """A plain low-temperature call (Analyst, Reviewer, readers...)."""
    return CallSettings(
        max_tokens=max_tokens,
        temperature=0,
        cache_instructions=cache_instructions,
        http_retries=http_retries,
    )


def _effort(effort: str) -> Effort:
    if effort in ("low", "medium", "high"):
        return effort
    raise ValueError(f"Unsupported reasoning effort {effort!r}; expected low, medium or high.")


def model_settings_for(provider: Provider, s: CallSettings) -> ModelSettings:
    """The request fields `s` becomes on `provider`.

    Pydantic AI maps `max_tokens` to each API's own field (OpenAI's
    `max_completion_tokens`, Gemini's `max_output_tokens`) and `thinking`
    to each one's reasoning knob (Claude: adaptive thinking at that effort).
    """
    out: dict[str, Any] = {}
    if s.max_tokens is not None:
        out["max_tokens"] = s.max_tokens
    elif provider == "claude":
        # Claude requires a ceiling; left unset, Pydantic AI asks for the
        # model's maximum (128k). Keep the 8k a plain answer has always had.
        out["max_tokens"] = _DEFAULT_OUTPUT_ALLOWANCE
    if s.thinking is not None:
        out["thinking"] = s.thinking
    sends_temperature = provider == "gemini" or (provider == "openai" and s.thinking is None)
    if s.temperature is not None and sends_temperature:
        out["temperature"] = s.temperature
    if s.cache_instructions and provider == "claude":
        out["anthropic_cache_instructions"] = True
    return ModelSettings(**out)


# --- models, one per call --------------------------------------------------

# An SDK client's connection pool belongs to the event loop that opened it,
# and the pipelines fan calls out over thread pools, each with its own loop.
# A client left to the garbage collector schedules its close on whatever loop
# is running at the time — another thread's — and fails with "Event loop is
# closed" (16 tracebacks in the 2026-10-03 filings run). So each call opens
# its client and closes it on the loop that used it.

_KEY_ENV = {
    "claude": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}


def _api_key(provider: Provider) -> str | None:
    """The environment's key, else the one `Settings` reads from `.env` —
    so a command that never loaded `.env` into the environment still works."""
    for name in _KEY_ENV[provider]:
        if key := os.environ.get(name):
            return key
    from .config import Settings

    s = Settings()
    return {
        "claude": s.anthropic_api_key,
        "openai": s.openai_api_key,
        "gemini": s.google_api_key,
    }[provider]


def _build_model(provider: Provider, model_id: str, http_retries: int) -> tuple[Model, Any]:
    """The model, and the SDK client this module opened for it (None when the
    provider opens and owns its own, which `async with model` closes)."""
    if provider == "claude":
        from anthropic import AsyncAnthropic
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        client = AsyncAnthropic(api_key=_api_key(provider), max_retries=http_retries)
        return AnthropicModel(model_id, provider=AnthropicProvider(anthropic_client=client)), client
    if provider == "openai":
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        client = AsyncOpenAI(api_key=_api_key(provider), max_retries=http_retries)
        return OpenAIChatModel(model_id, provider=OpenAIProvider(openai_client=client)), client
    if provider == "gemini":
        from google.genai.types import HttpRetryOptions
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(
            model_id,
            provider=GoogleProvider(
                api_key=_api_key(provider),
                retry_options=HttpRetryOptions(attempts=http_retries + 1),
            ),
        ), None
    raise ValueError(f"Unsupported provider {provider!r}. Expected one of {sorted(PROVIDERS)}.")


@asynccontextmanager
async def open_model(
    provider: Provider, model_id: str, http_retries: int = 2
) -> AsyncIterator[Model]:
    """A model client for one call, closed on the loop that used it."""
    model, client = _build_model(provider, model_id, http_retries)
    try:
        async with model:
            yield model
    finally:
        if client is not None:
            await client.close()


# --- errors ------------------------------------------------------------------


class OutputTruncatedError(RuntimeError):
    """The model stopped because it ran into its output ceiling.

    A cut-off JSON document is not a result, and treating it as one is how
    a lost plan used to read downstream as "hold everything". Carries the
    raw text so a caller that paid for it can still show it. Deliberately
    not a fallback error: another provider under the same ceiling would
    run out the same way.
    """

    def __init__(self, stage: str, max_output_tokens: int | None, raw_text: str) -> None:
        ceiling = f"{max_output_tokens:,}-token" if max_output_tokens else "provider's"
        super().__init__(f"{stage} hit its {ceiling} output ceiling before the answer was complete")
        self.stage = stage
        self.max_output_tokens = max_output_tokens
        self.raw_text = raw_text


class InvalidOutputError(RuntimeError):
    """A structured answer still failed validation after it was sent back to
    the model. Carries the last raw text for callers that keep it."""

    def __init__(self, stage: str, message: str, raw_text: str) -> None:
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.raw_text = raw_text


def is_fallback_error(e: BaseException) -> bool:
    """Worth retrying on another provider: an outage, a rate limit, a bad
    key or model id, a dropped connection. Not a request the API rejected
    as malformed (the fallback would reject it too)."""
    if isinstance(e, ModelHTTPError):
        return e.status_code not in _REQUEST_ERROR_STATUSES
    return isinstance(e, ModelAPIError)


# --- the agent -------------------------------------------------------------------


@dataclass(frozen=True)
class CallUsage:
    """Token counts in the ledger's terms: `input_tokens` uncached."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class RunResult:
    """What a call returned: the parsed model (structured) or text, and how
    it ended."""

    content: Any
    finish_reason: str | None
    usage: CallUsage
    provider: Provider
    model_id: str


def _ledger_usage(response: ModelResponse) -> SimpleNamespace:
    u = response.usage
    cache_read = u.cache_read_tokens or 0
    cache_write = u.cache_write_tokens or 0
    return SimpleNamespace(
        input_tokens=max(0, (u.input_tokens or 0) - cache_read - cache_write),
        output_tokens=u.output_tokens or 0,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


class _Guard(AbstractCapability[Any]):
    """Per response: record its tokens, and stop the run at the ceiling."""

    def __init__(self, agent: ModelAgent) -> None:
        self._agent = agent

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        a = self._agent
        TRACKER.record(a.name, a.model_id, _ledger_usage(response))
        if response.finish_reason == "length":
            logger.warning(
                "%s on %s/%s stopped at its output ceiling (%s tokens) — the answer was cut off",
                a.name,
                a.provider,
                a.model_id,
                a.settings.max_tokens or "provider default",
            )
            if a.structured:
                raise OutputTruncatedError(a.name, a.settings.max_tokens, response.text or "")
        return response


class ModelAgent:
    """One stage's model: instructions, an optional output schema, settings.

    `run(prompt)` returns a `RunResult` whose `content` is the validated
    schema instance (or text when no schema is set). Raises the provider's
    error (`ModelAPIError`), `OutputTruncatedError`, `InvalidOutputError`
    or `usage.BudgetExceededError`.
    """

    def __init__(
        self,
        name: str,
        provider: Provider,
        model: str,
        *,
        instructions: str = "",
        output_schema: type[BaseModel] | None = None,
        settings: CallSettings | None = None,
        tools: Sequence[Tool[Any] | Callable[..., Any]] = (),
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(
                f"Unsupported provider {provider!r}. Expected one of {sorted(PROVIDERS)}."
            )
        self.name = name
        self.provider: Provider = provider
        self.model_id = model
        self.settings = settings or CallSettings()
        self.structured = output_schema is not None
        self.instructions = instructions
        self._instruction_chars = len(instructions)
        self.agent: Agent[Check | None, Any] = Agent(
            output_type=NativeOutput(output_schema) if output_schema is not None else str,
            instructions=instructions or None,
            name=name,
            deps_type=Check | None,
            model_settings=model_settings_for(provider, self.settings),
            retries=AgentRetries(tools=1, output=self.settings.output_retries),
            tools=list(tools),
            capabilities=[_Guard(self)],
        )
        self.agent.output_validator(self._apply_check)

    def _apply_check(self, ctx: RunContext[Check | None], output: Any) -> Any:
        if ctx.deps is None or ctx.partial_output:
            return output
        problems = ctx.deps(output)
        if not problems:
            return output
        if ctx.retry < ctx.max_retries:
            logger.info("%s: asking the model to fix: %s", self.name, "; ".join(problems))
            raise ModelRetry(
                "Your answer breaks these rules. Fix them and answer again in full:\n- "
                + "\n- ".join(problems)
            )
        logger.warning(
            "%s: answer still breaks its rules after a retry (%s); keeping it for the "
            "caller's checks",
            self.name,
            "; ".join(problems),
        )
        return output

    @property
    def max_output_tokens(self) -> int:
        """The most output one call may produce, for the cost cap's estimate."""
        return self.settings.max_tokens or _DEFAULT_OUTPUT_ALLOWANCE

    def run(self, prompt: str, *, check: Check | None = None) -> RunResult:
        prompt_chars = self._instruction_chars + len(prompt)
        with (
            BUDGET.hold(self.name, self.model_id, prompt_chars, self.max_output_tokens),
            capture_run_messages() as messages,
        ):

            async def call() -> Any:
                async with open_model(
                    self.provider, self.model_id, self.settings.http_retries
                ) as model:
                    return await self.agent.run(prompt, model=model, deps=check)

            try:
                result = asyncio.run(call())
            except UnexpectedModelBehavior as e:
                raw = next(
                    (m.text or "" for m in reversed(messages) if isinstance(m, ModelResponse)),
                    "",
                )
                raise InvalidOutputError(self.name, str(e), raw) from e
        usage = result.usage
        cache_read, cache_write = usage.cache_read_tokens or 0, usage.cache_write_tokens or 0
        return RunResult(
            content=result.output,
            finish_reason=result.response.finish_reason,
            usage=CallUsage(
                input_tokens=max(0, (usage.input_tokens or 0) - cache_read - cache_write),
                output_tokens=usage.output_tokens or 0,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            ),
            provider=self.provider,
            model_id=self.model_id,
        )


# --- fallback ----------------------------------------------------------------------


def fallback_builder(
    fallback: tuple[Provider, str] | None,
    primary_provider: str,
    build: Callable[[Provider, str], ModelAgent],
) -> Callable[[], ModelAgent] | None:
    """The `build_fallback` that `run_with_fallback` takes: builds the
    fallback agent on demand, or None when there is no fallback or it is the
    primary's own provider (the same outage would fail it too)."""
    if not fallback or fallback[0] == primary_provider:
        return None
    provider, model = fallback
    return lambda: build(provider, model)


def run_with_fallback(
    primary: ModelAgent,
    build_fallback: Callable[[], ModelAgent] | None,
    prompt: str,
    *,
    check: Check | None = None,
) -> RunResult:
    """Run `primary`, retrying once on a fallback provider after an outage
    (`is_fallback_error`). Anything else (bad input, a cut-off or invalid
    answer, the cost cap) would fail the same way there, so it is raised."""
    try:
        return primary.run(prompt, check=check)
    except Exception as e:
        if build_fallback is None or not is_fallback_error(e):
            raise
        logger.warning(
            "%s call failed on %s/%s (%s) — retrying on fallback provider",
            primary.name,
            primary.provider,
            primary.model_id,
            e,
        )
        return build_fallback().run(prompt, check=check)
