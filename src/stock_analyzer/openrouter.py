"""OpenRouter client for the open models that read, never decide.

Each call runs through Pydantic AI's OpenRouter model (the same layer as
llm.py), which speaks OpenRouter's extensions natively: host routing
(`provider.only` = the approved hosts), reasoning effort, and the billed
cost and serving host on every reply. Transport retries on 429/5xx are
the OpenAI SDK's. Every call:

  - is refused up front if the day's billed spend, plus calls in flight,
    plus this call's worst case would pass OPENROUTER_DAILY_CAP_USD. The
    day's spend is read from `openrouter_spend`, so the cap holds across
    processes (a cron job and a manual run on the same day share it);
  - records what OpenRouter actually billed (`usage.cost`) in that table,
    and in the run's usage tracker so the run log shows the total.

The cap error is `usage.BudgetExceededError`, which no caller treats as a
provider outage — a refused call must never fall back to a paid provider.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import text

from .db.session import exec_sql, get_session
from .logging import get_logger
from .usage import TRACKER, BudgetExceededError, estimate_cost

if TYPE_CHECKING:
    from .providers import Provider

logger = get_logger(__name__)

_BASE_URL = "https://openrouter.ai/api/v1"
# A slow model on a long filing can take minutes.
_TIMEOUT_S = 300.0
# Worst case for a model with no known price: well above anything the
# reader or checker costs, so an unpriced model can't slip past the cap.
_UNKNOWN_PRICE_PER_MTOK = (3.0, 15.0)
_CHARS_PER_TOKEN = 3.5
# The hosts each model may run on — an allowlist, so a host OpenRouter adds
# tomorrow gets no traffic until it has been checked. Vetted 2026-09-28 on
# ~1,900 stored filing reads (quote match 99.7-100% on every host listed)
# and a known-answer test. Left out of GLM-5.3: Sail Research (billed
# $0.0153 a read) and AkashML ($0.0246), against Io Net $0.0092, Morph
# $0.0116 and Baidu $0.0061 — same model, same fp8. `sort: price` alone
# was not strict enough to keep reads off them.
APPROVED_HOSTS: dict[str, list[str]] = {
    "z-ai/glm-5.3": ["io-net", "morph", "novita", "baidu"],
    "z-ai/glm-5.3-flash": [
        "sail-research",
        "gmicloud",
        "novita",
        "phala",
        "streamlake",
        "parasail",
        "z-ai",
        "morph",
    ],
}
# OpenRouter's slug → the provider name its replies carry.
HOST_NAMES: dict[str, str] = {
    "io-net": "Io Net",
    "morph": "Morph",
    "novita": "Novita",
    "baidu": "Baidu",
    "sail-research": "Sail Research",
    "gmicloud": "GMICloud",
    "phala": "Phala",
    "streamlake": "StreamLake",
    "parasail": "Parasail",
    "z-ai": "Z.AI",
    "akashml": "AkashML",
}


class NoApprovedHostError(RuntimeError):
    """Every approved host for the model is excluded (failed its check or
    its quality slipped) — the caller's fallback takes over."""


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    seconds: float
    provider: str | None = None  # the host OpenRouter routed the call to


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def worst_case_cost(model: str, input_chars: int, max_tokens: int) -> float:
    est = estimate_cost(model, input_chars, max_tokens)
    if est is not None:
        return est
    p_in, p_out = _UNKNOWN_PRICE_PER_MTOK
    return (input_chars / _CHARS_PER_TOKEN * p_in + max_tokens * p_out) / 1_000_000


def spent_today(db_path: str, day: str | None = None) -> float:
    with get_session(db_path) as session:
        row = exec_sql(
            session,
            text("SELECT COALESCE(SUM(cost_usd), 0) FROM openrouter_spend WHERE day = :d"),
            {"d": day or _today()},
        ).one()
    return float(row[0] or 0.0)


def _record(db_path: str, stage: str, c: Completion) -> None:
    with get_session(db_path) as session:
        exec_sql(
            session,
            text(
                "INSERT INTO openrouter_spend "
                "(day, model, stage, calls, input_tokens, output_tokens, cost_usd) "
                "VALUES (:d, :m, :s, 1, :i, :o, :c) "
                "ON CONFLICT(day, model, stage) DO UPDATE SET "
                "calls = calls + 1, input_tokens = input_tokens + :i, "
                "output_tokens = output_tokens + :o, cost_usd = cost_usd + :c"
            ),
            {
                "d": _today(),
                "m": c.model,
                "s": stage,
                "i": c.input_tokens,
                "o": c.output_tokens,
                "c": c.cost_usd,
            },
        )


class OpenRouter:
    def __init__(
        self,
        api_key: str,
        db_path: str,
        *,
        daily_cap_usd: float,
        excluded: dict[str, set[str]] | None = None,
    ) -> None:
        self.db_path = db_path
        self.daily_cap_usd = daily_cap_usd
        # {model: hosts skipped this run} (openrouter_hosts.excluded_hosts).
        self.excluded = excluded or {}
        self._lock = threading.Lock()
        self._pending = 0.0
        self._api_key = api_key
        self._local = threading.local()
        # Tests route requests to a fake server: a factory of httpx2 clients.
        self.http_client_factory: Any = None

    def _model(self, model: str) -> Any:
        """This thread's Pydantic AI model for `model` (an SDK client's pool
        belongs to the event loop of the thread that opened it)."""
        cache: dict[str, Any] | None = getattr(self._local, "models", None)
        if cache is None:
            cache = self._local.models = {}
        if model not in cache:
            from openai import AsyncOpenAI
            from pydantic_ai.models.openrouter import OpenRouterModel
            from pydantic_ai.providers.openrouter import OpenRouterProvider

            client = AsyncOpenAI(
                base_url=_BASE_URL,
                api_key=self._api_key,
                max_retries=2,
                timeout=_TIMEOUT_S,
                # OpenRouter's attribution header; harmless if ignored.
                default_headers={"X-Title": "stock-analyzer"},
                http_client=self.http_client_factory() if self.http_client_factory else None,
            )
            cache[model] = OpenRouterModel(model, provider=OpenRouterProvider(openai_client=client))
        return cache[model]

    def allowed_hosts(self, model: str) -> list[str]:
        skip = self.excluded.get(model, set())
        return [h for h in APPROVED_HOSTS.get(model, []) if h not in skip]

    def _reserve(self, stage: str, model: str, est: float) -> None:
        with self._lock:
            projected = spent_today(self.db_path) + self._pending + est
            if projected > self.daily_cap_usd:
                raise BudgetExceededError(
                    f"{stage} call on {model} (est. ${est:.3f}) would pass the "
                    f"${self.daily_cap_usd:.2f}/day OpenRouter cap"
                )
            self._pending += est

    def _release(self, est: float) -> None:
        with self._lock:
            self._pending -= est

    def complete(
        self,
        stage: str,
        model: str,
        system: str,
        user: str,
        *,
        max_tokens: int = 4000,
        json_mode: bool = True,
        temperature: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> Completion:
        est = worst_case_cost(model, len(system) + len(user), max_tokens)
        self._reserve(stage, model, est)
        try:
            settings = self._settings(model, max_tokens, json_mode, temperature, extra or {})
            t0 = time.monotonic()
            response = self._call(model, system, user, settings)
            usage = response.usage
            inp = int(usage.input_tokens or 0)
            out = int(usage.output_tokens or 0)
            details = response.provider_details or {}
            cost = details.get("cost")
            if cost is None:
                # No billed figure: price the tokens, or charge the estimate.
                cost = estimate_cost(model, int(inp * _CHARS_PER_TOKEN), out) or est
            c = Completion(
                text=response.text or "",
                model=model,
                input_tokens=inp,
                output_tokens=out,
                cost_usd=float(cost),
                seconds=round(time.monotonic() - t0, 1),
                provider=details.get("downstream_provider"),
            )
        finally:
            self._release(est)
        _record(self.db_path, stage, c)
        TRACKER.record(
            stage,
            model,
            SimpleNamespace(input_tokens=c.input_tokens, output_tokens=c.output_tokens),
        )
        return c

    def _settings(
        self,
        model: str,
        max_tokens: int,
        json_mode: bool,
        temperature: float,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """The request as Pydantic AI settings: `reasoning` and `provider`
        (host routing, restricted to the approved hosts) are OpenRouter
        settings; anything else in `extra` goes into the body as given."""
        rest = dict(extra)
        settings: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "openrouter_usage": {"include": True},
        }
        if "reasoning" in rest:
            settings["openrouter_reasoning"] = rest.pop("reasoning")
        routing = rest.pop("provider", None)
        if isinstance(routing, dict):
            if model in APPROVED_HOSTS and "only" not in routing:
                allowed = self.allowed_hosts(model)
                if not allowed:
                    raise NoApprovedHostError(f"every approved host for {model} is excluded")
                routing = {**routing, "only": allowed}
            settings["openrouter_provider"] = routing
        if json_mode:
            rest["response_format"] = {"type": "json_object"}
        if rest:
            settings["extra_body"] = rest
        return settings

    def _call(self, model: str, system: str, user: str, settings: dict[str, Any]) -> Any:
        """The model's response. An empty or cut-off answer comes back as it
        is (empty text) rather than as an error, and is never re-asked here:
        the reader decides whether to retry, and how (filing_reader)."""
        from pydantic_ai import Agent, UnexpectedModelBehavior, capture_run_messages
        from pydantic_ai.messages import ModelResponse

        agent: Agent[None, str] = Agent(
            output_type=str, instructions=system, model_settings=cast("Any", settings), retries=0
        )
        with capture_run_messages() as messages:
            try:
                return agent.run_sync(user, model=self._model(model)).response
            except UnexpectedModelBehavior:
                last = next((m for m in reversed(messages) if isinstance(m, ModelResponse)), None)
                if last is None:
                    raise
                return last


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_object(raw: str) -> dict[str, Any] | None:
    """The first JSON object in a reply, tolerating a markdown fence or a
    sentence around it (open models add both, even in JSON mode)."""
    raw = _FENCE.sub("", raw or "").strip()
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def client_from_settings(settings: Any) -> OpenRouter | None:
    """A client that skips the hosts whose latest known-answer check failed
    or whose read quality slipped (openrouter_hosts.excluded_hosts)."""
    if not settings.openrouter_api_key:
        return None
    from .openrouter_hosts import excluded_hosts

    return OpenRouter(
        settings.openrouter_api_key,
        settings.discover_db_path,
        daily_cap_usd=settings.openrouter_daily_cap_usd,
        excluded=excluded_hosts(settings.discover_db_path, today=datetime.now(UTC).date()),
    )


# --- helper roles on an open model ---------------------------------------------

# Low-effort thinking on fp8-or-better hosts, cheapest first — the routing
# the filing reader was validated on (agents/filing_reader.READER_EXTRA).
HELPER_EXTRA: dict[str, Any] = {
    "reasoning": {"effort": "low"},
    "provider": {
        "quantizations": ["fp8", "bf16", "fp16"],
        "sort": "price",
        "allow_fallbacks": True,
    },
}
# Thinking comes out of the same budget as the answer.
HELPER_MAX_TOKENS = 8000


class OpenRouterAgent:
    """Stands in for `llm.ModelAgent` in a helper role: `.run(prompt)`
    returns an object with `.content`. Billed and capped like every other
    OpenRouter call. `fallback` builds the agent to use when OpenRouter
    fails, answers empty, or fails `validate` (prompt, reply → problems) —
    never when the daily cap refuses the call, which must not turn into
    paid spend elsewhere."""

    def __init__(
        self,
        name: str,
        model: str,
        instructions: str,
        *,
        client: OpenRouter | None,
        json_mode: bool = False,
        fallback: Any = None,
        validate: Any = None,
    ) -> None:
        self.name, self.model_id, self.instructions = name, model, instructions
        self.client, self.json_mode, self._fallback = client, json_mode, fallback
        self._validate = validate

    def run(self, prompt: str) -> Any:
        try:
            if self.client is None:
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            c = self.client.complete(
                self.name,
                self.model_id,
                self.instructions,
                prompt,
                max_tokens=HELPER_MAX_TOKENS,
                json_mode=self.json_mode,
                extra=HELPER_EXTRA,
            )
            if not c.text.strip():
                raise RuntimeError(f"empty reply from {self.model_id}")
            problems = self._validate(prompt, c.text) if self._validate else []
            if problems:
                raise RuntimeError(f"reply failed its check: {'; '.join(problems)[:200]}")
            return SimpleNamespace(content=c.text)
        except BudgetExceededError:
            raise
        except Exception as e:
            if self._fallback is None:
                raise
            logger.warning("%s on %s failed (%s) — using the fallback", self.name, self.model_id, e)
            return self._fallback().run(prompt)


def helper_agent(
    name: str,
    provider: str,
    model: str,
    instructions: str,
    *,
    json_mode: bool = False,
    fallback: tuple[str, str] | None = None,
    validate: Any = None,
) -> Any:
    """An agent for a helper role: `llm.ModelAgent` for claude/gemini/openai,
    `OpenRouterAgent` for "openrouter" (with `fallback` = (provider, model)
    to use if OpenRouter fails)."""
    from .llm import ModelAgent

    if provider != "openrouter":
        return ModelAgent(name, cast("Provider", provider), model, instructions=instructions)
    from .config import Settings

    back = None
    if fallback is not None:
        fb_provider, fb_model = fallback

        def back() -> Any:
            return ModelAgent(
                name, cast("Provider", fb_provider), fb_model, instructions=instructions
            )

    return OpenRouterAgent(
        name,
        model,
        instructions,
        client=client_from_settings(Settings()),
        json_mode=json_mode,
        fallback=back,
        validate=validate,
    )
