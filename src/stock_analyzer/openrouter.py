"""OpenRouter client for the open models that read, never decide.

A plain chat-completions call over the shared `HttpClient` (retries on
429/5xx): agno would add nothing here but a dependency on its OpenAI class
guessing at OpenRouter's extensions. Every call:

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
from typing import Any

from sqlalchemy import text

from .db.session import exec_sql, get_session
from .http_client import HttpClient, RetryPolicy
from .logging import get_logger
from .usage import TRACKER, BudgetExceededError, estimate_cost

logger = get_logger(__name__)

_URL = "https://openrouter.ai/api/v1/chat/completions"
# Worst case for a model with no known price: well above anything the
# reader or checker costs, so an unpriced model can't slip past the cap.
_UNKNOWN_PRICE_PER_MTOK = (3.0, 15.0)
_CHARS_PER_TOKEN = 3.5


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
    def __init__(self, api_key: str, db_path: str, *, daily_cap_usd: float) -> None:
        self.db_path = db_path
        self.daily_cap_usd = daily_cap_usd
        self._lock = threading.Lock()
        self._pending = 0.0
        self._http = HttpClient(
            default_headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # OpenRouter's attribution headers; harmless if ignored.
                "X-Title": "stock-analyzer",
            },
            # A slow model on a long filing can take minutes.
            timeout=300.0,
            retry_policy=RetryPolicy(max_attempts=3),
            name="openrouter",
        )

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
            body: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "usage": {"include": True},
                **(extra or {}),
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            t0 = time.monotonic()
            data = self._http.post_json(_URL, json=body)
            if "error" in data:
                raise RuntimeError(f"OpenRouter error on {model}: {data['error']}")
            usage = data.get("usage") or {}
            inp = int(usage.get("prompt_tokens") or 0)
            out = int(usage.get("completion_tokens") or 0)
            cost = usage.get("cost")
            if cost is None:
                # No billed figure: price the tokens, or charge the estimate.
                cost = estimate_cost(model, int(inp * _CHARS_PER_TOKEN), out) or est
            message = (data.get("choices") or [{}])[0].get("message") or {}
            c = Completion(
                text=message.get("content") or "",
                model=model,
                input_tokens=inp,
                output_tokens=out,
                cost_usd=float(cost),
                seconds=round(time.monotonic() - t0, 1),
                provider=data.get("provider"),
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
    if not settings.openrouter_api_key:
        return None
    return OpenRouter(
        settings.openrouter_api_key,
        settings.discover_db_path,
        daily_cap_usd=settings.openrouter_daily_cap_usd,
    )
