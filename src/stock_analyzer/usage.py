"""Per-run LLM token accounting and cost estimate.

Every model call goes through `llm.AgnoAgent.run`, which records the run's
token metrics here under the agent's name (the pipeline stage). The
report shows the per-stage totals so the cost of a run is visible instead
of discovered on the provider bill.

Prices are Anthropic first-party list rates (USD per million tokens).
Models without a known price (Gemini, OpenAI) still report tokens; their
cost is shown as unknown rather than guessed.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
# Cache reads bill at ~0.1x the input rate, 5-minute cache writes at 1.25x.
_CACHE_READ_MULT = 0.1
_CACHE_WRITE_MULT = 1.25


def price_for(model: str) -> tuple[float, float] | None:
    return _PRICES_PER_MTOK.get(model)


@dataclass
class UsageRow:
    stage: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def cost_usd(self) -> float | None:
        price = price_for(self.model)
        if price is None:
            return None
        p_in, p_out = price
        return (
            self.input_tokens * p_in
            + self.output_tokens * p_out
            + self.cache_read_tokens * p_in * _CACHE_READ_MULT
            + self.cache_write_tokens * p_in * _CACHE_WRITE_MULT
        ) / 1_000_000


class UsageTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[tuple[str, str], UsageRow] = {}

    def record(self, stage: str, model: str, metrics: Any) -> None:
        if metrics is None:
            return
        with self._lock:
            row = self._rows.setdefault((stage, model), UsageRow(stage, model))
            row.calls += 1
            row.input_tokens += getattr(metrics, "input_tokens", 0) or 0
            row.output_tokens += getattr(metrics, "output_tokens", 0) or 0
            if model.startswith("gemini"):
                # Gemini reports thinking apart from output, and bills it as output.
                row.output_tokens += getattr(metrics, "reasoning_tokens", 0) or 0
            row.cache_read_tokens += getattr(metrics, "cache_read_tokens", 0) or 0
            row.cache_write_tokens += getattr(metrics, "cache_write_tokens", 0) or 0

    def rows(self) -> list[UsageRow]:
        with self._lock:
            rows = list(self._rows.values())
        return sorted(rows, key=lambda r: (r.cost_usd or 0.0, r.output_tokens), reverse=True)

    def total_cost(self) -> tuple[float, bool]:
        """(sum of known costs, whether every row had a known price)."""
        rows = self.rows()
        known = [r.cost_usd for r in rows if r.cost_usd is not None]
        return sum(known), len(known) == len(rows)

    def report_data(self) -> dict[str, Any]:
        total, complete = self.total_cost()
        return {
            "rows": [
                {
                    "stage": r.stage,
                    "model": r.model,
                    "calls": r.calls,
                    "input_tokens": r.input_tokens + r.cache_read_tokens + r.cache_write_tokens,
                    "output_tokens": r.output_tokens,
                    "cost_usd": r.cost_usd,
                }
                for r in self.rows()
            ],
            "total_cost_usd": total,
            "cost_complete": complete,
            "budget": BUDGET.report_data(),
        }

    def reset(self) -> None:
        with self._lock:
            self._rows.clear()


TRACKER = UsageTracker()


def log_usage_summary() -> None:
    from .logging import get_logger

    logger = get_logger(__name__)
    for r in TRACKER.rows():
        cost = f"${r.cost_usd:.3f}" if r.cost_usd is not None else "cost n/a"
        logger.info(
            "LLM usage %-12s %-22s calls=%-3d in=%-8d out=%-7d cache_read=%-8d %s",
            r.stage,
            r.model,
            r.calls,
            r.input_tokens,
            r.output_tokens,
            r.cache_read_tokens,
            cost,
        )
    total, complete = TRACKER.total_cost()
    logger.info(
        "LLM usage total: $%.2f%s", total, "" if complete else " (+ unpriced non-Claude calls)"
    )


# --- per-run cost cap ----------------------------------------------------------

# A token is ~4 characters of English; JSON payloads run denser, so divide
# by less to estimate on the high side.
_CHARS_PER_TOKEN = 3.5


def set_extra_prices(spec: str) -> None:
    """Add prices for models the table doesn't know, from LLM_PRICES:
    "gemini-pro-latest=1.25:10,gpt-6-astra=2:8" (USD per million input:output
    tokens). Unknown models are never guessed — they just don't count."""
    for part in (spec or "").split(","):
        name, _, rates = part.strip().partition("=")
        p_in, _, p_out = rates.partition(":")
        try:
            _PRICES_PER_MTOK[name.strip()] = (float(p_in), float(p_out))
        except ValueError:
            continue


def estimate_cost(model: str, input_chars: int, output_tokens: int) -> float | None:
    price = price_for(model)
    if price is None:
        return None
    p_in, p_out = price
    return (input_chars / _CHARS_PER_TOKEN * p_in + output_tokens * p_out) / 1_000_000


class BudgetExceededError(RuntimeError):
    """A model call was refused because it could push the run past its cap.
    Deliberately not a provider error, so it never triggers a fallback
    provider (which could spend unpriced money instead)."""


class Budget:
    """Per-run cap on estimated (priced) model spend.

    Every call reserves its worst-case cost before it runs and is refused if
    spent + in-flight + that estimate would pass the cap, so parallel calls
    cannot jointly overshoot. Stages with cheaper options (Analyst tiers,
    extra Ranker rounds, Reviewer model) plan against `remaining()` first
    and record what they cut in `notes`, which the report shows.
    """

    # Share of the cap kept back for the stages after the Ranker (red team,
    # sizer, rebalancer, pre-mortem) so the early fan-outs can't starve them.
    FINAL_RESERVE = 0.25

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cap: float | None = None
        self._pending = 0.0
        self.notes: list[str] = []

    def configure(self, cap: float | None) -> None:
        with self._lock:
            self.cap = cap if cap and cap > 0 else None
            self._pending = 0.0
            self.notes = []

    def spent(self) -> float:
        return TRACKER.total_cost()[0]

    def remaining(self) -> float | None:
        if self.cap is None:
            return None
        with self._lock:
            return self.cap - self.spent() - self._pending

    def available_for(self, share: float = 1.0, *, keep_reserve: bool = True) -> float | None:
        """What a planning stage may spend: `share` of what is left after the
        final-stage reserve. None when no cap is set."""
        cap, left = self.cap, self.remaining()
        if cap is None or left is None:
            return None
        reserve = self.FINAL_RESERVE * cap if keep_reserve else 0.0
        return max(0.0, left - reserve) * share

    def note(self, message: str) -> None:
        from .logging import get_logger

        get_logger(__name__).warning("Cost cap: %s", message)
        with self._lock:
            self.notes.append(message)

    @contextmanager
    def hold(self, stage: str, model: str, input_chars: int, output_tokens: int) -> Iterator[None]:
        cap = self.cap
        est = estimate_cost(model, input_chars, output_tokens) if cap else None
        if cap is None or est is None:
            yield
            return
        with self._lock:
            projected = TRACKER.total_cost()[0] + self._pending + est
            if projected > cap:
                refused = True
            else:
                refused = False
                self._pending += est
        if refused:
            self.note(
                f"refused a {stage} call on {model} (est. ${est:.2f}): it would take the "
                f"run past the ${self.cap:.2f} cap"
            )
            raise BudgetExceededError(f"{stage} call would exceed the ${self.cap:.2f} cost cap")
        try:
            yield
        finally:
            with self._lock:
                self._pending -= est

    def report_data(self) -> dict[str, Any] | None:
        if self.cap is None:
            return None
        return {"cap_usd": self.cap, "spent_usd": self.spent(), "notes": list(self.notes)}


BUDGET = Budget()
