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
from dataclasses import dataclass
from typing import Any

_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
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
