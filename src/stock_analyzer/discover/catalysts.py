"""Forward-catalyst validation and prompt formatting.

The Analyst/Reviewer extract `upcoming_catalysts` from dated news
(`data/ticker_news.py`) plus filings/transcripts. This module is the
deterministic post-LLM check — same compute-then-force-correct shape as
cc_validation.py::validate_option_writes: drop catalysts that cite a
source the model was never given, drop ones dated in the past (they are
history, not upcoming), and null out unparseable dates rather than trust
them.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel

from ..logging import get_logger
from ..models.llm import Catalyst

logger = get_logger(__name__)

_FIXED_SOURCES = frozenset({"quarterly_mda", "earnings_transcript", "earnings_calendar"})


def _parse_date(value: str) -> date | None:
    text = value.strip()
    for candidate in (text, f"{text}-01"):
        try:
            return date.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def validate_catalysts(
    catalysts: list[Catalyst],
    *,
    news_ids: set[str],
    today: date,
    ticker: str = "",
) -> tuple[list[Catalyst], list[str]]:
    kept: list[Catalyst] = []
    warnings: list[str] = []
    for c in catalysts:
        source = c.source.strip()
        news_id = source.removeprefix("news:")
        if source not in _FIXED_SOURCES and news_id not in news_ids:
            warnings.append(f"{ticker}: dropped catalyst citing unknown source {source!r}")
            continue
        if c.expected_date:
            parsed = _parse_date(c.expected_date)
            if parsed is None:
                c = c.model_copy(update={"expected_date": None})
            elif parsed < today:
                warnings.append(
                    f"{ticker}: dropped catalyst dated in the past ({c.expected_date}): {c.event}"
                )
                continue
            else:
                c = c.model_copy(update={"expected_date": parsed.isoformat()})
        kept.append(c)
    for w in warnings:
        logger.warning("Catalyst validation: %s", w)
    return kept, warnings


def repair_catalysts[R: BaseModel](
    reports: dict[str, R],
    recent_news: dict[str, list[dict[str, Any]]],
    *,
    today: date | None = None,
) -> tuple[dict[str, R], list[str]]:
    """Apply `validate_catalysts` to every AnalystReport/HoldingReview."""
    today = today or date.today()
    out: dict[str, R] = {}
    warnings: list[str] = []
    for ticker, report in reports.items():
        catalysts = getattr(report, "upcoming_catalysts", None)
        if not catalysts:
            out[ticker] = report
            continue
        news_ids = {item["id"] for item in recent_news.get(ticker, []) if item.get("id")}
        kept, w = validate_catalysts(catalysts, news_ids=news_ids, today=today, ticker=ticker)
        warnings.extend(w)
        out[ticker] = (
            report if kept == catalysts else report.model_copy(update={"upcoming_catalysts": kept})
        )
    return out, warnings


def _sort_key(c: Catalyst) -> tuple[int, str]:
    return (0, c.expected_date) if c.expected_date else (1, "")


def format_catalyst_lines(catalysts: list[Catalyst]) -> list[str]:
    return [
        f"{c.expected_date or 'date n/a'} | {c.direction} | {c.impact} impact | {c.event}"
        for c in sorted(catalysts, key=_sort_key)
    ]


def format_catalyst_block(catalysts: list[Catalyst]) -> str:
    """Deterministic prompt block appended under a candidate's analysis."""
    if not catalysts:
        return "Upcoming catalysts: none identified in recent news or filings."
    return "Upcoming catalysts (validated, soonest first):\n" + "\n".join(
        f"  - {line}" for line in format_catalyst_lines(catalysts)
    )


def catalysts_to_dicts(catalysts: list[Catalyst]) -> list[dict[str, Any]]:
    """Report-layer shape, soonest first."""
    return [c.model_dump() for c in sorted(catalysts, key=_sort_key)]
