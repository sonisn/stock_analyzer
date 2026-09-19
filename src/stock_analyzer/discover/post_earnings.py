"""After a holding reports: did the long-term case change? No LLM.

The daily email tells the reader to "check the results against the
long-term thesis" when earnings come up; this does that check the day
after. For each holding whose earnings date passed in the last
RESULT_DAYS days it reports the EPS result (when Yahoo has it) and — the
part that matters for a 3-5 year hold — which way analysts moved their
estimates since: falling next-year estimates flag the thesis for review,
steady or rising ones confirm it.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from ..agents.stock_views import _day, last_earnings_event

RESULT_DAYS = 7


def recent_results(
    ticker_data: dict[str, dict[str, Any]], *, today: date, days: int = RESULT_DAYS
) -> list[dict[str, Any]]:
    """Holdings that reported within `days`, with the EPS line when known."""
    out = []
    for ticker, data in ticker_data.items():
        event = last_earnings_event(data, today)
        if event is None or (today - event).days > days:
            continue
        row = next(
            (
                r
                for r in (data.get("earnings") or {}).get("history") or []
                if _day(r.get("Earnings Date")) == event
            ),
            {},
        )
        out.append(
            {
                "ticker": ticker,
                "reported_on": event,
                "eps": row.get("Reported EPS"),
                "estimate": row.get("EPS Estimate"),
                "surprise_pct": row.get("Surprise(%)"),
            }
        )
    return sorted(out, key=lambda r: r["reported_on"], reverse=True)


def with_revisions(
    results: list[dict[str, Any]], revisions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach the estimate direction since the report. Next-year estimates
    carry the long-term signal; the last 7 days cover the post-report move."""
    out = []
    for r in results:
        rev = revisions.get(r["ticker"]) or {}
        next_year = (rev.get("next_year_up_30d") or 0) - (rev.get("next_year_down_30d") or 0)
        week = rev.get("net_revisions_7d") or 0
        if not rev:
            direction = "unknown"
        elif next_year < 0 or week < 0:
            direction = "lowering"
        elif next_year > 0 or week > 0:
            direction = "raising"
        else:
            direction = "steady"
        out.append({**r, "direction": direction, "next_year_net": next_year, "week_net": week})
    return out


def result_text(r: dict[str, Any]) -> str:
    eps = ""
    if r.get("eps") is not None and r.get("estimate") is not None:
        sur = f", {float(r['surprise_pct']):+.1f}%" if r.get("surprise_pct") is not None else ""
        eps = f" (EPS {r['eps']} vs {r['estimate']} est{sur})"
    head = f"{r['ticker']} reported {r['reported_on']:%b %d}{eps}"
    if r["direction"] == "lowering":
        return (
            f"{head}; analysts have cut estimates since (next year {r['next_year_net']:+d}, "
            f"last 7 days {r['week_net']:+d}) — re-check the long-term thesis."
        )
    if r["direction"] == "unknown":
        return f"{head}; estimate revisions unavailable — skim the results against the thesis."
    return f"{head}; estimates {r['direction']} since — the long-term case holds."
