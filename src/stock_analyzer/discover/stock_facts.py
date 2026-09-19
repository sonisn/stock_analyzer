"""What a holding actually did, for the mornings its headlines are filler.

Measured across four holdings' Yahoo feeds (2026-09-19), 35 of 40 items
were syndicated commentary — and after `data/news_rank.py` drops the ones
that aren't about the company, a holding can legitimately have nothing to
show. NVDA had zero relevant headlines that day and, in the same window,
an 8-K and a 10-Q.

So when the news section is thin this fills it with facts that are about
the company by construction, from sources that are free and have no
quota: SEC filings (EDGAR), analyst estimate revisions (yfinance) and
Form 4 insider transactions (Finnhub). No LLM.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

from ..logging import get_logger

logger = get_logger(__name__)

# How few ranked headlines counts as "nothing to read this morning".
THIN_NEWS = 2
_MAX_WORKERS = 4
_FILINGS_SHOWN = 2

_FORM_MEANING = {
    "8-K": "material event",
    "10-Q": "quarterly report",
    "10-K": "annual report",
}


def _day(value: str | None) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except TypeError, ValueError:
        return None


def filings_line(filings: list[dict[str, Any]]) -> str | None:
    """The newest filings, with what each form means and its link."""
    parts = []
    for f in filings[:_FILINGS_SHOWN]:
        filed = _day(f.get("filed_on"))
        if filed is None:
            continue
        meaning = _FORM_MEANING.get(f["form"], "filing")
        parts.append(f"{f['form']} filed {filed:%b %d} — {meaning} ({f['url']})")
    return "; ".join(parts) or None


def estimates_line(revisions: dict[str, Any] | None) -> str | None:
    """Where analysts have moved next year's number — the long-term signal,
    the same one the post-earnings check reads."""
    if not revisions:
        return None
    up = revisions.get("next_year_up_30d") or 0
    down = revisions.get("next_year_down_30d") or 0
    if not (up or down):
        return None
    net = up - down
    direction = "raising" if net > 0 else "cutting" if net < 0 else "holding"
    return f"Next-year EPS: {up} up / {down} down in 30 days — analysts {direction}"


def insider_line(activity: dict[str, Any] | None) -> str | None:
    """Form 4 activity, only when there was some."""
    if not activity:
        return None
    buys, sells = activity.get("n_buys") or 0, activity.get("n_sells") or 0
    if not (buys or sells):
        return None
    money = []
    if activity.get("buy_value_usd"):
        money.append(f"${activity['buy_value_usd'] / 1e6:.1f}M bought")
    if activity.get("sell_value_usd"):
        money.append(f"${activity['sell_value_usd'] / 1e6:.1f}M sold")
    tail = f" ({', '.join(money)})" if money else ""
    return f"Form 4s, last 90 days: {buys} buy(s), {sells} sale(s){tail}"


def company_facts(
    *,
    filings: list[dict[str, Any]] | None = None,
    revisions: dict[str, Any] | None = None,
    insider: dict[str, Any] | None = None,
) -> dict[str, str]:
    """{block label: text} for whichever facts exist — labels start with a
    letter so the email's field parser gives each its own row."""
    lines = {
        "Filings": filings_line(filings or []),
        "Estimates": estimates_line(revisions),
        "Insiders": insider_line(insider),
    }
    return {label: text for label, text in lines.items() if text}


def fetch_company_facts(
    tickers: list[str],
    *,
    today: date | None = None,
    days: int = 45,
    revisions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, str]]:
    """Facts for the holdings whose news came up thin. One EDGAR fetch and
    one Finnhub call per ticker, both free; `revisions` is passed in when
    the caller already batched it. Any source failing just means one
    fewer line."""
    if not tickers:
        return {}
    from ..data import finnhub as finnhub_data
    from ..data.eps_revisions import batch_eps_revisions
    from ..data.sec_edgar import fetch_recent_filings

    today = today or date.today()
    if revisions is None:
        revisions = _guarded("estimate revisions", lambda: batch_eps_revisions(tickers)) or {}
    client = finnhub_data._client()

    def facts_for(ticker: str) -> tuple[str, dict[str, str]]:
        filings = (
            _guarded(
                f"SEC filings for {ticker}",
                lambda: fetch_recent_filings(ticker, days=days, today=today),
            )
            or []
        )
        insider = (
            _guarded(
                f"insider activity for {ticker}",
                lambda: finnhub_data.fetch_insider_activity(client, ticker),
            )
            if client is not None
            else None
        )
        return ticker, company_facts(
            filings=filings, revisions=revisions.get(ticker), insider=insider
        )

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        return {t: facts for t, facts in ex.map(facts_for, tickers) if facts}


def _guarded(what: str, fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — a missing fact is not an error
        logger.warning("Could not fetch %s (%s)", what, e)
        return None


__all__ = [
    "THIN_NEWS",
    "company_facts",
    "estimates_line",
    "fetch_company_facts",
    "filings_line",
    "insider_line",
]
