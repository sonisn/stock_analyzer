"""Open-market insider trades (Form 4) on the names you hold or watch.

Filed data from Finnhub rather than news coverage: it does not depend on a
search quota, and a filing is a fact where a snippet is someone's summary.
Only open-market purchases (code P) and sales (code S) are kept — grants,
option exercises, tax withholding and gifts say nothing about what an
insider thinks the stock is worth. No LLM.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from ..logging import get_logger
from . import finnhub as finnhub_data

logger = get_logger(__name__)

_SIDES = {"P": "BUY", "S": "SELL"}
# Holdings also carry option symbols and dead CUSIPs; Form 4 lookups only
# make sense for listed common-stock tickers.
_TICKER = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")


def fetch_form4_trades(
    tickers: list[str] | set[str], *, days: int, today: date | None = None
) -> list[dict[str, Any]] | None:
    """Open-market buys and sells filed in the last `days`, newest first.

    None when Finnhub is not configured, so the caller can tell "no
    trades" from "could not look".
    """
    client = finnhub_data._client()
    if client is None:
        logger.warning("FINNHUB_API_KEY not set; Form 4 lookup skipped")
        return None
    today = today or date.today()
    start = (today - timedelta(days=days)).isoformat()
    out: list[dict[str, Any]] = []
    for ticker in sorted({t for t in tickers if _TICKER.match(t)}):
        activity = finnhub_data.fetch_insider_activity(client, ticker, days=days)
        for tx in (activity or {}).get("recent_transactions") or []:
            side = _SIDES.get(str(tx.get("code") or ""))
            if side is None or str(tx.get("date") or "") < start:
                continue
            shares = abs(float(tx.get("shares") or 0))
            price = float(tx.get("price") or 0)
            out.append(
                {
                    "ticker": ticker,
                    "date": tx.get("date"),
                    "name": str(tx.get("name") or "unnamed insider").title(),
                    "side": side,
                    "shares": shares,
                    "price": price,
                    "value_usd": shares * price,
                }
            )
    out.sort(key=lambda t: (t["date"] or "", t["value_usd"]), reverse=True)
    logger.info("Form 4: %d open-market trade(s) across %d ticker(s)", len(out), len(tickers))
    return out


def form4_section(trades: list[dict[str, Any]], *, days: int) -> str:
    """The trades as a report section (`Heading:` then `- ` bullets)."""
    lines = [f"Form 4 Filings on Your Holdings & Watchlist (last {days} days):"]
    if not trades:
        lines.append("- No open-market insider buys or sells filed.")
        return "\n".join(lines)
    for t in trades:
        value = f"${t['value_usd']:,.0f}" if t["value_usd"] else "value not stated"
        price = f" @ ${t['price']:,.2f}" if t["price"] else ""
        lines.append(
            f"- {t['ticker']}: {t['name']} {t['side']} {t['shares']:,.0f} sh{price} "
            f"({value}) — {t['date']}"
        )
    return "\n".join(lines)


__all__ = ["fetch_form4_trades", "form4_section"]
