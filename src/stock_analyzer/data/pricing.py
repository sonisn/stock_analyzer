"""One price per ticker, whoever is quoting it.

Each brokerage reports its own price for a position, and those prices can
be stale: on 2026-09-19 the HSA showed BE at $298.61 while the live quote
and the other two accounts said $265.63 — a $2,408 overstatement on 73
shares. Valuing positions at whatever each account happened to say meant
the same holding was worth two different amounts in one email, sector
weights and add-on sizing moved with a stale feed, and the number went
into `portfolio_snapshots`, where it permanently skews the time-weighted
return against SPY.

So valuation takes the live quote the run already fetched, falls back to
the broker's price only when there is no quote, and says out loud when an
account's price disagrees with the market by more than a couple of
percent — that disagreement is usually the first sign a feed is stale.
"""

from __future__ import annotations

from typing import Any

from ..logging import get_logger

logger = get_logger(__name__)

# How far an account's price may sit from the live quote before it is
# called out. Brokers round and lag by seconds; 2% is not that.
PRICE_TOLERANCE_PCT = 2.0


def quotes_from_ticker_data(ticker_data: dict[str, dict[str, Any]]) -> dict[str, float]:
    """{ticker: live price} from what the daily email already fetched."""
    out = {}
    for ticker, data in (ticker_data or {}).items():
        price = (data or {}).get("price_value")
        if price:
            out[str(ticker).upper()] = float(price)
    return out


def reconcile_prices(
    holdings: dict[str, list[dict[str, Any]]],
    quotes: dict[str, float] | None = None,
    *,
    tolerance_pct: float = PRICE_TOLERANCE_PCT,
) -> tuple[dict[str, float], list[str]]:
    """Return ({ticker: price to value at}, notes about disagreeing feeds).

    The live quote wins wherever there is one. Without it the broker
    prices are averaged per ticker, which at least keeps one holding from
    being worth two amounts in the same report.
    """
    quotes = {k.upper(): v for k, v in (quotes or {}).items() if v}
    broker: dict[str, list[tuple[str, float]]] = {}
    for account, items in (holdings or {}).items():
        for h in items:
            ticker = str(h.get("ticker") or "").upper()
            price = float(h.get("price") or 0)
            if ticker and price > 0:
                broker.setdefault(ticker, []).append((account, price))

    prices: dict[str, float] = {}
    notes: list[str] = []
    for ticker, quoted in broker.items():
        live = quotes.get(ticker)
        if live:
            prices[ticker] = live
            for account, price in quoted:
                off = (price / live - 1) * 100
                if abs(off) >= tolerance_pct:
                    note = f"{ticker} priced {off:+.0f}% off the market in {account}"
                    notes.append(note)
                    logger.warning(
                        "%s ($%.2f there vs $%.2f live) — valuing it at the live quote",
                        note,
                        price,
                        live,
                    )
        else:
            prices[ticker] = sum(p for _, p in quoted) / len(quoted)
    for ticker, live in quotes.items():
        prices.setdefault(ticker, live)
    return prices, notes


__all__ = ["PRICE_TOLERANCE_PCT", "quotes_from_ticker_data", "reconcile_prices"]
