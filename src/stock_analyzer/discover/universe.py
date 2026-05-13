"""Build the candidate universe for discovery.

Sources combined:
  - Tickers mentioned in recent insider buying coverage
  - Tickers mentioned in billionaire investor holdings/coverage
  - Tickers in watchlist (DISCOVER_WATCHLIST env var)

Tickers are extracted from coverage article text via regex + a blacklist of
common false positives. Counts feed a coarse `conviction` integer used as a
tiebreaker in the screen — billionaire mentions weighted 2x, watchlist 5x.
S&P 500 enumeration is intentionally NOT in v1: the screen's RS_6mo>0 +
price>200DMA filters already select for index-leading names.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..data.hedge_funds import fetch_hedge_fund_trades
from ..data.insider import fetch_insider_trades
from ..data.sec_edgar import load_ticker_cik_map
from ..logging import get_logger

logger = get_logger(__name__)

# Common acronyms / English words that match the bare [A-Z]{2,5} pattern.
# Adding to this list is the right move when you see noise in the universe.
_FALSE_POSITIVES = frozenset({
    "AN", "AND", "ALL", "AM", "AS", "AT", "BE", "BY", "DO", "FOR", "FROM",
    "GO", "HAS", "HE", "I", "IF", "IN", "IS", "IT", "ITS", "MY", "NEW", "NO",
    "NOT", "OF", "OK", "ON", "OR", "SO", "TO", "UP", "US", "WE", "YOU", "THE",
    "USA", "USD", "EUR", "GBP", "JPY", "CNY",
    "CEO", "CFO", "COO", "CTO", "CMO", "CIO",
    "FED", "FOMC", "GDP", "CPI", "PPI", "PMI", "FDA", "SEC", "IRS", "DOJ",
    "FTC", "DOE", "EPA", "DOD", "NSA", "CIA", "FBI", "NYSE", "AMEX", "OTC",
    "ETF", "IPO", "FYI", "AI", "ML", "AR", "VR", "EV", "OS",
    "PR", "PE", "EPS", "ROE", "ROI", "ROA", "FY", "Q1", "Q2", "Q3", "Q4",
    "YOY", "QOQ", "YTD", "MTD", "AGM", "PIE", "PT", "ST", "MT", "LT",
    "UK", "EU", "ASEAN", "G7", "G20",
})

_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
_EXCHANGE_RE = re.compile(r"(?:NYSE|NASDAQ|NYSEARCA|AMEX)\s*:\s*([A-Z]{1,5})\b")
# Bare uppercase tokens 2-5 chars — strict 5 chars to avoid section refs like 10-K.
_BARE_RE = re.compile(r"\b([A-Z]{2,5})\b")


def _extract_tickers(text: str) -> set[str]:
    if not text:
        return set()
    found: set[str] = set()
    for m in _CASHTAG_RE.finditer(text):
        found.add(m.group(1))
    for m in _EXCHANGE_RE.finditer(text):
        found.add(m.group(1))
    for m in _BARE_RE.finditer(text):
        sym = m.group(1)
        if sym not in _FALSE_POSITIVES:
            found.add(sym)
    return found - _FALSE_POSITIVES


def _tickers_from_items(items: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in items:
        text = " ".join((item.get("title") or "", item.get("snippet") or ""))
        for t in _extract_tickers(text):
            counter[t] += 1
    return counter


def build_universe(watchlist: tuple[str, ...] = ()) -> dict[str, dict[str, Any]]:
    """Return {ticker: {sources, conviction}}.

    sources: list[str] — which feeds the ticker appeared in
    conviction: int  — weighted sum of mentions across sources
    """
    insider_items = fetch_insider_trades(days=30, max_results=40)
    hedge_items = fetch_hedge_fund_trades(days=30, max_results=40)

    insider_counts = _tickers_from_items(insider_items)
    hedge_counts = _tickers_from_items(hedge_items)

    universe: dict[str, dict[str, Any]] = {}
    for ticker, n in insider_counts.items():
        u = universe.setdefault(ticker, {"sources": [], "conviction": 0})
        u["sources"].append("insider")
        u["conviction"] += n
    for ticker, n in hedge_counts.items():
        u = universe.setdefault(ticker, {"sources": [], "conviction": 0})
        u["sources"].append("billionaire")
        u["conviction"] += n * 2
    for ticker in watchlist:
        u = universe.setdefault(ticker, {"sources": [], "conviction": 0})
        if "watchlist" not in u["sources"]:
            u["sources"].append("watchlist")
            u["conviction"] += 5

    # Validate against the SEC's authoritative ticker→CIK map. The regex-based
    # extraction catches a lot of English words (HOME, TABLE, OFF, LP, LLC, etc.)
    # that aren't real listings; this drops them before they hit yfinance.
    # Tickers with share classes appear in SEC data with a dash (e.g. BRK-B);
    # accept both BRK.B and BRK-B forms.
    sec_tickers = load_ticker_cik_map()
    if sec_tickers:
        before = len(universe)
        valid = set(sec_tickers.keys())
        universe = {
            t: data
            for t, data in universe.items()
            if t in valid or t.replace(".", "-") in valid
        }
        logger.info(
            "Universe: %d candidates after SEC validation (was %d)",
            len(universe),
            before,
        )
    else:
        logger.warning(
            "SEC ticker map unavailable — keeping all %d regex-extracted "
            "candidates; expect yfinance 404s for noise",
            len(universe),
        )

    logger.info(
        "Sources: insider %d, billionaire %d, watchlist %d",
        len(insider_counts),
        len(hedge_counts),
        len(watchlist),
    )
    return universe
