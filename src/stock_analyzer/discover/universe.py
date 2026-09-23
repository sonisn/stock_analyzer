"""Build the candidate universe for discovery.

The universe has two distinct layers, and keeping them separate matters
more than anything else in this module:

  SAMPLING FRAME — which names are ELIGIBLE to be picked at all. This is
  the S&P 500 snapshot in `data/universe_base.py`, plus the user's
  watchlist and current holdings. A filter can only remove names the frame
  already contained, so the frame sets the ceiling on pick quality.

  CONVICTION OVERLAY — which of those names the press has been talking
  about: recent insider-buying coverage and hedge-fund/billionaire
  coverage, extracted from article text by regex. These ADD a conviction
  signal to names already in the frame, and may also admit an off-frame
  name that survives SEC validation.

Earlier versions used the overlay AS the frame, which made "appeared in
last month's coverage" a precondition for every pick — a recency and
popularity filter applied before any fundamental screen, and one that by
construction only surfaces theses already in print. The old rationale
("the screen's RS_6mo>0 + price>200DMA filters already select for
index-leading names") conflated filtering with sampling.

Mention counts feed a coarse `conviction` integer. It is a MEDIA ATTENTION
measure, not an edge — high attention is associated with crowding — so the
screen weights it lightly and `in_base_universe` / `sources` are what
callers should reason about. Watchlist membership grants eligibility, not
score: see `screen._score_conviction`.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..data.hedge_funds import fetch_hedge_fund_trades
from ..data.insider import fetch_insider_trades
from ..data.sec_edgar import load_ticker_cik_map
from ..data.universe_base import load_base_universe
from ..logging import get_logger

logger = get_logger(__name__)

# Common acronyms / English words that match the bare [A-Z]{2,5} pattern.
# Adding to this list is the right move when you see noise in the universe.
_FALSE_POSITIVES = frozenset(
    {
        "AN",
        "AND",
        "ALL",
        "AM",
        "AS",
        "AT",
        "BE",
        "BY",
        "DO",
        "FOR",
        "FROM",
        "GO",
        "HAS",
        "HE",
        "I",
        "IF",
        "IN",
        "IS",
        "IT",
        "ITS",
        "MY",
        "NEW",
        "NO",
        "NOT",
        "OF",
        "OK",
        "ON",
        "OR",
        "SO",
        "TO",
        "UP",
        "US",
        "WE",
        "YOU",
        "THE",
        "USA",
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CNY",
        "CEO",
        "CFO",
        "COO",
        "CTO",
        "CMO",
        "CIO",
        "FED",
        "FOMC",
        "GDP",
        "CPI",
        "PPI",
        "PMI",
        "FDA",
        "SEC",
        "IRS",
        "DOJ",
        "FTC",
        "DOE",
        "EPA",
        "DOD",
        "NSA",
        "CIA",
        "FBI",
        "NYSE",
        "AMEX",
        "OTC",
        "ETF",
        "IPO",
        "FYI",
        "AI",
        "ML",
        "AR",
        "VR",
        "EV",
        "OS",
        "PR",
        "PE",
        "EPS",
        "ROE",
        "ROI",
        "ROA",
        "FY",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "YOY",
        "QOQ",
        "YTD",
        "MTD",
        "AGM",
        "PIE",
        "PT",
        "ST",
        "MT",
        "LT",
        "UK",
        "EU",
        "ASEAN",
        "G7",
        "G20",
    }
)

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


def _entry(universe: dict[str, dict[str, Any]], ticker: str) -> dict[str, Any]:
    return universe.setdefault(
        ticker,
        {"sources": [], "conviction": 0, "in_base_universe": False},
    )


def _add_frame(universe: dict[str, dict[str, Any]], tickers: tuple[str, ...], source: str) -> None:
    """Names that are always eligible: the index, the watchlist, holdings."""
    for ticker in tickers:
        u = _entry(universe, ticker)
        u["in_base_universe"] = True
        if source not in u["sources"]:
            u["sources"].append(source)


def _add_overlay(
    universe: dict[str, dict[str, Any]], counts: dict[str, int], source: str, *, weight: int
) -> None:
    """News mentions: attention, weighted into `conviction`."""
    for ticker, n in counts.items():
        u = _entry(universe, ticker)
        if source not in u["sources"]:
            u["sources"].append(source)
        u["conviction"] += n * weight


def _drop_unlisted(universe: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate against the SEC's authoritative ticker→CIK map.

    The regex-based extraction catches a lot of English words (HOME, TABLE,
    OFF, LP, LLC, etc.) that aren't real listings; this drops them before
    they hit yfinance. Tickers with share classes appear in SEC data with a
    dash (e.g. BRK-B); accept both BRK.B and BRK-B forms.
    """
    sec_tickers = load_ticker_cik_map()
    if not sec_tickers:
        logger.warning(
            "SEC ticker map unavailable — keeping all %d regex-extracted "
            "candidates; expect yfinance 404s for noise",
            len(universe),
        )
        return universe
    before = len(universe)
    valid = set(sec_tickers.keys())
    # Only the regex-derived names need policing. A frame entry (index /
    # watchlist / holding) is there because a human or an index listed
    # it, so an SEC map that is stale or fetched badly must not silently
    # empty the sampling frame.
    universe = {
        t: data
        for t, data in universe.items()
        if data.get("in_base_universe") or t in valid or t.replace(".", "-") in valid
    }
    logger.info(
        "Universe: %d candidates after SEC validation (was %d)",
        len(universe),
        before,
    )
    return universe


def build_universe(
    watchlist: tuple[str, ...] = (),
    holdings: tuple[str, ...] = (),
    *,
    base_universe: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return {ticker: {sources, conviction, in_base_universe}}.

    sources: list[str]       — which layers the ticker came from
    conviction: int          — weighted media-mention count (attention, not edge)
    in_base_universe: bool   — was it in the sampling frame, or news-only

    `base_universe` defaults to `load_base_universe()` (the bundled S&P 500
    snapshot, overridable via DISCOVER_UNIVERSE_FILE). Pass an explicit
    tuple in tests or to screen a different frame.
    """
    frame = load_base_universe() if base_universe is None else tuple(base_universe)

    universe: dict[str, dict[str, Any]] = {}

    # --- layer 1: the sampling frame ---
    _add_frame(universe, frame, "index")
    # Watchlist and holdings are part of the frame by definition: the user
    # told us these matter, so they are always eligible for analysis. That
    # is an INCLUSION rule — it deliberately carries no score bonus, so the
    # ranking tests the user's prior instead of confirming it.
    _add_frame(universe, watchlist, "watchlist")
    _add_frame(universe, holdings, "holding")

    # --- layer 2: the conviction overlay ---
    insider_items = fetch_insider_trades(days=30, max_results=40)
    hedge_items = fetch_hedge_fund_trades(days=30, max_results=40)

    insider_counts = _tickers_from_items(insider_items)
    hedge_counts = _tickers_from_items(hedge_items)
    _add_overlay(universe, insider_counts, "insider", weight=1)
    _add_overlay(universe, hedge_counts, "billionaire", weight=2)

    universe = _drop_unlisted(universe)

    news_only = sum(1 for data in universe.values() if not data.get("in_base_universe"))
    logger.info(
        "Universe: %d total — frame %d (index %d + watchlist %d + holdings %d), "
        "news-only %d. Overlay: insider %d, billionaire %d.",
        len(universe),
        len(universe) - news_only,
        len(frame),
        len(watchlist),
        len(holdings),
        news_only,
        len(insider_counts),
        len(hedge_counts),
    )
    if not frame:
        logger.warning(
            "Base universe is EMPTY — the pipeline is running on news-derived "
            "names only, which is the weak sampling frame this layer exists "
            "to replace. Check DISCOVER_UNIVERSE_FILE or the bundled snapshot."
        )
    return universe
