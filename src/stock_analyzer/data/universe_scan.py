"""Every US-listed stock worth $2B or more, minus the ones that are traps.

The S&P indexes admit only profitable US companies, which leaves out most
of what this portfolio actually holds: pre-profit growth names (OKLO,
SMR), foreign companies listed here (TSM, ARM) and mid-caps like DINO.
Yahoo's screener lists every US-listed equity by market cap; `scan` asks
it for the ones that also pass QUALITY_RULES (growth, debt, cash flow,
return on equity — ~400 names, 2 requests) and `investable` drops, from
the screener's own fields — no extra request:

  - OTC listings: only NYSE, Nasdaq and NYSE American;
  - anything but common stock (quoteType EQUITY), and by name funds,
    shells and special-purpose acquisition companies, preferreds,
    warrants, units and rights;
  - prices under MIN_PRICE (penny-stock behaviour) for companies under
    PRICE_RULE_BELOW_CAP;
  - average daily dollar volume under MIN_DOLLAR_VOLUME (can't be traded
    without moving the price, and no usable options);
  - less than MIN_HISTORY_DAYS of trading (the trend features need a year);
  - a second share class of the same company (the most traded one stays).

Unprofitable companies get two more checks in the discover screen, where
their fundamentals are fetched anyway (discover/screen.financial_health).

The nightly earnings-watch job rescans (fundamentals move every earnings
season) and writes ~/.stock_analyzer/us_2b_universe.txt; `uv run ops
universe` does the same by hand. Without that file a run falls back to the
bundled snapshot (data/static/us_2b_universe.txt, taken WITHOUT the quality
rules, so it is ~1,900 names and slow).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from ..logging import get_logger

logger = get_logger(__name__)

MIN_MARKET_CAP = 2e9
MIN_PRICE = 5.0
# A low share price is a penny-stock warning only for a small company:
# Stellantis ($17B), Ambev and Bradesco trade under $5 and are not traps.
PRICE_RULE_BELOW_CAP = 10e9
MIN_DOLLAR_VOLUME = 10e6
MIN_HISTORY_DAYS = 365
# Yahoo's exchange codes: Nasdaq Global Select / Global / Capital, NYSE,
# NYSE American.
LISTED_EXCHANGES = frozenset({"NMS", "NGM", "NCM", "NYQ", "ASE"})
PAGE = 250

# The business-quality rules, applied by Yahoo's screener itself so the
# whole market costs two requests instead of one per company. Yahoo's
# fields are in percent. The first three are the discover hard filter's
# own (screen.MIN_REVENUE_GROWTH, MAX_DEBT_TO_EQUITY, positive operating
# cash flow); the last three were added 2026-09-27 to keep the screen to
# ~400 names: a business that funds itself, earns a real return on its
# equity, and grew over the whole year, not just the latest quarter. On
# that day they took the >= $2B market from 824 to 395. Pre-profit
# companies no longer enter this way; holdings, the watchlist, earnings
# standouts and insider clusters still join the frame on their own.
QUALITY_RULES: tuple[tuple[str, str, float], ...] = (
    ("gte", "quarterlyrevenuegrowth.quarterly", 8),
    ("lte", "totaldebtequity.lasttwelvemonths", 200),
    ("gt", "cashfromoperations.lasttwelvemonths", 0),
    ("gt", "leveredfreecashflow.lasttwelvemonths", 0),
    ("gte", "returnonequity.lasttwelvemonths", 10),
    ("gte", "totalrevenues1yrgrowth.lasttwelvemonths", 8),
)

# Not operating companies, or not the common stock of one. Matched on the
# name; "Trust" is deliberately absent (REITs and banks carry it).
_NOT_COMMON = re.compile(
    r"\b(acquisition corp|acquisition co|acquisition holdings|blank check|spac|"
    r"fund|funds|etf|etn|preferred|depositary shares|warrants?|units?|rights?|notes due)\b",
    re.IGNORECASE,
)
# Share-class suffixes that are not common stock: preferreds (-P...),
# warrants (-W...), units (-U), rights (-R). BRK-B and BF-B stay.
_SUFFIX = re.compile(r"-(P[A-Z]*|W[A-Z]*|U|UN|R|RT)$")

_COLUMNS = {
    "symbol": pl.String,
    "name": pl.String,
    "exchange": pl.String,
    "quote_type": pl.String,
    "market_cap": pl.Float64,
    "price": pl.Float64,
    "avg_volume": pl.Float64,
    "first_trade": pl.Date,
}


def _row(q: dict[str, Any]) -> dict[str, Any]:
    ms = q.get("firstTradeDateMilliseconds")
    return {
        "symbol": q.get("symbol"),
        "name": q.get("longName") or q.get("shortName") or "",
        "exchange": q.get("exchange"),
        "quote_type": q.get("quoteType"),
        "market_cap": q.get("marketCap"),
        "price": q.get("regularMarketPrice"),
        "avg_volume": q.get("averageDailyVolume3Month"),
        "first_trade": datetime.fromtimestamp(ms / 1000).date() if ms else None,
    }


def scan(
    min_market_cap: float = MIN_MARKET_CAP,
    *,
    quality: bool = True,
    screen: Callable[..., dict[str, Any]] | None = None,
    pause: float = 1.0,
) -> pl.DataFrame:
    """Every US-listed equity at or above `min_market_cap` (that also
    passes QUALITY_RULES unless `quality` is off), as one row each."""
    if screen is None:
        import yfinance as yf
        from yfinance import EquityQuery as Q

        # Operand lists mix field names and values; yfinance's stubs want one type.
        region: list[Any] = ["region", "us"]
        cap: list[Any] = ["intradaymarketcap", min_market_cap]
        exchanges: list[Any] = ["exchange", *sorted(LISTED_EXCHANGES)]
        rules: list[Any] = []
        for op, field, value in QUALITY_RULES if quality else ():
            operand: list[Any] = [field, value]
            rules.append(Q(op, operand))  # ty: ignore[invalid-argument-type]  # op is a str
        query = Q("and", [Q("eq", region), Q("gte", cap), Q("is-in", exchanges), *rules])

        def screen(offset: int) -> dict[str, Any]:
            return yf.screen(
                query, offset=offset, size=PAGE, sortField="intradaymarketcap", sortAsc=False
            )

    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = screen(offset) or {}
        quotes = page.get("quotes") or []
        rows.extend(_row(q) for q in quotes if q.get("symbol"))
        offset += PAGE
        if len(quotes) < PAGE or offset >= int(page.get("total") or 0):
            break
        time.sleep(pause)  # a paced crawl, not a burst
    logger.info(
        "Universe scan: %d US-listed equities >= $%.0fB%s",
        len(rows),
        min_market_cap / 1e9,
        " passing the quality rules" if quality else "",
    )
    return pl.DataFrame(rows, schema=_COLUMNS, orient="row").unique("symbol", keep="first")


def investable(rows: pl.DataFrame, *, today: date | None = None) -> pl.DataFrame:
    """The rows that pass every filter in the module docstring, with a
    `dollar_volume` column; sorted by market cap, largest first."""
    today = today or date.today()
    listed_by = today - timedelta(days=MIN_HISTORY_DAYS)
    out = rows.with_columns((pl.col("avg_volume") * pl.col("price")).alias("dollar_volume"))
    out = out.filter(
        pl.col("exchange").is_in(sorted(LISTED_EXCHANGES))
        & (pl.col("quote_type") == "EQUITY")
        & (pl.col("market_cap") >= MIN_MARKET_CAP)
        & ((pl.col("price") >= MIN_PRICE) | (pl.col("market_cap") >= PRICE_RULE_BELOW_CAP))
        & (pl.col("dollar_volume") >= MIN_DOLLAR_VOLUME)
        & (pl.col("first_trade") <= listed_by)
    )
    keep = [
        not _NOT_COMMON.search(name or "") and not _SUFFIX.search(sym or "")
        for sym, name in zip(out["symbol"].to_list(), out["name"].to_list(), strict=True)
    ]
    out = out.filter(pl.Series(keep))
    # One listing per company: GOOGL and GOOG, BRK-A and BRK-B, FOX and
    # FOXA would each take a screen slot on the same business. Keep the
    # share class with the most dollar volume (the one that trades).
    # A row with no name is keyed by its symbol, so it never collides.
    company = pl.when(pl.col("name").str.strip_chars() != "").then(pl.col("name"))
    out = (
        out.with_columns(company.otherwise(pl.col("symbol")).alias("_company"))
        .sort("dollar_volume", descending=True)
        .unique("_company", keep="first", maintain_order=True)
        .drop("_company")
    )
    return out.sort("market_cap", descending=True)


def symbols(rows: pl.DataFrame) -> list[str]:
    """Tickers in the package's spelling (BRK.B -> BRK-B), in row order."""
    return [s.upper().replace(".", "-") for s in rows["symbol"].to_list()]
