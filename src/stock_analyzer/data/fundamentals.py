"""Fundamentals fetch for mid-long term screening. yfinance-based.

Returns the fields the screen and analyst stages need. Missing fields are
left as None; downstream filters treat None as 'failed' (conservative).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from ..logging import get_logger
from . import bar_store, fetch_cache, frames, yf_gateway

logger = get_logger(__name__)

# Fan-out width for the batch. The real ceiling on concurrent requests is
# yf_gateway's process-wide semaphore + pacer, which every pipeline step
# shares; this only decides how many of this batch's tickers queue behind
# it at once. Before the gateway existed, each module owned its own pool
# and the parallel market-data block put 30+ requests in flight, which is
# what got the run rate-limited.
_MAX_WORKERS = 8


# Companies report about every 13 weeks.
_QUARTER = timedelta(days=91)


def _next_report(earnings_ts: float | None, today: date) -> str | None:
    """The next report date, ISO. Yahoo keeps showing the LAST report until
    the next is scheduled (MSFT read 2026-07-29 on 2026-09-27), so a past
    date is rolled forward by quarters. If that guess is early, the cache
    just refetches and gets Yahoo's real date."""
    if not earnings_ts:
        return None
    when = datetime.fromtimestamp(earnings_ts).date()
    while when < today:
        when += _QUARTER
    return when.isoformat()


def fetch_fundamentals(ticker: str) -> dict[str, Any] | None:
    # One request: `info` carries trailing-twelve-month operating cash flow.
    # (This used to read the latest QUARTER from a second endpoint, which
    # doubled the requests and failed seasonal businesses on one weak
    # quarter.)
    info = yf_gateway.ticker_call(ticker, "fundamentals.info", lambda t: t.info or {})
    if not info:
        return None

    market_cap = info.get("marketCap")
    debt = info.get("totalDebt") or 0
    equity = info.get("totalStockholderEquity")
    if not equity:
        book = info.get("bookValue") or 0
        shares = info.get("sharesOutstanding") or 0
        equity = book * shares if book and shares else None
    debt_to_equity = (debt / equity) if equity else None

    fcf = info.get("freeCashflow")
    fcf_yield = (fcf / market_cap) if (fcf and market_cap) else None

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    target_mean = info.get("targetMeanPrice")
    # Forward upside helps the LLM reason: positive = analysts see upside.
    target_upside_pct = None
    if current_price and target_mean:
        try:
            target_upside_pct = (target_mean - current_price) / current_price
        except TypeError, ZeroDivisionError:
            target_upside_pct = None

    ocf = info.get("operatingCashflow")
    earnings_ts = info.get("earningsTimestamp")
    return {
        "total_cash": info.get("totalCash"),
        "ticker": ticker,
        # When and at what price this was fetched, and the next report:
        # the cache (fetch_cache) expires it after that report, and
        # `reprice` brings the price-dependent fields to a later close.
        "quote_price": current_price,
        "quote_date": date.today().isoformat(),
        "next_earnings": _next_report(earnings_ts, date.today()),
        "name": info.get("shortName") or info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": market_cap,
        "revenue_growth_yoy": info.get("revenueGrowth"),
        "earnings_growth_yoy": info.get("earningsGrowth"),
        "operating_cash_flow": ocf,
        "free_cash_flow": fcf,
        "fcf_yield": fcf_yield,
        "debt_to_equity": debt_to_equity,
        "gross_margin": info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
        "return_on_equity": info.get("returnOnEquity"),
        "profit_margin": info.get("profitMargins"),
        # Forward-looking fields used by analyst + reviewer for forward thesis.
        "forward_pe": info.get("forwardPE"),
        "trailing_pe": info.get("trailingPE"),
        "peg_ratio": info.get("pegRatio"),
        "forward_eps": info.get("forwardEps"),
        "trailing_eps": info.get("trailingEps"),
        "analyst_target_mean": target_mean,
        "analyst_target_high": info.get("targetHighPrice"),
        "analyst_target_low": info.get("targetLowPrice"),
        "analyst_target_upside_pct": target_upside_pct,
        "analyst_recommendation": info.get("recommendationKey"),
        "analyst_recommendation_mean": info.get("recommendationMean"),
        "analyst_count": info.get("numberOfAnalystOpinions"),
        "shares_short_pct": info.get("shortPercentOfFloat"),
        # Days-to-cover (short ratio): how many days of average volume it
        # would take all short sellers to buy back. >5 = squeezable;
        # <1 = shorts can exit on any news.
        "short_ratio_days": info.get("shortRatio"),
    }


# How old the newest filed quarter may be before yfinance's trailing
# numbers are the better answer. A 10-Q lands ~45 days after quarter end,
# so 150 days means a filing has been missed — BE sat at 2026-03-31 on
# 2026-09-20 while every other holding had a June quarter.
FILED_MAX_AGE_DAYS = 150

# The fields Wisesheets derives from the filings themselves. Everything
# forward-looking (estimates, targets, recommendations, short interest)
# has no counterpart there and stays exactly as yfinance reported it.
_FILED_FIELDS = (
    "gross_margin",
    "operating_margin",
    "profit_margin",
    "free_cash_flow",
    "debt_to_equity",
)


def _overlay_filed_values(results: dict[str, dict[str, Any]], *, as_of: date | None = None) -> None:
    """Replace the derived fundamentals with the figures as filed with the
    SEC, in place, and record what disagreed.

    yfinance understated NVDA's trailing free cash flow as $41.8B against
    $127.0B in the filings, and GOOGL's as $22.7B against $53.3B — errors
    that flow straight into the screen's scoring. The filed value wins,
    except where the two are too far apart to both be right (see
    `reconcile_fundamentals`), and every substitution is auditable: the
    filing's period and URL go on the row.
    """
    from ..discover.data_reconciliation import flag_non_usd_fundamentals, reconcile_fundamentals
    from . import wisesheets

    # Independent of Wisesheets: a foreign issuer's balance sheet arrives
    # in its own currency and nothing downstream would otherwise notice.
    for ticker, row in (results or {}).items():
        currency_flag = flag_non_usd_fundamentals(ticker, row)
        if currency_flag:
            row["data_reconciliation_flags"] = [
                *(row.get("data_reconciliation_flags") or []),
                currency_flag,
            ]

    if not results or not wisesheets.is_configured():
        return
    try:
        filed = wisesheets.fetch_trailing_fundamentals(list(results), as_of=as_of)
    except Exception as e:  # noqa: BLE001 — a second opinion is never worth a run
        logger.warning("Filed fundamentals unavailable (%s) — using yfinance alone", e)
        return

    horizon = as_of or date.today()
    for ticker, row in results.items():
        values = filed.get(ticker)
        if not values:
            continue
        period_end = values.get("period_end")
        age = (horizon - date.fromisoformat(period_end)).days if period_end else None
        if age is not None and age > FILED_MAX_AGE_DAYS:
            row["filed_note"] = (
                f"newest filing is {period_end} ({age}d old) — kept yfinance's trailing figures"
            )
            logger.info("%s: filed data stale (%s) — keeping yfinance", ticker, period_end)
            continue

        warnings, rejected = reconcile_fundamentals(row, values)
        replaced = []
        for field in _FILED_FIELDS:
            value = values.get(field)
            if value is None or field in rejected:
                continue
            if row.get(field) != value:
                replaced.append(field)
            row[field] = value
        row["filed_period_end"] = period_end
        row["filed_source_url"] = values.get("filing_url")
        row["filed_fields"] = replaced
        if warnings:
            row["data_reconciliation_flags"] = [
                *(row.get("data_reconciliation_flags") or []),
                *warnings,
            ]
        # fcf_yield is derived, so it has to follow the value it came from.
        if "free_cash_flow" in replaced and row.get("market_cap"):
            row["fcf_yield"] = row["free_cash_flow"] / row["market_cap"]


def _stored_close(ticker: str) -> tuple[date, float] | None:
    """The latest close in the on-disk bar store (no request), or None."""
    stored = bar_store.load(ticker)
    if stored is None or stored.frame.is_empty():
        return None
    last = stored.frame.tail(1)
    close = last["Close"][0]
    return (last[frames.DATE][0], float(close)) if close else None


# Fields that move with the share price, and how: market cap and the
# multiples scale with it, yields scale against it.
_WITH_PRICE = ("market_cap", "forward_pe", "trailing_pe", "peg_ratio")
_AGAINST_PRICE = ("fcf_yield",)


def reprice(row: dict[str, Any], close: tuple[date, float] | None) -> None:
    """Bring a cached answer's price-dependent fields to a later close, in
    place. Only when the close is from after the day it was fetched, so a
    fresh answer keeps Yahoo's live price."""
    then, fetched_on = row.get("quote_price"), row.get("quote_date")
    if close is None or not then or not fetched_on or close[0].isoformat() <= fetched_on:
        return
    ratio = close[1] / then
    for field in _WITH_PRICE:
        if row.get(field) is not None:
            row[field] = row[field] * ratio
    for field in _AGAINST_PRICE:
        if row.get(field) is not None:
            row[field] = row[field] / ratio
    if row.get("analyst_target_mean"):
        row["analyst_target_upside_pct"] = row["analyst_target_mean"] / close[1] - 1
    row["quote_price"], row["quote_date"] = close[1], close[0].isoformat()


def batch_fundamentals(
    tickers: list[str], *, as_of: date | None = None, refresh: list[str] | tuple[str, ...] = ()
) -> dict[str, dict[str, Any]]:
    # The Yahoo answers are cached for a week (fetch_cache) and repriced to
    # the latest stored close; the filed-value overlay is one batched call
    # and runs fresh on every batch.
    results: dict[str, dict[str, Any]] = fetch_cache.fetch_many(
        "fundamentals",
        tickers,
        lambda todo: yf_gateway.map_symbols(fetch_fundamentals, todo, workers=_MAX_WORKERS),
        refresh=refresh,
    )
    for ticker, row in results.items():
        reprice(row, _stored_close(ticker))
    _overlay_filed_values(results, as_of=as_of)
    return results
