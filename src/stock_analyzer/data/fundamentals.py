"""Fundamentals fetch for mid-long term screening. yfinance-based.

Returns the fields the screen and analyst stages need. Missing fields are
left as None; downstream filters treat None as 'failed' (conservative).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from ..logging import get_logger
from . import yf_gateway

logger = get_logger(__name__)

# Fan-out width for the batch. The real ceiling on concurrent requests is
# yf_gateway's process-wide semaphore + pacer, which every pipeline step
# shares; this only decides how many of this batch's tickers queue behind
# it at once. Before the gateway existed, each module owned its own pool
# and the parallel market-data block put 30+ requests in flight, which is
# what got the run rate-limited.
_MAX_WORKERS = 8

_OCF_ROW_NAMES = (
    "Operating Cash Flow",
    "Total Cash From Operating Activities",
    "Cash Flow From Continuing Operating Activities",
)


def _latest_ocf(quarterly_cashflow: pd.DataFrame | None) -> float | None:
    if quarterly_cashflow is None or quarterly_cashflow.empty:
        return None
    for row in _OCF_ROW_NAMES:
        if row in quarterly_cashflow.index:
            val = quarterly_cashflow.loc[row].iloc[0]
            if val is not None and pd.notna(val):
                return float(val)
    return None


def fetch_fundamentals(ticker: str) -> dict[str, Any] | None:
    # Two paced calls rather than one: `info` and `quarterly_cashflow` are
    # separate Yahoo endpoints, so they each need their own rate-limit slot.
    info = yf_gateway.ticker_call(ticker, "fundamentals.info", lambda t: t.info or {})
    if not info:
        return None
    cashflow = yf_gateway.ticker_call(
        ticker, "fundamentals.cashflow", lambda t: t.quarterly_cashflow
    )

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

    return {
        "ticker": ticker,
        "name": info.get("shortName") or info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": market_cap,
        "revenue_growth_yoy": info.get("revenueGrowth"),
        "earnings_growth_yoy": info.get("earningsGrowth"),
        "operating_cash_flow": _latest_ocf(cashflow),
        "free_cash_flow": fcf,
        "fcf_yield": fcf_yield,
        "debt_to_equity": debt_to_equity,
        "gross_margin": info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
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


def batch_fundamentals(
    tickers: list[str], *, as_of: date | None = None
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for ticker, r in yf_gateway.map_symbols(fetch_fundamentals, tickers, workers=_MAX_WORKERS):
        if r:
            results[ticker] = r
    _overlay_filed_values(results, as_of=as_of)
    return results
