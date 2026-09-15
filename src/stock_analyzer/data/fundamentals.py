"""Fundamentals fetch for mid-long term screening. yfinance-based.

Returns the fields the screen and analyst stages need. Missing fields are
left as None; downstream filters treat None as 'failed' (conservative).
"""

from __future__ import annotations

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


def batch_fundamentals(tickers: list[str]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for ticker, r in yf_gateway.map_symbols(fetch_fundamentals, tickers, workers=_MAX_WORKERS):
        if r:
            results[ticker] = r
    return results
