"""Fundamentals fetch for mid-long term screening. yfinance-based.

Returns the fields the screen and analyst stages need. Missing fields are
left as None; downstream filters treat None as 'failed' (conservative).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import yfinance as yf

from ..logging import get_logger

logger = get_logger(__name__)

_MAX_WORKERS = 5

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
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        cashflow = t.quarterly_cashflow
    except Exception as e:
        logger.warning("fundamentals fetch failed for %s: %s", ticker, e)
        return None
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
        except (TypeError, ZeroDivisionError):
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
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for ticker, r in zip(tickers, ex.map(fetch_fundamentals, tickers), strict=False):
            if r:
                results[ticker] = r
    return results
