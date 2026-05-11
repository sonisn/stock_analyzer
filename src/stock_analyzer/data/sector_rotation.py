"""Sector rotation — identify leading vs lagging sectors via SPDR ETF returns.

Cheap data (one yfinance batch call). The ranker prompt uses this so it can
overweight stocks in leading sectors and warn about lagging-sector picks.
"""
from __future__ import annotations

from typing import Any

import yfinance as yf

from ..logging import get_logger

logger = get_logger(__name__)

# yfinance often returns sectors via slight name variants — both keys map to
# the same ETF. Reverse lookup uses the canonical SPDR name set.
SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Financial": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


def fetch_sector_returns(months: int = 6) -> dict[str, float]:
    """Return {canonical_sector_name: pct_return_over_N_months}."""
    unique_etfs = sorted(set(SECTOR_ETFS.values()))
    try:
        data = yf.download(
            unique_etfs,
            period=f"{months}mo",
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        logger.warning("sector returns batch fetch failed: %s", e)
        return {}
    if data is None or data.empty:
        return {}

    closes = data["Close"] if "Close" in data.columns.get_level_values(0) else data

    # Compute per-ETF return, then attribute back to a canonical sector name.
    returns_by_etf: dict[str, float] = {}
    for etf in unique_etfs:
        try:
            series = closes[etf].dropna()
        except (KeyError, TypeError):
            continue
        if len(series) < 2:
            continue
        returns_by_etf[etf] = float(series.iloc[-1] / series.iloc[0] - 1)

    # Map back to canonical sector names. Prefer the first variant per ETF in
    # SECTOR_ETFS dict order so output is deterministic.
    canonical: dict[str, str] = {}
    for sector, etf in SECTOR_ETFS.items():
        canonical.setdefault(etf, sector)

    return {canonical[etf]: ret for etf, ret in returns_by_etf.items() if etf in canonical}


def rank_sectors(
    returns: dict[str, float], top_n: int = 3, bottom_n: int = 3
) -> dict[str, list[str]]:
    if not returns:
        return {"leaders": [], "laggards": []}
    sorted_sectors = sorted(returns.keys(), key=lambda s: returns[s], reverse=True)
    return {
        "leaders": sorted_sectors[:top_n],
        "laggards": list(reversed(sorted_sectors[-bottom_n:])),
    }


def sector_rotation_summary(months: int = 6) -> dict[str, Any]:
    """One-shot fetch + rank. Used by the workflow step."""
    returns = fetch_sector_returns(months)
    ranks = rank_sectors(returns)
    return {
        "lookback_months": months,
        "returns_by_sector": returns,
        "leaders": ranks["leaders"],
        "laggards": ranks["laggards"],
    }


def sector_bias(sector: str | None, summary: dict[str, Any]) -> str:
    """Return 'leader' / 'laggard' / 'neutral' for a candidate's sector."""
    if not sector:
        return "neutral"
    # Map the input sector through SECTOR_ETFS to find its canonical name.
    etf = SECTOR_ETFS.get(sector)
    if not etf:
        return "neutral"
    leaders_etfs = {SECTOR_ETFS.get(s) for s in summary.get("leaders", [])}
    laggards_etfs = {SECTOR_ETFS.get(s) for s in summary.get("laggards", [])}
    if etf in leaders_etfs:
        return "leader"
    if etf in laggards_etfs:
        return "laggard"
    return "neutral"
