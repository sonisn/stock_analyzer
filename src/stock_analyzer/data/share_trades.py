"""Share trade data via yfinance — insider Form 4 + institutional 13F.

Pulls four sources per ticker:
  - 6-month insider purchase/sale aggregate (count + share volume + net %)
  - Recent individual insider transactions (last ~3mo, Form 4 -> yfinance)
  - Top 5 institutional holders + their QoQ position change
  - Major-holder percentages (insider % vs institution %)

All via yfinance (no extra API key). Adds a derived `insider_signal`
classification so the LLM can latch onto the signal without parsing
the raw aggregate every time.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import yfinance as yf

from ..logging import get_logger

logger = get_logger(__name__)

_MAX_WORKERS = 5
_MAX_RECENT_TRANSACTIONS = 10


def _row_value(df: pd.DataFrame | None, label: str) -> float | None:
    """Pull a single value from a 2-column dataframe keyed by first-column label."""
    if df is None or df.empty:
        return None
    matches = df[df.iloc[:, 0] == label]
    if matches.empty:
        return None
    v = matches.iloc[0, 1]
    try:
        return float(v) if pd.notna(v) else None
    except (ValueError, TypeError):
        return None


def _row_count(df: pd.DataFrame | None, label: str) -> int | None:
    """Pull the 'Trans' (count) column for a given label."""
    if df is None or df.empty or len(df.columns) < 3:
        return None
    matches = df[df.iloc[:, 0] == label]
    if matches.empty:
        return None
    v = matches.iloc[0, 2]
    try:
        return int(v) if pd.notna(v) else None
    except (ValueError, TypeError):
        return None


def _classify_insider_signal(summary: dict[str, Any]) -> str:
    """Map the 6mo insider aggregate to a coarse signal label."""
    pct = summary.get("net_pct_of_held") or 0
    net = summary.get("net_shares") or 0
    if pct >= 0.05 or net >= 5_000_000:
        return "heavy_buying"
    if pct >= 0.01 or net >= 500_000:
        return "modest_buying"
    if pct <= -0.05 or net <= -5_000_000:
        return "heavy_selling"
    if pct <= -0.01 or net <= -500_000:
        return "modest_selling"
    return "neutral"


def fetch_share_trade_data(ticker: str) -> dict[str, Any] | None:
    try:
        t = yf.Ticker(ticker)
        ip = t.insider_purchases
        it = t.insider_transactions
        ih = t.institutional_holders
        mh = t.major_holders
    except Exception as e:
        logger.warning("share trade fetch failed for %s: %s", ticker, e)
        return None

    out: dict[str, Any] = {"ticker": ticker}

    if ip is not None and not ip.empty:
        summary = {
            "purchases_shares": _row_value(ip, "Purchases"),
            "sales_shares": _row_value(ip, "Sales"),
            "net_shares": _row_value(ip, "Net Shares Purchased (Sold)"),
            "purchases_count": _row_count(ip, "Purchases"),
            "sales_count": _row_count(ip, "Sales"),
            "net_pct_of_held": _row_value(ip, "% Net Shares Purchased (Sold)"),
            "total_insider_shares_held": _row_value(ip, "Total Insider Shares Held"),
        }
        summary["insider_signal"] = _classify_insider_signal(summary)
        out["insider_summary_6mo"] = summary

    if it is not None and not it.empty:
        transactions: list[dict[str, Any]] = []
        for _, row in it.head(_MAX_RECENT_TRANSACTIONS).iterrows():
            tx: dict[str, Any] = {
                "shares": int(row["Shares"])
                if "Shares" in row and pd.notna(row["Shares"])
                else None,
                "value_usd": float(row["Value"])
                if "Value" in row and pd.notna(row["Value"])
                else None,
                "date": str(row["Transaction Start Date"])
                if "Transaction Start Date" in row
                and pd.notna(row["Transaction Start Date"])
                else None,
                "ownership": str(row["Ownership"])
                if "Ownership" in row and pd.notna(row["Ownership"])
                else None,
            }
            for col_name in ("Insider", "Text"):
                if col_name in it.columns and pd.notna(row.get(col_name)):
                    tx[col_name.lower()] = str(row[col_name])
            transactions.append(tx)
        out["insider_recent_transactions"] = transactions

    if mh is not None and not mh.empty:
        try:
            # major_holders is a 1-column DataFrame indexed by metric name.
            mh_dict: dict[str, float] = {}
            for idx, row in mh.iterrows():
                val = row.iloc[0] if len(row) else None
                if val is not None and pd.notna(val):
                    try:
                        mh_dict[str(idx)] = float(val)
                    except (ValueError, TypeError):
                        continue
        except Exception:
            mh_dict = {}
        out["ownership_summary"] = {
            "pct_held_by_insiders": mh_dict.get("insidersPercentHeld"),
            "pct_held_by_institutions": mh_dict.get("institutionsPercentHeld"),
            "institution_count": int(mh_dict["institutionsCount"])
            if mh_dict.get("institutionsCount")
            else None,
        }

    if ih is not None and not ih.empty:
        top_holders: list[dict[str, Any]] = []
        for _, row in ih.head(5).iterrows():
            top_holders.append({
                "holder": str(row["Holder"])
                if "Holder" in row and pd.notna(row["Holder"])
                else None,
                "value_usd": float(row["Value"])
                if "Value" in row and pd.notna(row["Value"])
                else None,
                "pct_change": float(row["pctChange"])
                if "pctChange" in row and pd.notna(row["pctChange"])
                else None,
                "date_reported": str(row["Date Reported"])
                if "Date Reported" in row and pd.notna(row["Date Reported"])
                else None,
            })
        out["top_institutional_holders"] = top_holders

    if len(out) == 1:  # only "ticker" key — no data
        return None
    return out


def batch_share_trade_data(tickers: list[str]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for ticker, data in zip(tickers, ex.map(fetch_share_trade_data, tickers), strict=False):
            if data:
                results[ticker] = data
    return results
