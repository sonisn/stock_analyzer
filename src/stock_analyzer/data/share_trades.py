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

from typing import Any

import polars as pl

from ..logging import get_logger
from . import frames, yf_gateway

logger = get_logger(__name__)

_MAX_WORKERS = 5
_MAX_RECENT_TRANSACTIONS = 10


def _data_columns(df: pl.DataFrame) -> list[str]:
    """yfinance's own columns (frames.table_from_pandas adds `index`)."""
    return [c for c in df.columns if c != "index"]


def _present(v: Any) -> bool:
    return v is not None and v == v


def _labelled(df: pl.DataFrame | None, label: str, position: int) -> Any:
    """The value in data column `position` of the row whose first data
    column reads `label` (insider_purchases: label, Shares, Trans)."""
    if df is None or df.is_empty():
        return None
    cols = _data_columns(df)
    if len(cols) <= position:
        return None
    matches = df.filter(pl.col(cols[0]).cast(pl.String) == label)
    return None if matches.is_empty() else matches[cols[position]][0]


def _row_value(df: pl.DataFrame | None, label: str) -> float | None:
    """Value column of the insider-purchases row named `label`."""
    v = _labelled(df, label, 1)
    try:
        return float(v) if _present(v) else None
    except TypeError, ValueError:
        return None


def _row_count(df: pl.DataFrame | None, label: str) -> int | None:
    """Transaction-count column of the insider-purchases row named `label`."""
    v = _labelled(df, label, 2)
    try:
        return int(v) if _present(v) else None
    except TypeError, ValueError:
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


def _fetch_holder_tables(t: Any) -> tuple[Any, Any, Any, Any]:
    """All four holder tables come off the same Yahoo holders payload, which
    yfinance caches on the Ticker — so this is one paced request, not four."""
    return (
        t.insider_purchases,
        t.insider_transactions,
        t.institutional_holders,
        t.major_holders,
    )


def fetch_share_trade_data(ticker: str) -> dict[str, Any] | None:
    tables = yf_gateway.ticker_call(ticker, "share_trades", _fetch_holder_tables)
    if tables is None:
        return None
    ip, it, ih, mh = (frames.table_from_pandas(t) for t in tables)

    out: dict[str, Any] = {"ticker": ticker}

    if ip is not None:
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

    if it is not None:
        transactions: list[dict[str, Any]] = []
        for row in it.head(_MAX_RECENT_TRANSACTIONS).iter_rows(named=True):
            tx: dict[str, Any] = {
                "shares": int(row["Shares"]) if _present(row.get("Shares")) else None,
                "value_usd": float(row["Value"]) if _present(row.get("Value")) else None,
                "date": str(row["Transaction Start Date"])
                if _present(row.get("Transaction Start Date"))
                else None,
                "ownership": str(row["Ownership"]) if _present(row.get("Ownership")) else None,
            }
            for col_name in ("Insider", "Text"):
                if _present(row.get(col_name)):
                    tx[col_name.lower()] = str(row[col_name])
            transactions.append(tx)
        out["insider_recent_transactions"] = transactions

    if mh is not None:
        try:
            # major_holders is a 1-column table indexed by metric name.
            mh_dict: dict[str, float] = {}
            value_cols = _data_columns(mh)
            for row in mh.iter_rows(named=True):
                val: Any = row[value_cols[0]] if value_cols else None
                if _present(val):
                    try:
                        mh_dict[str(row["index"])] = float(val)
                    except ValueError, TypeError:
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

    if ih is not None:
        top_holders: list[dict[str, Any]] = []
        for row in ih.head(5).iter_rows(named=True):
            top_holders.append(
                {
                    "holder": str(row["Holder"]) if _present(row.get("Holder")) else None,
                    "value_usd": float(row["Value"]) if _present(row.get("Value")) else None,
                    "pct_change": float(row["pctChange"])
                    if _present(row.get("pctChange"))
                    else None,
                    "date_reported": str(row["Date Reported"])
                    if _present(row.get("Date Reported"))
                    else None,
                }
            )
        out["top_institutional_holders"] = top_holders

    if len(out) == 1:  # only "ticker" key — no data
        return None
    return out


def batch_share_trade_data(tickers: list[str]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for ticker, data in yf_gateway.map_symbols(
        fetch_share_trade_data, tickers, workers=_MAX_WORKERS
    ):
        if data:
            results[ticker] = data
    return results
