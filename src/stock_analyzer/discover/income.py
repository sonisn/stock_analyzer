"""Dividend income — forward and received. No LLM.

Forward income is shares × the company's current annual dividend rate
(yfinance `dividendRate`, else the trailing annual rate); received income
is the DIVIDEND activity of the last 12 months from the brokerage, split
into what was reinvested automatically (a same-day REI) and what arrived
as cash — cash that is waiting for a destination.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from ..data import yf_gateway


def forward_dividend_rates(tickers: list[str]) -> dict[str, float]:
    """{ticker: annual dividend $/share} for payers; non-payers left out."""
    out: dict[str, float] = {}
    for t in tickers:
        info = yf_gateway.ticker_call(t, "ticker.info", lambda tk: tk.info or {}, default={}) or {}
        rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
        try:
            if rate and float(rate) > 0:
                out[t] = float(rate)
        except TypeError, ValueError:
            continue
    return out


def dividend_income(
    *,
    units: dict[str, float],
    values: dict[str, float],
    rates: dict[str, float],
    received: list[dict[str, Any]],
    today: date | None = None,
) -> dict[str, Any]:
    """Portfolio dividend picture: forward annual income and yield, what
    the last 12 months paid (reinvested vs cash), and the payers."""
    today = today or date.today()
    since = today - timedelta(days=365)
    recent = [d for d in received if d["date"] >= since]
    by_ticker: dict[str, dict[str, Any]] = {}
    for t, rate in rates.items():
        annual = units.get(t, 0.0) * rate
        if annual <= 0:
            continue
        value = values.get(t) or 0.0
        by_ticker[t] = {
            "ticker": t,
            "annual": annual,
            "yield_pct": annual / value * 100 if value else None,
            "received_12m": 0.0,
            "reinvested": None,
        }
    for d in recent:
        row = by_ticker.setdefault(
            d["ticker"],
            {
                "ticker": d["ticker"],
                "annual": 0.0,
                "yield_pct": None,
                "received_12m": 0.0,
                "reinvested": None,
            },
        )
        row["received_12m"] += d["amount"]
        row["reinvested"] = bool(row["reinvested"] or d.get("reinvested"))
    total_value = sum(values.values())
    forward = sum(r["annual"] for r in by_ticker.values())
    cash_rows = [d for d in recent if not d.get("reinvested")]
    return {
        "forward_annual": forward,
        "yield_pct": forward / total_value * 100 if total_value else None,
        "received_12m": sum(d["amount"] for d in recent),
        "reinvested_12m": sum(d["amount"] for d in recent if d.get("reinvested")),
        "cash_12m": sum(d["amount"] for d in cash_rows),
        "cash_accounts": sorted({d["account"] for d in cash_rows}),
        "rows": sorted(by_ticker.values(), key=lambda r: -(r["annual"] or r["received_12m"])),
    }
