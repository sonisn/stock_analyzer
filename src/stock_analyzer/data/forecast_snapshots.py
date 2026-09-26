"""A nightly, point-in-time record of analysts' forecasts for tracked stocks.

Yahoo shows today's consensus and how it moved over the last 90 days;
anything older is gone. Stored every weekday, the history lets the
forward-return model use estimate revisions as features without seeing
the future (a revision is only known from the day it was stored), and
lets any later review ask what the forecast was on the day of a decision.

Tracked: the stocks the daily email wrote about in the last two weeks
(holdings, and standouts shown), discover picks of the last year, the
last three discover runs' screen survivors and recent earnings
standouts — about 175 names. Three yfinance requests each (quote summary,
EPS trend, revenue estimate), paced by the gateway: a few minutes inside
the nightly `earnings-watch` job. No LLM calls.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import text

from ..db.session import exec_sql, get_session
from ..db.tables import ForecastSnapshot
from ..logging import get_logger
from . import yf_gateway

logger = get_logger(__name__)

MAX_TICKERS = 250
_FIELDS = (
    "price",
    "eps_current_year",
    "eps_next_year",
    "revenue_current_year",
    "revenue_next_year",
    "analysts",
    "target_mean",
    "target_high",
    "target_low",
    "recommendation_mean",
)


def tracked_tickers(db_path: str, *, today: date) -> list[str]:
    since = {
        "views": (today - timedelta(days=14)).isoformat(),
        "picks": (today - timedelta(days=365)).isoformat(),
        "standouts": (today - timedelta(days=60)).isoformat(),
    }
    queries = [
        "SELECT ticker FROM stock_views WHERE written_on >= :views OR news_on >= :views",
        "SELECT DISTINCT p.ticker FROM picks p JOIN runs r ON r.id = p.run_id "
        "WHERE r.run_at >= :picks",
        "SELECT DISTINCT ticker FROM candidates WHERE passed_filter = 1 AND run_id IN "
        "(SELECT id FROM runs WHERE kind = 'discover' ORDER BY id DESC LIMIT 3)",
        "SELECT ticker FROM earnings_events WHERE status = 'standout' AND decided_on >= :standouts",
    ]
    out: list[str] = []
    with get_session(db_path) as session:
        for sql in queries:
            for (t,) in exec_sql(session, text(sql), params=since).all():
                if t and t.upper() not in out:
                    out.append(t.upper())
    return out[:MAX_TICKERS]


def _cell(frame: Any, row: str, col: str) -> float | None:
    try:
        v = frame.loc[row, col]
    except KeyError, AttributeError, TypeError:
        return None
    return None if v is None or pd.isna(v) else float(v)


def _num(info: dict[str, Any], key: str) -> float | None:
    v = info.get(key)
    return float(v) if isinstance(v, int | float) and not pd.isna(v) else None


def fetch_forecast(ticker: str) -> dict[str, Any] | None:
    """Today's consensus for `ticker`, or None when Yahoo has no forecast
    at all (a money-market fund, an unknown symbol)."""
    info = yf_gateway.ticker_call(ticker, "info", lambda t: t.info, default={}) or {}
    trend = yf_gateway.ticker_call(ticker, "eps_trend", lambda t: t.eps_trend)
    revenue = yf_gateway.ticker_call(ticker, "revenue_estimate", lambda t: t.revenue_estimate)
    analysts = _num(info, "numberOfAnalystOpinions")
    row = {
        "price": _num(info, "currentPrice") or _num(info, "regularMarketPrice"),
        "eps_current_year": _cell(trend, "0y", "current"),
        "eps_next_year": _cell(trend, "+1y", "current"),
        "revenue_current_year": _cell(revenue, "0y", "avg"),
        "revenue_next_year": _cell(revenue, "+1y", "avg"),
        "analysts": int(analysts) if analysts is not None else None,
        "target_mean": _num(info, "targetMeanPrice"),
        "target_high": _num(info, "targetHighPrice"),
        "target_low": _num(info, "targetLowPrice"),
        "recommendation_mean": _num(info, "recommendationMean"),
    }
    forecasts = [v for k, v in row.items() if k != "price"]
    return row if any(v is not None for v in forecasts) else None


def record_snapshots(
    db_path: str,
    *,
    today: date,
    tickers: list[str],
    fetch: Callable[[str], dict[str, Any] | None] = fetch_forecast,
) -> dict[str, int]:
    """Store today's forecast for every ticker not stored yet today.
    Returns {"stored", "no_forecast", "already"}."""
    day = today.isoformat()
    with get_session(db_path) as session:
        done = {
            t
            for (t,) in exec_sql(
                session,
                text("SELECT ticker FROM forecast_snapshots WHERE day = :d"),
                params={"d": day},
            ).all()
        }
    todo = [t for t in tickers if t not in done]
    rows: list[ForecastSnapshot] = []
    empty = 0
    for ticker, found in yf_gateway.map_symbols(fetch, todo, workers=4):
        if not found:
            empty += 1
            continue
        rows.append(ForecastSnapshot(ticker=ticker, day=day, **{k: found.get(k) for k in _FIELDS}))
    with get_session(db_path) as session:
        for row in rows:
            session.add(row)
    summary = {"stored": len(rows), "no_forecast": empty, "already": len(done & set(tickers))}
    logger.info(
        "Forecast snapshot %s: %d stored, %d without a forecast, %d already stored today",
        day,
        summary["stored"],
        summary["no_forecast"],
        summary["already"],
    )
    return summary
