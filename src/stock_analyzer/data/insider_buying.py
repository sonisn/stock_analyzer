"""Clusters of insider buying: several executives or directors buying their
own company's stock on the open market within a few months.

Tested before it was built (S&P 500 current members, 2015-2026, 133
month-ends, SEC's quarterly insider data sets, point in time by filing
date): stocks where 2+ insiders had each bought $10,000+ in the last 90
days beat SPY by 9.5% on average over the next 12 months (median +3.0%),
against 3.2% (median -1.4%) with no buying; the rank link was positive in
both halves of the decade. Its t-statistic after correcting for
overlapping windows is 2.07 at six months but 1.85 at twelve, so it is
treated as promising, not proven:
a cluster makes a stock eligible for discover and shows in the daily
email, with no score bonus, and the six-month scorecard grades it live.
Share-count changes (buybacks vs issuance) were tested the same way and
failed (t -0.7 / -1.2, the heaviest issuers did best), so they are not
used.

Source: Finnhub's insider transactions, one request per company, paced
at Finnhub's free-tier rate — the S&P 500 plus tracked stocks is about
ten minutes inside the nightly job. Finnhub lists every Section 16
insider (officers, directors, 10% holders), a slightly wider net than the
study's officers and directors. No LLM.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from sqlalchemy import text

from ..db.session import exec_sql, get_session
from ..db.tables import InsiderBuy
from ..logging import get_logger
from . import finnhub as finnhub_data

logger = get_logger(__name__)

WINDOW_DAYS = 90
MIN_BUYERS = 2
# An insider counts once their purchases in the window reach this. Plan and
# reinvestment buys are coded as purchases too: TSM's staff buying 40-54
# shares each at one price on one day, SPG's directors 2-13 shares. With
# the floor the history test got slightly stronger (6-month t 2.07 vs
# 1.91; 12-month +9.5% vs +3.2%, against +8.9% without it).
MIN_BUY_USD = 10_000.0
# Plan buys have a signature no conviction purchase has: many insiders at
# one price on one day. Rows sharing (day, price) with this many others
# are left out whatever they add up to — TSM's monthly staff purchases
# pass the dollar floor over 90 days.
PLAN_BUYERS = 5
INSIDER_KEEP_DAYS = 400
# Finnhub has no officer/director field, and lists 10%-holder funds as
# insiders (TPL's "Horizon Kinetics Asset Management Llc"). The study
# counted officers and directors only, so entities are left out by name.
_ENTITY = re.compile(
    r"\b(llc|lp|l\.p\.|inc|corp|corporation|ltd|limited|management|fund|funds|trust|"
    r"partners|capital|holdings|advisors|investments|foundation)\b",
    re.IGNORECASE,
)


def is_person(name: str) -> bool:
    return not _ENTITY.search(name)


def fetch_insider_buys(
    ticker: str, *, today: date, days: int = WINDOW_DAYS
) -> list[dict[str, Any]]:
    """Open-market purchases filed in the last `days`; [] when none or on error."""
    client = finnhub_data._client()
    if client is None:
        return []
    raw = finnhub_data._safe_call(
        "insider_transactions",
        ticker,
        client.stock_insider_transactions,
        ticker,
        (today - timedelta(days=days)).isoformat(),
        today.isoformat(),
    )
    out = []
    for tx in (raw or {}).get("data") or []:
        if (
            tx.get("transactionCode") != "P"
            or tx.get("isDerivative")
            or (tx.get("change") or 0) <= 0
            # Form 4s only: a foreign issuer's rows come from its home
            # exchange (TSM's staff share plan appeared as 31 "insiders").
            or str(tx.get("source") or "sec").lower() != "sec"
        ):
            continue
        out.append(
            {
                "ticker": ticker,
                "filing_id": str(tx.get("id") or ""),
                "name": str(tx.get("name") or "").strip().title(),
                "filed": str(tx.get("filingDate") or "")[:10],
                "traded": str(tx.get("transactionDate") or "")[:10],
                "shares": float(tx.get("change") or 0),
                "price": float(tx["transactionPrice"]) if tx.get("transactionPrice") else None,
            }
        )
    return [b for b in out if b["name"] and b["filed"] and is_person(str(b["name"]))]


def record_buys(db_path: str, rows: list[dict[str, Any]]) -> int:
    added = 0
    with get_session(db_path) as session:
        for r in rows:
            if session.get(InsiderBuy, (r["ticker"], r["filing_id"], r["name"])) is not None:
                continue
            session.add(InsiderBuy(**r))
            added += 1
    return added


def clusters(
    db_path: str,
    *,
    today: date,
    days: int = WINDOW_DAYS,
    min_buyers: int = MIN_BUYERS,
    min_usd: float = MIN_BUY_USD,
) -> list[dict[str, Any]]:
    """Stocks where `min_buyers`+ different insiders each bought at least
    `min_usd` within the last `days`, by filing date: {ticker, buyers,
    value_usd, names, formed_on}. `formed_on` is the filing that made it a
    cluster, so a caller can tell a new cluster from one that has stood
    for weeks. Newest first."""
    since = (today - timedelta(days=days)).isoformat()
    with get_session(db_path) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT ticker, name, filed, shares, price FROM insider_buys "
                "WHERE filed >= :s AND filed <= :t ORDER BY filed"
            ),
            params={"s": since, "t": today.isoformat()},
        ).all()
    same_print: dict[tuple[str, str, float], set[str]] = {}
    for ticker, name, filed, _, price in rows:
        same_print.setdefault((ticker, filed, round(price or 0.0, 2)), set()).add(name)
    by_ticker: dict[str, list[tuple[str, str, float]]] = {}
    for ticker, name, filed, shares, price in rows:
        if len(same_print[(ticker, filed, round(price or 0.0, 2))]) >= PLAN_BUYERS:
            continue  # a plan purchase, not a decision
        by_ticker.setdefault(ticker, []).append((name, filed, shares * (price or 0.0)))
    out = []
    for ticker, buys in by_ticker.items():
        spent: dict[str, float] = {}
        qualified: list[str] = []
        formed = None
        for name, filed, value in buys:
            spent[name] = spent.get(name, 0.0) + value
            if name not in qualified and spent[name] >= min_usd:
                qualified.append(name)
                if len(qualified) == min_buyers:
                    formed = filed
        if formed is None:
            continue
        out.append(
            {
                "ticker": ticker,
                "buyers": len(qualified),
                "value_usd": sum(spent[n] for n in qualified),
                "names": qualified,
                "formed_on": formed,
            }
        )
    return sorted(out, key=lambda c: (c["formed_on"], c["buyers"]), reverse=True)


def watch(
    db_path: str,
    *,
    today: date,
    tickers: list[str],
    fetch: Callable[..., list[dict[str, Any]]] = fetch_insider_buys,
) -> dict[str, Any]:
    """One night: fetch every watched company's recent purchases, store the
    new ones, prune old ones. Returns {"checked", "added", "clusters"}."""
    added = 0
    for i, ticker in enumerate(tickers, 1):
        added += record_buys(db_path, fetch(ticker, today=today))
        if i % 100 == 0:
            logger.info("Insider buying: %d/%d companies checked", i, len(tickers))
    cutoff = (today - timedelta(days=INSIDER_KEEP_DAYS)).isoformat()
    with get_session(db_path) as session:
        exec_sql(session, text("DELETE FROM insider_buys WHERE filed < :c"), params={"c": cutoff})
    found = clusters(db_path, today=today)
    logger.info(
        "Insider buying: %d companies checked, %d new purchases, %d clusters: %s",
        len(tickers),
        added,
        len(found),
        ", ".join(f"{c['ticker']}({c['buyers']})" for c in found[:10]) or "none",
    )
    return {"checked": len(tickers), "added": added, "clusters": found}
