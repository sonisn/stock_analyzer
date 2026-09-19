"""Brokerage activity history, stored once and kept up to date.

Every daily email and rebalance used to download three years of SnapTrade
activity (twice per daily run), and anything older than that window
dropped out — an old purchase vanished from the tax lots, which skewed the
short/long-term split and first-in-first-out matching of sales. Now each
activity is stored once in `brokerage_activities` (compact: only the
fields the app reads), and a sync fetches just what is new since the last
stored date per account, with a small overlap for late-posting entries.

The first sync reaches back `FIRST_SYNC_YEARS`; the brokerage returns
whatever history it has. The history is small (a few hundred rows a year)
and is never pruned. If SnapTrade is unreachable, what is stored is used.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from typing import Any

from sqlmodel import func, select

from ..db.session import get_session
from ..db.tables import BrokerageActivity
from ..logging import get_logger

logger = get_logger(__name__)

FIRST_SYNC_YEARS = 10
OVERLAP_DAYS = 10

# One sync per database per process: the daily email reads activities for
# tax lots, cash flows and dividends in the same run.
_synced: set[str] = set()


def compact(activity: dict[str, Any]) -> dict[str, Any]:
    """The fields the app reads, in the shape its parsers expect."""
    from .brokerage import _extract_ticker

    ticker = _extract_ticker(activity)
    opt = activity.get("option_symbol")
    if isinstance(opt, dict):
        opt = {"id": opt.get("id"), "ticker": opt.get("ticker")}
    elif opt:
        opt = {"id": str(opt), "ticker": str(opt)}
    out = {
        k: activity.get(k)
        for k in ("id", "type", "trade_date", "settlement_date", "units", "price", "amount", "fee")
    }
    out["symbol"] = {"symbol": ticker} if ticker else None
    out["option_symbol"] = opt or None
    return out


def _key(row: dict[str, Any], account: str) -> str:
    if row.get("id"):
        return str(row["id"])
    raw = json.dumps([account, row], sort_keys=True, default=str)
    return "h:" + hashlib.sha1(raw.encode()).hexdigest()


def _day(row: dict[str, Any]) -> str | None:
    from .transactions import _coerce_date

    d = _coerce_date(row.get("trade_date") or row.get("settlement_date"))
    return d.isoformat() if d else None


def store_activities(db_path: str, by_account: dict[str, list[dict[str, Any]]]) -> int:
    """Insert activities not stored yet; returns how many were added."""
    added = 0
    with get_session(db_path) as session:
        for account, rows in by_account.items():
            for raw in rows:
                row = compact(raw)
                day = _day(row)
                if day is None:
                    continue
                key = _key(row, account)
                if session.get(BrokerageActivity, key) is not None:
                    continue
                session.add(
                    BrokerageActivity(
                        id=key,
                        account=account,
                        trade_date=day,
                        type=(row.get("type") or "").upper(),
                        data=json.dumps(row, default=str, separators=(",", ":")),
                    )
                )
                added += 1
    return added


def sync_activities(db_path: str, *, today: date | None = None) -> int:
    """Fetch activity newer than what is stored (per account) and store it."""
    from .transactions import _live_activities_by_account

    today = today or date.today()
    with get_session(db_path) as session:
        latest = dict(
            session.exec(
                select(BrokerageActivity.account, func.max(BrokerageActivity.trade_date)).group_by(
                    BrokerageActivity.account
                )
            ).all()
        )
    # Each stored account resumes a little before its latest activity; an
    # account with nothing stored (first sync, or newly connected) gets the
    # full first-sync horizon.
    start = today - timedelta(days=FIRST_SYNC_YEARS * 365)
    since = {a: date.fromisoformat(d) - timedelta(days=OVERLAP_DAYS) for a, d in latest.items()}
    fetched = _live_activities_by_account(start, today, since=since)
    added = store_activities(db_path, fetched)
    logger.info(
        "Activity history: %d new activit%s stored (fetched since %s)",
        added,
        "y" if added == 1 else "ies",
        start,
    )
    return added


def ledger_activities(
    db_path: str, *, start: date | None = None
) -> dict[str, list[dict[str, Any]]]:
    """{account: activities since `start` (None = all)}, oldest first,
    after bringing the stored history up to date (once per process)."""
    if db_path not in _synced:
        try:
            sync_activities(db_path)
        except Exception as e:  # noqa: BLE001 — stored history still serves
            logger.warning("Activity sync failed (%s) — using stored history", e)
        _synced.add(db_path)
    query = select(BrokerageActivity).order_by(BrokerageActivity.trade_date)
    if start is not None:
        query = query.where(BrokerageActivity.trade_date >= start.isoformat())
    out: dict[str, list[dict[str, Any]]] = {}
    with get_session(db_path) as session:
        for row in session.exec(query):
            out.setdefault(row.account, []).append(json.loads(row.data))
    return out
