"""Earnings standouts: companies whose results clearly beat, that the market
rewarded, and whose analysts then raised next year's forecast.

Post-earnings drift — stocks that beat and get revised up keep beating
for weeks to months — is one of the best-documented patterns in equity
returns, and strongest outside the mega-caps. This finds those names
across the whole US market, in three steps a night apart:

  1. REPORTED: one Finnhub earnings-calendar request lists every company
     that reported, with EPS and revenue against estimates. A clear beat
     on both (EPS_BEAT, REVENUE_BEAT) by a company big enough to trade
     (MIN_QUARTER_REVENUE, SEC-listed) is stored; so is any report by a
     past pick, beat or not. Nothing else is stored.
  2. REACTION: the move from the last close before the report to the
     first close after it, minus SPY's. Two sessions, so a before-the-open
     and an after-the-close report are measured alike. A beat the market
     shrugs off (< REACTION_MIN_PCT) is dropped: sources disagree on what
     "EPS" means (Finnhub and FMP gave Costco's Sep-2026 quarter as a
     miss and a beat), and the price is the tiebreak.
  3. CONFIRMED: CONFIRM_AFTER_DAYS later, next fiscal year's consensus EPS
     against 30 days earlier (yfinance eps_trend). Raised by REVISION_MIN
     or more passes.
  4. NOT A LONE BLIP: it also beat the quarter before, and revenue is
     above the same quarter a year ago (data/earnings_history.py). One
     quarter, not a year, on purpose: a company that has just turned the
     corner is what a six-month idea is looking for. Passing all four
     makes a standout.

A standout is an idea for the discover screen and the daily email, not a
buy signal: it still has to pass the long-term (3-5 year) analysis.

No LLM calls. Idempotent: re-running a night changes nothing, and each
run looks back LOOKBACK_DAYS so a missed night is caught up.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import text

from ..db.session import exec_sql, get_session
from ..db.tables import EarningsEvent
from ..logging import get_logger

logger = get_logger(__name__)

EPS_BEAT = 0.05  # actual at least 5% above the estimate
REVENUE_BEAT = 0.01
MIN_QUARTER_REVENUE = 250e6
REACTION_MIN_PCT = 3.0  # vs SPY, over the two sessions around the report
REVISION_MIN = 0.03  # next-year EPS estimate raised 3%+
MIN_PRIOR_BEATS = 1  # the quarter before this one (earnings_history.PRIOR_QUARTERS)
CONFIRM_AFTER_DAYS = 7  # calendar days: ~5 sessions for analysts to revise
GIVE_UP_DAYS = 25  # after this, eps_trend's 30-day baseline is post-report
LOOKBACK_DAYS = 4
KEEP_DAYS = 365
# How long a standout stays in the discover universe: the drift it trades
# on plays out over weeks to a few months.
DISCOVER_DAYS = 60
# Analyst actions are kept from this long before a report: coverage and
# target moves going in are the context for the ones after it.
ANALYST_CONTEXT_DAYS = 90

Closes = Callable[[list[str], date], dict[str, pd.Series]]
# (ticker, report day, this quarter's revenue) -> data/earnings_history.fetch_track_record
TrackRecord = Callable[[str, date, float | None], dict[str, Any]]


def is_clear_beat(e: dict[str, Any] | EarningsEvent) -> bool:
    get = e.get if isinstance(e, dict) else lambda k: getattr(e, k)
    eps, eps_est = get("eps_actual"), get("eps_estimate")
    rev, rev_est = get("revenue_actual"), get("revenue_estimate")
    if eps is None or eps_est is None or rev is None or rev_est is None or rev_est <= 0:
        return False
    eps, eps_est, rev, rev_est = float(eps), float(eps_est), float(rev), float(rev_est)
    # Against |estimate| so a smaller-than-expected loss counts as a beat.
    eps_ok = eps - eps_est >= EPS_BEAT * max(abs(eps_est), 0.05)
    return eps_ok and rev >= rev_est * (1 + REVENUE_BEAT) and rev >= MIN_QUARTER_REVENUE


def record_reports(
    db_path: str, rows: list[dict[str, Any]], *, picks: set[str], listed: set[str]
) -> int:
    """Store the reports worth following; returns how many were new."""
    keep = [
        r
        for r in rows
        if r.get("eps_actual") is not None
        and (r["ticker"] in picks or (r["ticker"] in listed and is_clear_beat(r)))
    ]
    added = 0
    with get_session(db_path) as session:
        for r in keep:
            if session.get(EarningsEvent, (r["ticker"], r["date"])) is not None:
                continue
            session.add(
                EarningsEvent(
                    ticker=r["ticker"],
                    report_date=r["date"],
                    hour=r.get("hour") or "",
                    eps_estimate=r.get("eps_estimate"),
                    eps_actual=r.get("eps_actual"),
                    revenue_estimate=r.get("revenue_estimate"),
                    revenue_actual=r.get("revenue_actual"),
                )
            )
            added += 1
    return added


def _pending(db_path: str) -> list[EarningsEvent]:
    with get_session(db_path) as session:
        rows = exec_sql(
            session,
            text("SELECT ticker, report_date FROM earnings_events WHERE status = 'pending'"),
        ).all()
        out = []
        for ticker, day in rows:
            row = session.get(EarningsEvent, (ticker, day))
            if row is not None:
                session.expunge(row)
                out.append(row)
        return out


def _by_day(series: pd.Series) -> dict[date, float]:
    s = series.dropna()
    days = pd.DatetimeIndex(s.index).date  # ty: ignore[unresolved-attribute]  # delegated, invisible to checkers
    return {d: float(v) for d, v in zip(days, s.to_numpy(), strict=True)}


def reaction(close: pd.Series, spy: pd.Series, report_day: date, last_final: date) -> float | None:
    """Percent move from the last close before `report_day` to the first
    close after it, minus SPY's over the same bars. None until that close
    is final."""
    spy_at, px_at = _by_day(spy), _by_day(close)
    before = [d for d in spy_at if d < report_day]
    after = [d for d in spy_at if d > report_day]
    if not before or not after or min(after) > last_final:
        return None
    b, a = max(before), min(after)
    if b not in px_at or a not in px_at or px_at[b] <= 0:
        return None
    return (px_at[a] / px_at[b] - spy_at[a] / spy_at[b]) * 100


def measure_reactions(db_path: str, closes: Closes, *, last_final: date) -> int:
    todo = [e for e in _pending(db_path) if e.reaction_pct is None]
    if not todo:
        return 0
    start = min(date.fromisoformat(e.report_date) for e in todo) - timedelta(days=10)
    series = closes(sorted({e.ticker for e in todo} | {"SPY"}), start)
    spy = series.get("SPY")
    if spy is None:
        return 0
    measured = 0
    with get_session(db_path) as session:
        for e in todo:
            if e.ticker not in series:
                continue
            r = reaction(series[e.ticker], spy, date.fromisoformat(e.report_date), last_final)
            if r is None:
                continue
            row = session.get(EarningsEvent, (e.ticker, e.report_date))
            if row is not None:
                row.reaction_pct = r
                session.add(row)
                measured += 1
    return measured


def passes_track_record(record: dict[str, Any]) -> bool:
    """Not a lone blip: beat the quarter before too, and revenue above the
    same quarter a year ago when Yahoo has that quarter (the EPS beat
    alone decides when it doesn't)."""
    growth = record.get("revenue_yoy_pct")
    return record.get("prior_beats", 0) >= MIN_PRIOR_BEATS and (growth is None or growth > 0)


def decide(
    db_path: str,
    estimate_change: Callable[[str], float | None],
    track_record: TrackRecord,
    *,
    today: date,
) -> list[str]:
    """Settle every pending event that can be settled; returns new standouts."""
    verdicts: dict[tuple[str, str], dict[str, Any]] = {}
    for e in _pending(db_path):
        key = (e.ticker, e.report_date)
        age = (today - date.fromisoformat(e.report_date)).days
        if e.reaction_pct is None:
            if age > GIVE_UP_DAYS:  # no price: delisted, or not a stock Yahoo has
                verdicts[key] = {"status": "no"}
            continue
        if not is_clear_beat(e) or e.reaction_pct < REACTION_MIN_PCT:
            verdicts[key] = {"status": "no"}
            continue
        if age < CONFIRM_AFTER_DAYS:
            continue
        change = estimate_change(e.ticker)
        if change is None:
            if age > GIVE_UP_DAYS:
                verdicts[key] = {"status": "no"}
            continue
        if change < REVISION_MIN:
            verdicts[key] = {"status": "no", "revision_pct": change}
            continue
        # Last gate, so the two extra requests are spent on a handful.
        record = track_record(e.ticker, date.fromisoformat(e.report_date), e.revenue_actual)
        verdicts[key] = {
            "status": "standout" if passes_track_record(record) else "no",
            "revision_pct": change,
            **record,
        }

    with get_session(db_path) as session:
        for key, fields in verdicts.items():
            row = session.get(EarningsEvent, key)
            if row is None:
                continue
            for k, v in fields.items():
                setattr(row, k, v)
            row.decided_on = today.isoformat()
            session.add(row)
    return sorted(t for (t, _), v in verdicts.items() if v["status"] == "standout")


def recent_standouts(db_path: str, *, days: int, today: date | None = None) -> list[dict[str, Any]]:
    """Standouts confirmed in the last `days` days, newest first."""
    since = ((today or date.today()) - timedelta(days=days)).isoformat()
    with get_session(db_path) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT ticker, report_date, eps_estimate, eps_actual, revenue_estimate, "
                "revenue_actual, reaction_pct, revision_pct, decided_on, prior_beats, "
                "prior_quarters, revenue_yoy_pct FROM earnings_events "
                "WHERE status = 'standout' AND decided_on >= :since "
                "ORDER BY decided_on DESC, revision_pct DESC"
            ),
            params={"since": since},
        ).all()
    from ..data.analyst_actions import summarize

    out, seen = [], set()
    for t, day, eps_est, eps, rev_est, rev, react, revision, decided, beats, quarters, yoy in rows:
        if t in seen:
            continue
        seen.add(t)
        out.append(
            {
                "ticker": t,
                "report_date": day,
                "eps_surprise_pct": (eps - eps_est) / abs(eps_est) * 100 if eps_est else None,
                "revenue_surprise_pct": (rev / rev_est - 1) * 100 if rev_est else None,
                "reaction_pct": react,
                "revision_pct": revision * 100 if revision is not None else None,
                "decided_on": decided,
                "prior_beats": beats,
                "prior_quarters": quarters,
                "revenue_yoy_pct": yoy,
                # What analysts did from the report on (analyst_actions).
                "analysts": summarize(db_path, t, since=date.fromisoformat(day)),
            }
        )
    return out


def track_analysts(
    db_path: str, fetch: Callable[[str], list[dict[str, Any]]], *, today: date
) -> int:
    """Store analyst actions for every event still being followed (pending,
    or a standout of the last DISCOVER_DAYS), from ANALYST_CONTEXT_DAYS
    before its report on. One request per stock; returns rows added."""
    from ..data.analyst_actions import record_actions

    since = (today - timedelta(days=DISCOVER_DAYS)).isoformat()
    with get_session(db_path) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT ticker, MIN(report_date) FROM earnings_events "
                "WHERE status IN ('pending', 'standout') AND report_date >= :s GROUP BY ticker"
            ),
            params={"s": since},
        ).all()
    added = 0
    for ticker, first in rows:
        start = date.fromisoformat(first) - timedelta(days=ANALYST_CONTEXT_DAYS)
        added += record_actions(db_path, ticker, fetch(ticker), since=start)
    return added


def prune(db_path: str, *, today: date) -> int:
    cutoff = (today - timedelta(days=KEEP_DAYS)).isoformat()
    with get_session(db_path) as session:
        result = exec_sql(
            session,
            text("DELETE FROM earnings_events WHERE report_date < :c"),
            params={"c": cutoff},
        )
        return int(getattr(result, "rowcount", 0) or 0)


def watch(
    db_path: str,
    *,
    today: date,
    last_final: date,
    calendar: Callable[[date, date], list[dict[str, Any]]],
    closes: Closes,
    estimate_change: Callable[[str], float | None],
    track_record: TrackRecord,
    picks: set[str],
    listed: set[str],
    analyst_actions: Callable[[str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """One night: record new reports, measure reactions, settle what can be,
    and store what analysts have done on the stocks still followed."""
    rows = calendar(today - timedelta(days=LOOKBACK_DAYS), today)
    summary: dict[str, Any] = {
        "reported": sum(r.get("eps_actual") is not None for r in rows),
        "recorded": record_reports(db_path, rows, picks=picks, listed=listed),
        "reactions": measure_reactions(db_path, closes, last_final=last_final),
    }
    summary["standouts"] = decide(db_path, estimate_change, track_record, today=today)
    if analyst_actions is not None:
        summary["analyst_actions"] = track_analysts(db_path, analyst_actions, today=today)
    summary["pruned"] = prune(db_path, today=today)
    logger.info(
        "Earnings watch: %d reported, %d recorded, %d reactions measured, "
        "%d analyst actions stored, standouts: %s",
        summary["reported"],
        summary["recorded"],
        summary["reactions"],
        summary.get("analyst_actions", 0),
        ", ".join(summary["standouts"]) or "none",
    )
    return summary
