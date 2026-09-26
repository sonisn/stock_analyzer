"""Six months after each pick: did the discover picks beat SPY?

Reads the realized 126-trading-day outcomes (`candidate_outcomes`, the
same label the model trains on: entry at the first close after the run,
exit 126 bars later, minus SPY over the same bars) for every row in
`picks`, grouped by the month the pick was made — one cohort per batch.

A ticker picked on several runs in the same month is one decision, not
several: consecutive daily runs re-pick the same names, and counting each
would let one stock's result stand in for a dozen. The earliest pick of
the month is kept.

A pick whose window has closed with no outcome (no price — usually a
delisting, the worst result a pick can have) is counted as unmeasured,
not dropped and not left "maturing" forever.

The earnings standouts the daily email showed get the same card
(`standout_scorecard`), from the first close after the email.

No LLM calls. Picks read stored outcomes (`label_candidates(only_picks=True)`
writes them first); standouts read closes from the bar store.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from ..db.session import exec_sql, get_session

HORIZON_DAYS = 126  # trading days ≈ 6 months

Closes = Callable[[list[str], date], dict[str, pd.Series]]


def _due(picked: date, horizon: int) -> date:
    """When a pick's window closes (a few days' slack for holidays): entry
    is the next session's close, exit `horizon` sessions after that."""
    due = np.busday_offset(np.datetime64(picked), horizon + 1, roll="forward")
    return due.astype(date) + timedelta(days=3)


def _cohort(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    return {
        "cohort": label,
        "picks": len(rows),
        "return_pct": mean(r["return_pct"] for r in rows),
        "spy_pct": mean(r["spy_pct"] for r in rows),
        "excess_pct": mean(r["excess_pct"] for r in rows),
        "beat_spy": sum(r["excess_pct"] > 0 for r in rows) / len(rows),
    }


def pick_scorecard(
    db_path: str, *, horizon: int = HORIZON_DAYS, today: date | None = None
) -> dict[str, Any]:
    """{"cohorts": [...], "overall": {...} | None, "maturing": n,
    "next_due": date | None, "unmeasured": [ticker, ...]}. Due dates
    count weekdays, not exchange holidays, so they are approximate."""
    with get_session(db_path) as session:
        picks = exec_sql(
            session,
            text(
                "SELECT p.ticker, r.run_at, o.return_pct, o.spy_return_pct "
                "FROM picks p JOIN runs r ON r.id = p.run_id "
                "LEFT JOIN candidate_outcomes o ON o.run_id = p.run_id "
                "AND o.ticker = p.ticker AND o.horizon_days = :h "
                "ORDER BY r.run_at"
            ),
            params={"h": horizon},
        ).all()
    entries = [
        (t, datetime.fromisoformat(str(run_at)).date(), ret, spy) for t, run_at, ret, spy in picks
    ]
    return _summarize(entries, horizon=horizon, today=today or date.today())


def standout_scorecard(
    db_path: str, closes: Closes, *, horizon: int = HORIZON_DAYS, today: date | None = None
) -> dict[str, Any]:
    """The same card for the earnings standouts the daily email showed (the
    STANDOUT rows of `suggestions`): entry at the first close after the
    email, exit `horizon` sessions later, against SPY over the same bars.
    Prices come from `closes` (the bar store), not a stored label."""
    today = today or date.today()
    with get_session(db_path) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT ticker, MIN(suggested_on) FROM suggestions WHERE action = 'STANDOUT' "
                "GROUP BY ticker, substr(suggested_on, 1, 7) ORDER BY 2"
            ),
        ).all()
    if not rows:
        return _summarize([], horizon=horizon, today=today)
    shown = [(t, date.fromisoformat(d)) for t, d in rows]
    series = closes(sorted({t for t, _ in shown} | {"SPY"}), min(d for _, d in shown))
    spy = series.get("SPY")
    entries = []
    for t, day in shown:
        outcome = _outcome(series.get(t), spy, day, horizon)
        entries.append((t, day, *(outcome or (None, None))))
    return _summarize(entries, horizon=horizon, today=today)


def _outcome(
    close: pd.Series | None, spy: pd.Series | None, day: date, horizon: int
) -> tuple[float, float] | None:
    """(return %, SPY %) from the first close after `day` to `horizon`
    sessions later; None while the window is open or a price is missing."""
    if close is None or spy is None:
        return None
    sessions = sorted(_by_day(spy).items())
    px = _by_day(close)
    start = next((i for i, (d, _) in enumerate(sessions) if d > day), None)
    if start is None or start + horizon >= len(sessions):
        return None
    (d0, s0), (d1, s1) = sessions[start], sessions[start + horizon]
    if not px.get(d0) or not px.get(d1):
        return None
    return (px[d1] / px[d0] - 1) * 100, (s1 / s0 - 1) * 100


def _by_day(series: pd.Series) -> dict[date, float]:
    s = series.dropna()
    days = pd.DatetimeIndex(s.index).date  # ty: ignore[unresolved-attribute]  # delegated, invisible to checkers
    return {d: float(v) for d, v in zip(days, s.to_numpy(), strict=True)}


def _summarize(
    entries: list[tuple[str, date, float | None, float | None]], *, horizon: int, today: date
) -> dict[str, Any]:
    """Group (ticker, day, return %, SPY %) by month, one decision per
    ticker per month; a missing return is maturing or unmeasured."""
    seen: set[tuple[str, str]] = set()
    graded: dict[str, list[dict[str, Any]]] = {}
    maturing: list[date] = []
    unmeasured: list[str] = []
    for ticker, day, ret, spy in entries:
        month = day.strftime("%Y-%m")
        if (ticker, month) in seen:
            continue
        seen.add((ticker, month))
        if ret is None or spy is None:
            if _due(day, horizon) > today:
                maturing.append(_due(day, horizon))
            else:
                unmeasured.append(ticker)
            continue
        graded.setdefault(month, []).append(
            {"ticker": ticker, "return_pct": ret, "spy_pct": spy, "excess_pct": ret - spy}
        )

    cohorts = [
        _cohort(rows, datetime.strptime(m, "%Y-%m").strftime("%b %Y"))
        for m, rows in sorted(graded.items())
    ]
    all_rows = [r for rows in graded.values() for r in rows]
    return {
        "horizon": horizon,
        "cohorts": cohorts,
        "overall": _cohort(all_rows, "All") if len(cohorts) > 1 else None,
        "maturing": len(maturing),
        "next_due": min(maturing, default=None),
        "unmeasured": unmeasured,
    }
