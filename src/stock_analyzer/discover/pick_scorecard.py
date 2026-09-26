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

No LLM calls and no price fetches — `label_candidates(only_picks=True)`
writes the outcomes first.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from statistics import mean
from typing import Any

import numpy as np
from sqlalchemy import text

from ..db.session import exec_sql, get_session

HORIZON_DAYS = 126  # trading days ≈ 6 months


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
    today = today or date.today()
    with get_session(db_path) as session:
        picks = exec_sql(
            session,
            text(
                "SELECT p.ticker, r.run_at, o.return_pct, o.spy_return_pct, o.excess_pct "
                "FROM picks p JOIN runs r ON r.id = p.run_id "
                "LEFT JOIN candidate_outcomes o ON o.run_id = p.run_id "
                "AND o.ticker = p.ticker AND o.horizon_days = :h "
                "ORDER BY r.run_at"
            ),
            params={"h": horizon},
        ).all()

    seen: set[tuple[str, str]] = set()
    graded: dict[str, list[dict[str, Any]]] = {}
    maturing: list[date] = []
    unmeasured: list[str] = []
    for ticker, run_at, ret, spy, excess in picks:
        picked = datetime.fromisoformat(str(run_at)).date()
        month = picked.strftime("%Y-%m")
        if (ticker, month) in seen:
            continue
        seen.add((ticker, month))
        if excess is None:
            if _due(picked, horizon) > today:
                maturing.append(_due(picked, horizon))
            else:
                unmeasured.append(ticker)
            continue
        graded.setdefault(month, []).append(
            {"ticker": ticker, "return_pct": ret, "spy_pct": spy, "excess_pct": excess}
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
