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

The earnings standouts and insider-buying clusters the daily email showed
get the same card (`suggestion_scorecard`), from the first close after
the email.

Two checks ride along with the picks:

  - `vs_screen`: per cohort, the picks against what the screen alone would
    have chosen from the same runs — its top SCREEN_TOP by score, and every
    name that passed it. If the picks keep trailing the screen's top, the
    model step is not earning its cost (May 2026 at 63 days: picks -10.1%
    median vs SPY, screen pool -11.1%; Sept so far: picks +1.8%, screen
    top 10 +4.0%).
  - `vs_screen` also carries the top SCREEN_TOP by the evidence score
    (discover/evidence.py): a ranking from only the signals that held up
    when tested, which the model never sees. Recorded from 2026-10-07 on.
  - `calibration`: graded picks split by the model's conviction and by how
    many providers agreed, shown once CALIBRATION_MIN_PICKS are graded —
    whether either one deserves to size a position.
  - `analyst`: the Analyst's own 1-10 score ("Score: N" in each stored
    scorecard) for every name it read, ~25 a run against ~5 picks, so the
    verdict on the model's judgement arrives five times sooner. In May 2026
    it pointed the wrong way (63 days: scored 7-10 -4.7% median vs SPY,
    1-5 +2.0%; n=26, noise-level).

No LLM calls. Picks and screen survivors read stored outcomes
(`label_candidates(only_passed=True)` writes them first); standouts read
closes from the bar store.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from statistics import mean
from typing import Any

import numpy as np
import polars as pl
from sqlalchemy import text

from ..data import frames
from ..db.session import exec_sql, get_session

HORIZON_DAYS = 126  # trading days ≈ 6 months
SCREEN_TOP = 10  # the screen's own choice: its top names by score per run
CALIBRATION_MIN_PICKS = 50  # below this a conviction split is noise
HIGH_CONVICTION = 7  # conviction runs 1-10; the picks so far sit at 5-8
# The Analyst's score bands, as validate-screen groups the ranker's conviction.
ANALYST_BANDS: tuple[tuple[str, int, int], ...] = (
    ("Scored 8-10", 8, 10),
    ("Scored 6-7", 6, 7),
    ("Scored 1-5", 1, 5),
)
_ANALYST_SCORE = re.compile(r"^\s*Score:\s*(\d+)", re.MULTILINE)

# Which pool discover picked from, from each date on. Picks from different
# pools are different experiments, so the card never blends them: cohorts
# and the overall row are split by pool once picks span more than one.
# Add a row whenever the universe or its rules change materially.
UNIVERSE_ERAS: tuple[tuple[date, str], ...] = (
    (date.min, "S&P 500"),
    (date(2026, 9, 27), "US >= $2B quality"),
)


def universe_on(day: date, eras: tuple[tuple[date, str], ...] = UNIVERSE_ERAS) -> str:
    return [label for start, label in eras if start <= day][-1]


Closes = Callable[[list[str], date], dict[str, pl.DataFrame]]


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
    db_path: str,
    *,
    horizon: int = HORIZON_DAYS,
    today: date | None = None,
    eras: tuple[tuple[date, str], ...] = UNIVERSE_ERAS,
) -> dict[str, Any]:
    """{"cohorts": [...], "overall": {...} | None, "maturing": n,
    "next_due": date | None, "unmeasured": [ticker, ...], "vs_screen": [...],
    "calibration": {...}}. Due dates count weekdays, not exchange holidays,
    so they are approximate."""
    today = today or date.today()
    with get_session(db_path) as session:
        picks = exec_sql(
            session,
            text(
                "SELECT p.ticker, r.run_at, o.return_pct, o.spy_return_pct, "
                "p.conviction, p.agreement_ratio "
                "FROM picks p JOIN runs r ON r.id = p.run_id "
                "LEFT JOIN candidate_outcomes o ON o.run_id = p.run_id "
                "AND o.ticker = p.ticker AND o.horizon_days = :h "
                "ORDER BY r.run_at"
            ),
            params={"h": horizon},
        ).all()
        # Everything that passed the screen in a run that made picks, best
        # score first within each run.
        screened = exec_sql(
            session,
            text(
                "SELECT c.run_id, c.ticker, r.run_at, c.score, o.return_pct, o.spy_return_pct, "
                "c.score_breakdown "
                "FROM candidates c JOIN runs r ON r.id = c.run_id "
                "LEFT JOIN candidate_outcomes o ON o.run_id = c.run_id "
                "AND o.ticker = c.ticker AND o.horizon_days = :h "
                "WHERE c.passed_filter = 1 "
                "AND c.run_id IN (SELECT DISTINCT run_id FROM picks) "
                "ORDER BY r.run_at, c.run_id, c.score IS NULL, c.score DESC"
            ),
            params={"h": horizon},
        ).all()
        analysed = exec_sql(
            session,
            text(
                "SELECT s.ticker, r.run_at, s.analyst_text, o.return_pct, o.spy_return_pct "
                "FROM scorecards s JOIN runs r ON r.id = s.run_id "
                "LEFT JOIN candidate_outcomes o ON o.run_id = s.run_id "
                "AND o.ticker = s.ticker AND o.horizon_days = :h "
                "ORDER BY r.run_at"
            ),
            params={"h": horizon},
        ).all()
    entries = [
        (t, datetime.fromisoformat(str(run_at)).date(), ret, spy, conv, agree)
        for t, run_at, ret, spy, conv, agree in picks
    ]
    card = _summarize([e[:4] for e in entries], horizon=horizon, today=today, eras=eras)
    pool: list[Entry] = []
    top: list[Entry] = []
    taken: dict[int, int] = {}
    by_evidence: dict[int, list[tuple[float, Entry]]] = {}
    for run_id, t, run_at, score, ret, spy, breakdown in screened:
        entry = (t, datetime.fromisoformat(str(run_at)).date(), ret, spy)
        pool.append(entry)
        if score is not None and taken.get(run_id, 0) < SCREEN_TOP:
            taken[run_id] = taken.get(run_id, 0) + 1
            top.append(entry)
        ev = evidence_of(breakdown)
        if ev is not None:
            by_evidence.setdefault(run_id, []).append((ev, entry))
    evidence = [
        e
        for ranked in by_evidence.values()
        for _, e in sorted(ranked, key=lambda x: -x[0])[:SCREEN_TOP]
    ]
    evidence.sort(key=lambda e: e[1])
    card["vs_screen"] = _vs_screen(
        [e[:4] for e in entries], top, pool, evidence=evidence, eras=eras
    )
    card["calibration"] = _calibration(entries)
    card["analyst"] = _analyst_bands(
        [
            (t, datetime.fromisoformat(str(run_at)).date(), ret, spy, analyst_score(body))
            for t, run_at, body, ret, spy in analysed
        ]
    )
    return card


def evidence_of(breakdown: Any) -> float | None:
    """A candidate's evidence score (discover/evidence.py) from its stored
    score_breakdown, or None for runs before it was recorded."""
    if isinstance(breakdown, str):
        try:
            breakdown = json.loads(breakdown)
        except ValueError:
            return None
    ev = (breakdown or {}).get("evidence") if isinstance(breakdown, dict) else None
    score = ev.get("score") if isinstance(ev, dict) else None
    return float(score) if isinstance(score, (int, float)) else None


def analyst_score(body: str | None) -> int | None:
    """The 1-10 "Score: N" line of an Analyst scorecard, or None."""
    m = _ANALYST_SCORE.search(body or "")
    score = int(m.group(1)) if m else None
    return score if score is not None and 1 <= score <= 10 else None


def _analyst_bands(entries: list[tuple[Any, ...]]) -> dict[str, Any]:
    """{"graded": n, "rows": [{"group", **_cohort}]}: every name the Analyst
    scored, graded, by score band (one decision per ticker per month)."""
    graded = [
        e
        for e in _first_per_month([e for e in entries if e[4] is not None])
        if e[2] is not None and e[3] is not None
    ]
    rows = []
    for name, lo, hi in ANALYST_BANDS:
        band = [
            {"return_pct": e[2], "spy_pct": e[3], "excess_pct": e[2] - e[3]}
            for e in graded
            if lo <= e[4] <= hi
        ]
        if band:
            rows.append({"group": name, **_cohort(band, name)})
    return {"graded": len(graded), "rows": rows}


Entry = tuple[str, date, float | None, float | None]


def _first_per_month(entries: list[Any]) -> list[Any]:
    """One decision per ticker per month: the earliest entry is kept."""
    seen: set[tuple[str, str]] = set()
    out = []
    for e in entries:
        key = (e[0], e[1].strftime("%Y-%m"))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def _graded_by_cohort(
    entries: list[Entry], eras: tuple[tuple[date, str], ...] | None
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """{(month, universe): graded rows}, one decision per ticker per month."""
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for ticker, day, ret, spy in _first_per_month(entries):
        if ret is None or spy is None:
            continue
        pool = universe_on(day, eras) if eras else ""
        out.setdefault((day.strftime("%Y-%m"), pool), []).append(
            {"ticker": ticker, "return_pct": ret, "spy_pct": spy, "excess_pct": ret - spy}
        )
    return out


def _vs_screen(
    picks: list[Entry],
    top: list[Entry],
    pool: list[Entry],
    *,
    evidence: Sequence[Entry] = (),
    eras: tuple[tuple[date, str], ...] | None,
) -> list[dict[str, Any]]:
    """Per graded cohort of picks: {"cohort", "picks", "top", "evidence",
    "pool"}, each a `_cohort` dict (None when that group has nothing
    graded), plus an "All" row per universe once there is more than one
    cohort."""
    kinds = ("picks", "top", "evidence", "pool")
    groups = [_graded_by_cohort(list(g), eras) for g in (picks, top, evidence, pool)]
    keys = sorted(groups[0])
    pools = list(dict.fromkeys(p for _, p in keys))
    split = len(pools) > 1

    def label(month: str, pool_name: str) -> str:
        name = datetime.strptime(month, "%Y-%m").strftime("%b %Y")
        return f"{name} · {pool_name}" if split else name

    def row(name: str, wanted: list[tuple[str, str]]) -> dict[str, Any]:
        out: dict[str, Any] = {"cohort": name}
        for kind, g in zip(kinds, groups, strict=True):
            rows = [r for k in wanted for r in g.get(k, [])]
            out[kind] = _cohort(rows, name) if rows else None
        return out

    out = [row(label(*k), [k]) for k in keys]
    if len(keys) > 1:
        for pool_name in pools:
            out.append(
                row(
                    f"All · {pool_name}" if split else "All", [k for k in keys if k[1] == pool_name]
                )
            )
    return out


def _calibration(entries: list[tuple[Any, ...]]) -> dict[str, Any]:
    """{"graded": n, "rows": [{"group", **_cohort}]}: graded picks split by
    conviction and by provider agreement (one decision per ticker per month)."""
    graded = [e for e in _first_per_month(entries) if e[2] is not None and e[3] is not None]

    def group(name: str, keep: Callable[[Any, Any], bool]) -> dict[str, Any] | None:
        rows = [
            {"return_pct": e[2], "spy_pct": e[3], "excess_pct": e[2] - e[3]}
            for e in graded
            if keep(e[4], e[5])
        ]
        return {"group": name, **_cohort(rows, name)} if rows else None

    splits = [
        group(
            f"Conviction {HIGH_CONVICTION}+", lambda c, _: c is not None and c >= HIGH_CONVICTION
        ),
        group(
            f"Conviction under {HIGH_CONVICTION}",
            lambda c, _: c is not None and c < HIGH_CONVICTION,
        ),
        group("All providers agreed", lambda _, a: a is not None and a >= 1),
        group("Split vote", lambda _, a: a is not None and a < 1),
    ]
    return {"graded": len(graded), "rows": [r for r in splits if r]}


def suggestion_scorecard(
    db_path: str,
    closes: Closes,
    *,
    action: str = "STANDOUT",
    horizon: int = HORIZON_DAYS,
    today: date | None = None,
) -> dict[str, Any]:
    """The same card for ideas the daily email showed, by their `suggestions`
    action (STANDOUT: earnings standouts; INSIDER_BUYS: insider-buying
    clusters): entry at the first close after the email, exit `horizon`
    sessions later, against SPY over the same bars. Prices come from
    `closes` (the bar store), not a stored label."""
    today = today or date.today()
    with get_session(db_path) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT ticker, MIN(suggested_on) FROM suggestions WHERE action = :a "
                "GROUP BY ticker, substr(suggested_on, 1, 7) ORDER BY 2"
            ),
            params={"a": action},
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
    close: pl.DataFrame | None, spy: pl.DataFrame | None, day: date, horizon: int
) -> tuple[float, float] | None:
    """(return %, SPY %) from the first close after `day` to `horizon`
    sessions later; None while the window is open or a price is missing."""
    if close is None or spy is None:
        return None
    sessions = sorted(frames.by_day(spy).items())
    px = frames.by_day(close)
    start = next((i for i, (d, _) in enumerate(sessions) if d > day), None)
    if start is None or start + horizon >= len(sessions):
        return None
    (d0, s0), (d1, s1) = sessions[start], sessions[start + horizon]
    if not px.get(d0) or not px.get(d1):
        return None
    return (px[d1] / px[d0] - 1) * 100, (s1 / s0 - 1) * 100


def _summarize(
    entries: list[tuple[str, date, float | None, float | None]],
    *,
    horizon: int,
    today: date,
    eras: tuple[tuple[date, str], ...] | None = None,
) -> dict[str, Any]:
    """Group (ticker, day, return %, SPY %) by month, one decision per
    ticker per month; a missing return is maturing or unmeasured. With
    `eras`, also by universe, labelled once graded picks span two."""
    graded: dict[tuple[str, str], list[dict[str, Any]]] = {}
    maturing: list[date] = []
    unmeasured: list[str] = []
    for ticker, day, ret, spy in _first_per_month(entries):
        month = day.strftime("%Y-%m")
        if ret is None or spy is None:
            if _due(day, horizon) > today:
                maturing.append(_due(day, horizon))
            else:
                unmeasured.append(ticker)
            continue
        pool = universe_on(day, eras) if eras else ""
        graded.setdefault((month, pool), []).append(
            {"ticker": ticker, "return_pct": ret, "spy_pct": spy, "excess_pct": ret - spy}
        )

    pools = list(dict.fromkeys(p for _, p in sorted(graded)))
    split = len(pools) > 1

    def label(month: str, pool: str) -> str:
        name = datetime.strptime(month, "%Y-%m").strftime("%b %Y")
        return f"{name} · {pool}" if split else name

    cohorts = [_cohort(rows, label(m, p)) for (m, p), rows in sorted(graded.items())]
    if split:
        overall = [
            _cohort(
                [r for (_, p), rows in graded.items() if p == pool for r in rows], f"All · {pool}"
            )
            for pool in pools
        ]
    else:
        all_rows = [r for rows in graded.values() for r in rows]
        overall = [_cohort(all_rows, "All")] if len(cohorts) > 1 else []
    return {
        "horizon": horizon,
        "cohorts": cohorts,
        "overall": overall[0] if len(overall) == 1 else None,
        "overall_by_universe": overall if split else [],
        "maturing": len(maturing),
        "next_due": min(maturing, default=None),
        "unmeasured": unmeasured,
    }
