"""Grade past catalyst calls against what the stock actually did.

For every stored pick catalyst whose expected date has passed, measure the
stock's move vs SPY from the last close before the date to the close
~5 trading days after it (the reaction window). A directional call
("positive"/"negative") is a hit when the excess move has that sign; an
"uncertain" call is a hit when the stock really moved (|excess| >= 3%).

This grades whether the calls carried price information. It cannot tell
whether the event itself happened on schedule — a slipped date simply
shows up as a quiet window.

The summary is appended to the Ranker's calibration block so it can
discount catalyst reasoning that hasn't been paying off.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from ..db.session import get_session
from ..logging import get_logger
from .track_record import _close_on_or_before, _fetch_history

logger = get_logger(__name__)

REACTION_DAYS = 7  # calendar days after the event ≈ 5 trading days
MOVER_THRESHOLD_PCT = 3.0
MIN_GRADED_FOR_PROMPT = 5


@dataclass(frozen=True)
class GradedCatalyst:
    ticker: str
    expected_date: str
    direction: str
    impact: str
    excess_move_pct: float

    @property
    def hit(self) -> bool:
        if self.direction == "positive":
            return self.excess_move_pct > 0
        if self.direction == "negative":
            return self.excess_move_pct < 0
        return abs(self.excess_move_pct) >= MOVER_THRESHOLD_PCT


@dataclass
class CatalystReport:
    graded: list[GradedCatalyst] = field(default_factory=list)

    def _subset(self, *, directional: bool) -> list[GradedCatalyst]:
        return [g for g in self.graded if (g.direction != "uncertain") == directional]

    @property
    def directional_hit_rate(self) -> float | None:
        d = self._subset(directional=True)
        return sum(g.hit for g in d) / len(d) if d else None

    @property
    def uncertain_mover_rate(self) -> float | None:
        u = self._subset(directional=False)
        return sum(g.hit for g in u) / len(u) if u else None

    def mean_abs_move_by_impact(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for impact in ("high", "medium", "low"):
            moves = [abs(g.excess_move_pct) for g in self.graded if g.impact == impact]
            if moves:
                out[impact] = statistics.mean(moves)
        return out


def _load_due(db_path: str, today: date, lookback_days: int) -> list[dict[str, Any]]:
    from sqlalchemy import text

    earliest = (today - timedelta(days=lookback_days)).isoformat()
    latest = (today - timedelta(days=REACTION_DAYS)).isoformat()
    with get_session(db_path) as session:
        rows = session.exec(
            text(
                "SELECT ticker, expected_date, direction, impact FROM pick_catalysts "
                "WHERE expected_date IS NOT NULL "
                "AND expected_date >= :earliest AND expected_date <= :latest "
                "ORDER BY run_id ASC"
            ),
            params={"earliest": earliest, "latest": latest},
        ).all()
    # The same event is often re-named on consecutive runs; grade it once.
    seen: set[tuple[str, str, str]] = set()
    due: list[dict[str, Any]] = []
    for ticker, expected_date, direction, impact in rows:
        key = (ticker, expected_date, direction)
        if key in seen:
            continue
        seen.add(key)
        due.append(
            {
                "ticker": ticker,
                "expected_date": expected_date,
                "direction": direction,
                "impact": impact,
            }
        )
    return due


def _window_move(closes: Any, event: date) -> float | None:
    before = _close_on_or_before(closes, event - timedelta(days=1))
    after = _close_on_or_before(closes, event + timedelta(days=REACTION_DAYS))
    if before is None or after is None or after[1] <= before[1] or before[0] <= 0:
        return None
    return (after[0] / before[0] - 1) * 100


def grade_catalysts(
    db_path: str,
    *,
    today: date | None = None,
    lookback_days: int = 365,
    fetch: Callable[[str, date, date], Any] = _fetch_history,
) -> CatalystReport:
    today = today or date.today()
    due = _load_due(db_path, today, lookback_days)
    if not due:
        return CatalystReport()
    dates = [date.fromisoformat(d["expected_date"]) for d in due]
    span_start = min(dates) - timedelta(days=10)
    span_end = max(dates) + timedelta(days=REACTION_DAYS + 3)

    spy = fetch("SPY", span_start, span_end)
    if spy is None or spy.empty:
        logger.warning("Catalyst grading: no SPY history — skipping")
        return CatalystReport()
    spy_closes = spy["Close"].dropna()

    closes_by_ticker: dict[str, Any] = {}
    for ticker in {d["ticker"] for d in due}:
        frame = fetch(ticker, span_start, span_end)
        if frame is not None and not frame.empty:
            closes_by_ticker[ticker] = frame["Close"].dropna()

    report = CatalystReport()
    for d, event in zip(due, dates, strict=True):
        closes = closes_by_ticker.get(d["ticker"])
        if closes is None:
            continue
        move = _window_move(closes, event)
        spy_move = _window_move(spy_closes, event)
        if move is None or spy_move is None:
            continue
        report.graded.append(
            GradedCatalyst(
                ticker=d["ticker"],
                expected_date=d["expected_date"],
                direction=d["direction"],
                impact=d["impact"],
                excess_move_pct=move - spy_move,
            )
        )
    return report


def format_catalyst_grading_block(report: CatalystReport) -> str:
    n = len(report.graded)
    if n < MIN_GRADED_FOR_PROMPT:
        return (
            f"Catalyst calls graded so far: {n} (too few to judge — treat your "
            f"catalyst reasoning as unproven)."
        )
    lines = [f"Catalyst calls graded: {n} (move vs SPY, day before to ~5 trading days after)."]
    hit = report.directional_hit_rate
    if hit is not None:
        k = len([g for g in report.graded if g.direction != "uncertain"])
        verdict = (
            "your direction calls have been informative"
            if hit >= 0.6
            else "your direction calls have NOT beaten a coin flip — weight them less"
            if hit <= 0.5
            else "your direction calls have been only marginally informative"
        )
        lines.append(f"  Directional calls: {hit:.0%} right on {k} — {verdict}.")
    mover = report.uncertain_mover_rate
    if mover is not None:
        lines.append(
            f"  'Uncertain' binary events that actually moved the stock >="
            f"{MOVER_THRESHOLD_PCT:.0f}%: {mover:.0%}."
        )
    by_impact = report.mean_abs_move_by_impact()
    if by_impact:
        parts = ", ".join(f"{k} {v:.1f}%" for k, v in by_impact.items())
        lines.append(f"  Mean |excess move| by stated impact: {parts}.")
        if "high" in by_impact and "low" in by_impact and by_impact["high"] <= by_impact["low"]:
            lines.append(
                "  Your 'high impact' labels have not moved stocks more than 'low' — "
                "don't let the impact label alone widen a scenario."
            )
    return "\n".join(lines)
