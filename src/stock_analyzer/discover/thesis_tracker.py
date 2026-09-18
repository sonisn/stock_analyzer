"""Between-run thesis check for open picks.

The track record grades a pick once it is 30/90 days old; nothing looked
at it in between, and the only mechanical exit was the -20% stop-loss on
holdings. This re-checks every recent pick against what the pipeline said
when it made the call, with no LLM involved:

  - the pick's own bear / bull scenario targets (`pick_scenarios`): a
    return already at or past the bear-case target means the downside the
    ranker priced in has happened; past the bull target means the upside
    is used up and the pick needs re-underwriting;
  - the trend rule the screen required at entry (close above the 200-day
    average), and performance against SPY since the pick;
  - the pick's dated catalysts (`pick_catalysts`): a directional call
    whose reaction went the wrong way, and events coming up soon;
  - EPS estimate revisions, when this run already fetched them.

A pick is "open" for `OPEN_WINDOW_DAYS` after its latest run; re-picking a
name replaces its thesis (new entry price, new scenarios).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal

from ..db.session import get_session
from ..logging import get_logger
from .catalyst_grading import MOVER_THRESHOLD_PCT, REACTION_DAYS, _window_move
from .track_record import _close_on_or_after, _close_on_or_before, _fetch_history

logger = get_logger(__name__)

OPEN_WINDOW_DAYS = 180
LAG_THRESHOLD_PTS = 10.0
UPCOMING_DAYS = 14
_TREND_DAYS = 200

Severity = Literal["broken", "target", "watch", "info"]
_STATUS_ORDER = {"BROKEN": 0, "TARGET HIT": 1, "WATCH": 2, "INTACT": 3}


@dataclass(frozen=True)
class OpenPick:
    run_id: int
    ticker: str
    pick_date: date
    entry_price: float | None
    bear_target_pct: float | None
    bull_target_pct: float | None
    catalysts: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ThesisSignal:
    severity: Severity
    text: str


@dataclass
class ThesisCheck:
    ticker: str
    pick_date: date
    return_pct: float
    spy_return_pct: float | None
    bear_target_pct: float | None
    bull_target_pct: float | None
    signals: list[ThesisSignal] = field(default_factory=list)

    @property
    def excess_pct(self) -> float | None:
        if self.spy_return_pct is None:
            return None
        return self.return_pct - self.spy_return_pct

    @property
    def status(self) -> str:
        severities = {s.severity for s in self.signals}
        if "broken" in severities:
            return "BROKEN"
        if "target" in severities:
            return "TARGET HIT"
        if "watch" in severities:
            return "WATCH"
        return "INTACT"


def load_open_picks(
    db_path: str, *, today: date | None = None, window_days: int = OPEN_WINDOW_DAYS
) -> list[OpenPick]:
    """Latest pick per ticker made in the last `window_days`, with its
    scenario targets and every catalyst named for it in that window."""
    from sqlalchemy import text

    today = today or date.today()
    earliest = (today - timedelta(days=window_days)).isoformat()
    with get_session(db_path) as session:
        picks = session.exec(
            text(
                "SELECT p.run_id, p.rank, p.ticker, p.entry_price, r.run_at "
                "FROM picks p JOIN runs r ON r.id = p.run_id "
                "WHERE r.run_at >= :earliest ORDER BY p.run_id ASC"
            ),
            params={"earliest": earliest},
        ).all()
        scenarios = session.exec(
            text(
                "SELECT run_id, rank, label, target_return_pct FROM pick_scenarios "
                "WHERE run_id IN (SELECT id FROM runs WHERE run_at >= :earliest)"
            ),
            params={"earliest": earliest},
        ).all()
        catalysts = session.exec(
            text(
                "SELECT c.ticker, c.event, c.expected_date, c.direction, c.impact "
                "FROM pick_catalysts c JOIN runs r ON r.id = c.run_id "
                "WHERE r.run_at >= :earliest AND c.expected_date IS NOT NULL "
                "ORDER BY c.run_id ASC"
            ),
            params={"earliest": earliest},
        ).all()

    targets: dict[tuple[int, int], dict[str, float]] = {}
    for run_id, rank, label, target in scenarios:
        targets.setdefault((run_id, rank), {})[label] = target

    # The same event is usually re-named on consecutive runs; keep it once.
    events: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for ticker, event, expected_date, direction, impact in catalysts:
        events.setdefault(ticker, {})[(expected_date, direction)] = {
            "event": event,
            "expected_date": expected_date,
            "direction": direction,
            "impact": impact,
        }

    latest: dict[str, OpenPick] = {}
    for run_id, rank, ticker, entry_price, run_at in picks:
        t = targets.get((run_id, rank), {})
        latest[ticker] = OpenPick(
            run_id=run_id,
            ticker=ticker,
            pick_date=datetime.fromisoformat(str(run_at)).date(),
            entry_price=entry_price,
            bear_target_pct=t.get("bear"),
            bull_target_pct=t.get("bull"),
            catalysts=tuple(
                sorted(
                    events.get(ticker, {}).values(),
                    key=lambda c: c["expected_date"],
                )
            ),
        )
    return sorted(latest.values(), key=lambda p: p.pick_date)


def _catalyst_signals(
    pick: OpenPick, closes: Any, spy_closes: Any, today: date
) -> list[ThesisSignal]:
    out: list[ThesisSignal] = []
    for c in pick.catalysts:
        try:
            when = date.fromisoformat(c["expected_date"])
        except ValueError:
            continue
        label = f"{c['event']} ({when.isoformat()})"
        if today <= when <= today + timedelta(days=UPCOMING_DAYS):
            out.append(
                ThesisSignal(
                    "info",
                    f"Upcoming in {(when - today).days}d: {label}, called {c['direction']}",
                )
            )
            continue
        if when < pick.pick_date or when > today - timedelta(days=REACTION_DAYS):
            continue
        move, spy_move = _window_move(closes, when), _window_move(spy_closes, when)
        if move is None or spy_move is None:
            continue
        excess = move - spy_move
        # Only a drop matters to a long thesis: a positive call that fell
        # failed, and a negative/uncertain one that fell was the risk landing.
        if excess < -MOVER_THRESHOLD_PCT:
            what = (
                "Catalyst went the wrong way"
                if c["direction"] == "positive"
                else "Flagged risk event hit"
            )
            out.append(
                ThesisSignal(
                    "watch",
                    f"{what}: {label}, called {c['direction']}, moved {excess:+.1f}% vs SPY",
                )
            )
    return out


def check_theses(
    picks: list[OpenPick],
    *,
    today: date | None = None,
    eps_revisions: dict[str, dict[str, Any]] | None = None,
    fetch: Callable[[str, date, date], Any] = _fetch_history,
    lag_threshold_pts: float = LAG_THRESHOLD_PTS,
) -> list[ThesisCheck]:
    """Score each open pick's thesis against prices since the pick. Picks
    with no price history are skipped (logged), not guessed at."""
    if not picks:
        return []
    today = today or date.today()
    eps_revisions = eps_revisions or {}
    start = min(p.pick_date for p in picks) - timedelta(days=int(_TREND_DAYS * 1.6))
    spy = fetch("SPY", start, today)
    spy_closes = spy["Close"].dropna() if spy is not None and not spy.empty else None

    results: list[ThesisCheck] = []
    for pick in picks:
        frame = fetch(pick.ticker, start, today)
        if frame is None or frame.empty:
            logger.warning("Thesis check: no price history for %s — skipped", pick.ticker)
            continue
        closes = frame["Close"].dropna()
        last = _close_on_or_before(closes, today)
        entry = pick.entry_price or (_close_on_or_after(closes, pick.pick_date) or (None,))[0]
        if last is None or not entry:
            continue
        ret = (last[0] / entry - 1) * 100
        spy_ret = None
        if spy_closes is not None:
            spy_entry = _close_on_or_after(spy_closes, pick.pick_date)
            spy_last = _close_on_or_before(spy_closes, today)
            if spy_entry and spy_last and spy_entry[0] > 0:
                spy_ret = (spy_last[0] / spy_entry[0] - 1) * 100
        check = ThesisCheck(
            ticker=pick.ticker,
            pick_date=pick.pick_date,
            return_pct=ret,
            spy_return_pct=spy_ret,
            bear_target_pct=pick.bear_target_pct,
            bull_target_pct=pick.bull_target_pct,
        )
        excess = check.excess_pct
        lagging = excess is not None and excess <= -lag_threshold_pts

        if pick.bear_target_pct is not None and ret <= pick.bear_target_pct:
            check.signals.append(
                ThesisSignal(
                    "broken",
                    f"{ret:+.1f}% since the pick, at or past its own bear-case "
                    f"target of {pick.bear_target_pct:+.0f}%",
                )
            )
        elif pick.bull_target_pct is not None and ret >= pick.bull_target_pct:
            check.signals.append(
                ThesisSignal(
                    "target",
                    f"{ret:+.1f}% since the pick, past its bull-case target of "
                    f"{pick.bull_target_pct:+.0f}%: re-underwrite or take profits",
                )
            )

        trailing = closes.tail(_TREND_DAYS)
        sma = float(trailing.mean()) if len(trailing) >= _TREND_DAYS else None
        if sma is not None and last[0] < sma:
            text = (
                f"Closed {(1 - last[0] / sma) * 100:.1f}% below its 200-day average: "
                f"the screen's entry trend rule no longer holds"
            )
            if lagging:
                check.signals.append(
                    ThesisSignal("broken", f"{text}, and {excess:+.1f} pts vs SPY")
                )
            else:
                check.signals.append(ThesisSignal("watch", text))
        elif lagging:
            check.signals.append(
                ThesisSignal("watch", f"Lagging SPY by {-excess:.1f} pts since the pick")
            )

        if spy_closes is not None:
            check.signals.extend(_catalyst_signals(pick, closes, spy_closes, today))

        rev = eps_revisions.get(pick.ticker) or {}
        if rev.get("direction_30d") == "lowering":
            net = rev.get("net_revisions_30d")
            detail = f" (net {net:+d} revisions in 30d)" if isinstance(net, int) else ""
            check.signals.append(ThesisSignal("watch", f"Analysts cutting EPS estimates{detail}"))
        results.append(check)
    return sorted(results, key=lambda c: (_STATUS_ORDER[c.status], c.ticker))


def thesis_report_data(checks: list[ThesisCheck]) -> list[dict[str, Any]]:
    """Plain-dict form for the report section and the Reviewer payload."""
    return [
        {
            "ticker": c.ticker,
            "status": c.status,
            "pick_date": c.pick_date.isoformat(),
            "return_pct": round(c.return_pct, 2),
            "spy_return_pct": None if c.spy_return_pct is None else round(c.spy_return_pct, 2),
            "excess_pct": None if c.excess_pct is None else round(c.excess_pct, 2),
            "bear_target_pct": c.bear_target_pct,
            "bull_target_pct": c.bull_target_pct,
            "signals": [{"severity": s.severity, "text": s.text} for s in c.signals],
        }
        for c in checks
    ]
