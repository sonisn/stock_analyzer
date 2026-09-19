"""Your portfolio vs SPY — time-weighted, from daily snapshots. No LLM.

The daily email stores the portfolio's total value (holdings + cash) each
run (`portfolio_snapshots`). Between two snapshots the return is

    r = (V_end - net_flows) / V_start - 1

where net_flows are deposits, withdrawals and transfers in that window
(from the brokerage's activity history), so adding money never counts as
performance. Chaining those gives the time-weighted return — the number
comparable to an index. SPY is measured over the same dates (adjusted
closes, so its dividends count too, as yours do).
"""

from __future__ import annotations

import html
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from ..discover.track_record import _close_on_or_before, _fetch_history
from ..logging import get_logger

logger = get_logger(__name__)


def time_weighted_return(
    snapshots: list[tuple[date, float]], flows: list[tuple[date, float]]
) -> float | None:
    """Chained return (%) across consecutive snapshots. A flow dated after
    one snapshot and on/before the next belongs to that window. Windows
    starting from a zero value are skipped."""
    if len(snapshots) < 2:
        return None
    growth = 1.0
    for (d0, v0), (d1, v1) in zip(snapshots, snapshots[1:], strict=False):
        flow = sum(a for d, a in flows if d0 < d <= d1)
        if v0 <= 0:
            continue
        growth *= (v1 - flow) / v0
    return (growth - 1) * 100


def spy_return(start: date, end: date, fetch: Callable = _fetch_history) -> float | None:
    frame = fetch("SPY", start - timedelta(days=7), end)
    if frame is None or frame.empty:
        return None
    closes = frame["Close"].dropna()
    first, last = _close_on_or_before(closes, start), _close_on_or_before(closes, end)
    if not first or not last or first[0] <= 0:
        return None
    return (last[0] / first[0] - 1) * 100


def performance_vs_spy(
    snapshots: list[tuple[date, float]],
    flows: list[tuple[date, float]],
    *,
    windows: dict[str, date],
    fetch: Callable = _fetch_history,
) -> list[dict[str, Any]]:
    """One row per named window ({"Since tracking began": start, ...}):
    portfolio TWR, SPY, and the difference. Windows with < 2 snapshots in
    them are skipped."""
    rows = []
    for label, start in windows.items():
        inside = [(d, v) for d, v in snapshots if d >= start]
        if len(inside) < 2:
            continue
        mine = time_weighted_return(inside, flows)
        spy = spy_return(inside[0][0], inside[-1][0], fetch)
        rows.append(
            {
                "label": label,
                "start": inside[0][0],
                "end": inside[-1][0],
                "portfolio_pct": mine,
                "spy_pct": spy,
                "diff_pts": None if mine is None or spy is None else mine - spy,
            }
        )
    return rows


def render_performance_html(rows: list[dict[str, Any]], *, first_day: date | None) -> str:
    from .health import _table

    parts = ["<h2>Your portfolio vs SPY</h2>"]
    if not rows:
        since = f" (tracking began {first_day:%b %d, %Y})" if first_day else ""
        parts.append(
            f"<p>Not enough history yet{since}: the daily email records the portfolio's "
            "value each weekday, and this comparison fills in from there.</p>"
        )
        return "".join(parts)

    def pct(v: float | None) -> str:
        return "—" if v is None else f"{v:+.1f}%"

    parts.append(
        _table(
            ["Period", "Your portfolio", "SPY", "Difference"],
            [
                [
                    html.escape(f"{r['label']} ({r['start']:%b %d} – {r['end']:%b %d})"),
                    pct(r["portfolio_pct"]),
                    pct(r["spy_pct"]),
                    "—" if r["diff_pts"] is None else f"{r['diff_pts']:+.1f} pts",
                ]
                for r in rows
            ],
        )
    )
    parts.append(
        '<p style="font-size:13px;color:#6b7280">Time-weighted: deposits, withdrawals and '
        "transfers are taken out, so only investment results count. Both include "
        "dividends.</p>"
    )
    return "".join(parts)
