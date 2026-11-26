"""The whole portfolio's fall from its high, and an email the day it deepens.

The per-holding alert (drop_alert.py) fires on one stock's unusual day.
From Oct 21 to Nov 20 2026 the portfolio lost $124k (-26%) without one
email saying so: the falls were spread over weeks and many names. This
checks the stored daily totals (portfolio_snapshots) after each close:

  - the drawdown is today's total against the highest close of the past
    year;
  - the email goes out the first evening it reaches PORTFOLIO_DRAWDOWN_ALERT_PCT
    (10%), and again at each further PORTFOLIO_DRAWDOWN_STEP_PCT (15%,
    20%, ...) — never twice for the same step within one fall. A new high
    resets it.

The email says what fell (holdings by dollars lost since the high), how
SPY did over the same days, how concentrated the book is, and the market
status — and, from a 2005-2026 replay, that selling on the drop itself has
cost more than it saved. Deposits and withdrawals are in the totals: a
large withdrawal reads as a fall, and the email says the number is the
account total, not a return.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

LOOKBACK_DAYS = 365


@dataclass
class Drawdown:
    peak_day: date
    peak: float
    day: date
    total: float
    band: int  # the deepest alert step reached today (0 below the threshold)
    prior_band: int  # the deepest step reached earlier in this fall

    @property
    def pct(self) -> float:
        return (1 - self.total / self.peak) * 100 if self.peak else 0.0

    @property
    def loss(self) -> float:
        return self.peak - self.total

    @property
    def is_new_step(self) -> bool:
        return self.band > self.prior_band


def _band(pct: float, threshold: float, step: float) -> int:
    if threshold <= 0 or pct < threshold:
        return 0
    return int(threshold + ((pct - threshold) // step) * step) if step > 0 else int(threshold)


def drawdown(
    totals: list[tuple[date, float]],
    *,
    threshold: float = 10.0,
    step: float = 5.0,
    lookback_days: int = LOOKBACK_DAYS,
) -> Drawdown | None:
    """The latest total against its 1-year high. None without history."""
    rows = sorted((d, t) for d, t in totals if t and t > 0)
    if len(rows) < 2:
        return None
    today, total = rows[-1]
    window = [(d, t) for d, t in rows if d >= today - timedelta(days=lookback_days)]
    peak_i = max(range(len(window)), key=lambda i: (window[i][1], window[i][0]))
    peak_day, peak = window[peak_i]
    prior = 0
    for _, t in window[peak_i + 1 : -1]:
        prior = max(prior, _band((1 - t / peak) * 100, threshold, step))
    return Drawdown(
        peak_day=peak_day,
        peak=peak,
        day=today,
        total=total,
        band=_band((1 - total / peak) * 100, threshold, step),
        prior_band=prior,
    )


def change_between(bars: list[tuple[date, float]], start: date, end: date) -> float | None:
    """Percent change from the close on/before `start` to the close on/before `end`."""
    a = next((c for d, c in reversed(bars) if d <= start), None)
    b = next((c for d, c in reversed(bars) if d <= end), None)
    return (b / a - 1) * 100 if a and b else None


def attribution(
    units: dict[str, float],
    series: dict[str, list[tuple[date, float]]],
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    """Dollars each holding lost (or made) since `start`, at today's share
    count — trades in between are not seen. Biggest losers first."""
    out = []
    for t, n in units.items():
        bars = series.get(t.upper()) or []
        a = next((c for d, c in reversed(bars) if d <= start), None)
        b = next((c for d, c in reversed(bars) if d <= end), None)
        if not a or not b or not n:
            continue
        out.append({"ticker": t, "pct": (b / a - 1) * 100, "usd": n * (b - a)})
    return sorted(out, key=lambda r: r["usd"])


# --- the email ---------------------------------------------------------------------


def _money(v: float) -> str:
    return f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}"


def build_alert(
    dd: Drawdown,
    *,
    spy_pct: float | None,
    movers: list[dict[str, Any]],
    concentration: dict[str, Any] | None,
    market: dict[str, Any] | None,
    core_fund: str,
) -> tuple[str, str]:
    """(subject, html body)."""
    subject = f"Portfolio down {dd.pct:.0f}% from its high ({_money(-dd.loss)})"
    p = [
        "<html><body style='font-family:sans-serif;max-width:640px'>",
        f"<p>The portfolio closed at <b>{_money(dd.total)}</b> on {dd.day:%b %d}, "
        f"<b>{dd.pct:.1f}%</b> ({_money(-dd.loss)}) below its {dd.peak_day:%b %d} high of "
        f"{_money(dd.peak)}."
        + (f" SPY over the same days: {spy_pct:+.1f}%." if spy_pct is not None else "")
        + "</p>",
    ]
    losers = [m for m in movers if m["usd"] < 0][:6]
    if losers:
        p.append("<h3>Where it went</h3><ul>")
        p += [
            f"<li><b>{html.escape(m['ticker'])}</b> {m['pct']:+.1f}% ({_money(m['usd'])})</li>"
            for m in losers
        ]
        p.append("</ul>")
    conc = concentration or {}
    for g in conc.get("groups") or []:
        if not g.get("over"):
            continue
        p.append(
            f"<p><b>Concentration:</b> {html.escape(', '.join(g['tickers']))} move together and "
            f"are {g['pct']:.0f}% of the portfolio, against a "
            f"{conc.get('max_group_pct', 0):g}% cap"
            + (
                f"; in the past year they fell as much as {g['worst_drop_pct']:.0f}% together"
                if g.get("worst_drop_pct")
                else ""
            )
            + ". That, more than the market, is what sets how far the total falls. The "
            f"fix is size: trim toward the cap in the IRA (no tax) into {html.escape(core_fund)} "
            "at the next rebalance, a step at a time.</p>"
        )
    over_vol = [v for v in conc.get("vol_caps") or [] if v.get("over")]
    if over_vol:
        p.append(
            "<p><b>Too big for how much they swing:</b> "
            + "; ".join(
                f"{html.escape(v['ticker'])} {v['pct']:.0f}% of the portfolio, swings "
                f"{v['vol_pct']:.0f}%/yr, cap {v['cap_pct']:g}%"
                for v in over_vol
            )
            + ".</p>"
        )
    if market and market.get("status"):
        p.append(
            f"<p><b>Market:</b> {html.escape(market['status'])}"
            + (f" — {html.escape(market['detail'])}" if market.get("detail") else "")
            + ".</p>"
        )
    p.append(
        "<p><i>What not to do: sell everything now. Replaying 2005-2026, selling QQQ when the "
        "market entered a correction and buying back later cut the worst fall from 53% to 44% "
        "but cost 5.4 points a year; the three months after a correction signal averaged "
        "+6%, better than other days. For a 3-5 year holding the lever is how much sits in "
        "one bet, not when to be out. The total includes deposits and withdrawals.</i></p>"
    )
    p.append("</body></html>")
    return subject, "\n".join(p)


def latest_market(db: str) -> dict[str, Any] | None:
    """The newest ibd_market row as {day, status, detail}, or None."""
    import os
    import sqlite3

    try:
        with sqlite3.connect(f"file:{os.path.expanduser(db)}?mode=ro", uri=True) as c:
            row = c.execute(
                "SELECT day, status, detail FROM ibd_market ORDER BY day DESC LIMIT 1"
            ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    return {"day": row[0], "status": row[1], "detail": row[2]} if row else None
