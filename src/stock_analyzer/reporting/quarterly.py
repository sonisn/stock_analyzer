"""Quarterly review — how last quarter's advice worked out. No LLM calls.

Sent on the first trading day of each quarter. It grades every suggestion
made during the quarter that just ended — daily-email decision lines and
rebalance actions (the `suggestions` table) plus discover picks — against
what the stock did from the day it was suggested to today, next to SPY
and, for a sale, next to the stock it was suggested to switch into:

  - SELL / TRIM / TAX_LOSS: a good call when the stock then lagged SPY
    (switch: when the suggested replacement beat it);
  - BUY / ADD / discover picks: a good call when the stock beat SPY;
  - REVIEW (a -20% thesis re-check, i.e. "hold unless broken"): shown,
    graded as holding — good when the stock then beat SPY;
  - WRITE_CALL / SELL_PUT: the underlying's move is shown; the option's
    own outcome isn't reconstructed;
  - STANDOUT (an earnings standout the daily email showed): a good call
    when the stock then beat SPY.

A suggestion repeated on several days is graded once, from the first day.
"Acted on?" compares today's position with the one held when suggested.
A quarter is a short window for 3-5 year holdings — the email says so;
the value is in the pattern across quarters.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from datetime import date, timedelta
from statistics import mean
from typing import Any

from sqlalchemy import text

from ..db.repository import fetch_suggestions
from ..db.session import exec_sql, get_session
from ..discover.track_record import _close_on_or_after, _close_on_or_before, _fetch_history
from ..logging import get_logger
from .health import _badge, _table

logger = get_logger(__name__)

SELLS = {"SELL", "TRIM", "TAX_LOSS"}
BUYS = {"BUY", "ADD"}
OPTIONS = {"WRITE_CALL", "SELL_PUT"}

# --- calendar ------------------------------------------------------------------


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    """NYSE rule: a Saturday holiday is observed Friday, a Sunday one Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def market_closed(d: date) -> bool:
    """Weekends and the NYSE holidays that can fall in a quarter's first
    days: New Year's Day, Good Friday, Independence Day."""
    if d.weekday() >= 5:
        return True
    # New Year's on a Saturday is not observed on Dec 31 by the NYSE.
    new_year = date(d.year, 1, 1)
    if new_year.weekday() != 5 and d == _observed(new_year):
        return True
    return d in (_easter(d.year) - timedelta(days=2), _observed(date(d.year, 7, 4)))


def quarter_of(d: date) -> int:
    return (d.month - 1) // 3 + 1


def first_trading_day(year: int, quarter: int) -> date:
    d = date(year, 3 * (quarter - 1) + 1, 1)
    while market_closed(d):
        d += timedelta(days=1)
    return d


def is_first_trading_day_of_quarter(d: date) -> bool:
    return d == first_trading_day(d.year, quarter_of(d))


def previous_quarter(today: date) -> tuple[str, date, date]:
    """(label, first day, last day) of the quarter before `today`'s."""
    q, y = quarter_of(today) - 1, today.year
    if q == 0:
        q, y = 4, y - 1
    start = date(y, 3 * (q - 1) + 1, 1)
    end = (date(y + (q == 4), (3 * q) % 12 + 1, 1)) - timedelta(days=1)
    return f"Q{q} {y}", start, end


# --- collect -------------------------------------------------------------------


def collect_suggestions(db_path: str, start: date, end: date) -> list[dict[str, Any]]:
    """Suggestions made in [start, end], one per (source, action, ticker):
    the first time it was given."""
    items: dict[tuple[str, str, str], dict[str, Any]] = {}
    with get_session(db_path) as session:
        for s in fetch_suggestions(session, start=start.isoformat(), end=end.isoformat()):
            key = (s.source, s.action, s.ticker)
            items.setdefault(
                key,
                {
                    "suggested_on": s.suggested_on,
                    "source": s.source,
                    "action": s.action,
                    "ticker": s.ticker,
                    "detail": s.detail,
                    "price": s.price,
                    "units_held": s.units_held,
                    "reinvest_into": s.reinvest_into,
                },
            )
        picks = exec_sql(
            session,
            text(
                "SELECT p.ticker, r.run_at, p.entry_price, p.rank FROM picks p "
                "JOIN runs r ON r.id = p.run_id "
                "WHERE substr(r.run_at, 1, 10) BETWEEN :s AND :e ORDER BY r.run_at"
            ),
            params={"s": start.isoformat(), "e": end.isoformat()},
        ).all()
        # Picks made before the ledger recorded positions: a holdings
        # review of the ticker in the 60 days up to the pick means it was
        # already owned, so "acted on" can't be read from today's position.
        reviewed = exec_sql(
            session,
            text(
                "SELECT h.ticker, substr(r.run_at, 1, 10) FROM holdings_reviews h "
                "JOIN runs r ON r.id = h.run_id"
            ),
        ).all()
    reviewed_on: dict[str, list[str]] = {}
    for ticker, day in reviewed:
        reviewed_on.setdefault(ticker, []).append(day)
    for ticker, run_at, entry, rank in picks:
        day = str(run_at)[:10]
        lo = (date.fromisoformat(day) - timedelta(days=60)).isoformat()
        items.setdefault(
            ("discover", "BUY", ticker),
            {
                "suggested_on": day,
                "source": "discover",
                "action": "BUY",
                "ticker": ticker,
                "detail": f"discover pick #{rank}",
                "price": entry,
                "units_held": None,
                "reinvest_into": None,
                "held_before": any(lo <= d <= day for d in reviewed_on.get(ticker, [])),
            },
        )
    return sorted(items.values(), key=lambda i: (i["suggested_on"], i["ticker"]))


# --- grade ---------------------------------------------------------------------


def _return_since(
    ticker: str, since: date, today: date, fetch: Callable, entry: float | None = None
) -> float | None:
    frame = fetch(ticker, since - timedelta(days=7), today)
    if frame is None or frame.empty:
        return None
    closes = frame["Close"].dropna()
    last = _close_on_or_before(closes, today)
    if entry is None:
        first = _close_on_or_after(closes, since)
        entry = first[0] if first else None
    if not last or not entry:
        return None
    return (last[0] / entry - 1) * 100


def _acted(item: dict[str, Any], units_now: dict[str, float]) -> str:
    action, before = item["action"], item.get("units_held")
    now = units_now.get(item["ticker"], 0.0)
    if action in SELLS:
        return "?" if before is None else ("yes" if now < before - 1e-6 else "no")
    if item.get("held_before"):
        return "held already"
    if action in BUYS or item["source"] == "discover":
        return "yes" if now > (before or 0.0) + 1e-6 else "no"
    return "—"


# Below this age a suggestion has no outcome worth naming. 30 days is
# the shortest horizon anything else here is measured on, and it is still
# noise against a 3-5 year holding period — the label says "too early"
# rather than inventing a verdict from a few days of drift.
VERDICT_MIN_AGE_DAYS = 30


def grade_suggestions(
    items: list[dict[str, Any]],
    *,
    today: date,
    units_now: dict[str, float],
    fetch: Callable = _fetch_history,
) -> list[dict[str, Any]]:
    """Each item plus: return_pct and spy_pct since it was suggested,
    reinvest_pct (the suggested replacement), edge_pct (positive = the
    advice helped), verdict and acted."""
    spy_cache: dict[str, float | None] = {}
    out = []
    for item in items:
        since = date.fromisoformat(item["suggested_on"])
        ret = _return_since(item["ticker"], since, today, fetch)
        if item["suggested_on"] not in spy_cache:
            spy_cache[item["suggested_on"]] = _return_since("SPY", since, today, fetch)
        spy = spy_cache[item["suggested_on"]]
        swap = (
            _return_since(item["reinvest_into"], since, today, fetch)
            if item.get("reinvest_into")
            else None
        )
        edge = None
        action = item["action"]
        if ret is not None:
            if action in SELLS:
                # Against the named replacement when there was one, else SPY.
                base = swap if swap is not None else spy
                edge = None if base is None else base - ret
            elif action not in OPTIONS and spy is not None:
                edge = ret - spy
        verdict = "—"
        age = (today - since).days
        if edge is not None and action not in OPTIONS:
            if age < VERDICT_MIN_AGE_DAYS:
                # Advice given days ago has no outcome yet. Grading it
                # anyway prints "good call" against a 0.0% edge, which
                # reads as a result and is only the absence of one.
                verdict = "too early"
            else:
                verdict = "good call" if edge >= 0 else "missed"
        out.append(
            {
                **item,
                "return_pct": ret,
                "spy_pct": spy,
                "reinvest_pct": swap,
                "edge_pct": edge,
                "verdict": verdict,
                "acted": _acted(item, units_now),
            }
        )
    return out


GROUPS = (
    ("Sell / trim advice", lambda g: g["action"] in SELLS),
    ("Buy / add advice and discover picks", lambda g: g["action"] in BUYS),
    ("Thesis re-checks (hold unless broken)", lambda g: g["action"] == "REVIEW"),
    ("Option ideas", lambda g: g["action"] in OPTIONS),
    ("Earnings standouts", lambda g: g["action"] == "STANDOUT"),
)


def summarize(graded: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {}
    for name, keep in GROUPS:
        rows = [g for g in graded if keep(g)]
        scored = [g["edge_pct"] for g in rows if g["edge_pct"] is not None]
        groups[name] = {
            "n": len(rows),
            "n_scored": len(scored),
            "n_good": sum(1 for e in scored if e >= 0),
            "mean_edge": mean(scored) if scored else None,
            "n_acted": sum(1 for g in rows if g["acted"] == "yes"),
            "n_actable": sum(1 for g in rows if g["acted"] in ("yes", "no")),
        }
    return {"n": len(graded), "groups": groups}


# --- render --------------------------------------------------------------------


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:+.1f}%"


def _pts(v: float | None) -> str:
    return "—" if v is None else f"{v:+.1f} pts"


def headline(summary: dict[str, Any]) -> list[str]:
    lines = []
    for name, s in summary["groups"].items():
        if not s["n"] or name == "Option ideas":
            continue
        line = f"{name}: {s['n']} suggestion(s)"
        if s["n_scored"]:
            line += (
                f", {s['n_good']} of {s['n_scored']} worked out "
                f"(average edge {_pts(s['mean_edge'])})"
            )
        if s["n_actable"]:
            line += f"; you acted on {s['n_acted']} of {s['n_actable']}"
        lines.append(line + ".")
    return lines


def render_options_income_html(income: Any, *, label: str) -> str:
    """Premium earned from written options, and what it cost in shares.

    Without this a called-away holding is graded as a sale into strength
    and the premium that paid for it is invisible — the wheel looks like
    bad timing. Long options and same-day round trips are shown apart
    from it: buying a call is a bet, not income.
    """
    if income is None or not getattr(income, "contracts_sold", 0):
        return ""
    rows = [
        [
            html.escape(ticker),
            f"{vals['contracts']:.0f}",
            f"${vals['net_premium']:,.0f}",
            f"{vals['shares_called_away']:,.0f}" if vals["shares_called_away"] else "—",
        ]
        for ticker, vals in sorted(
            income.by_underlying.items(), key=lambda kv: -kv[1]["net_premium"]
        )
    ]
    parts = [
        f'<section class="health"><h2>Options written — {html.escape(label)}</h2>',
        f"<p>${income.premium_collected:,.0f} collected, ${-income.premium_paid_to_close:,.0f} "
        f"paid to close: <b>${income.net_premium:,.0f} net</b> from {income.contracts_sold} "
        f"contract(s). {income.expired} expired worthless, {income.bought_back} bought back, "
        f"{income.assigned} assigned.</p>",
    ]
    if income.shares_called_away:
        called = ", ".join(
            f"{a.contracts * 100:,.0f} {a.underlying} at ${a.strike:,.0f}"
            for a in income.assignments
        )
        parts.append(
            f'<p style="color:#9c1010">Called away: {html.escape(called)}. Those shares left '
            f"the portfolio at the strike — the premium above is what you were paid for "
            f"giving up the upside past it.</p>"
        )
    parts.append(_table(["Underlying", "Contracts", "Net premium", "Shares called away"], rows))
    extras = []
    if income.long_positions:
        net = sum(p.net_premium for p in income.long_positions)
        extras.append(f"{len(income.long_positions)} long position(s), ${net:,.0f}")
    if income.day_trades:
        net = sum(p.net_premium for p in income.day_trades)
        extras.append(f"{len(income.day_trades)} same-day round trip(s), ${net:,.0f}")
    if extras:
        parts.append(
            '<p style="font-size:13px;color:#6b7280">Not counted as premium income: '
            + html.escape("; ".join(extras))
            + ". Buying options is a trade, not income earned on a holding.</p>"
        )
    parts.append("</section>")
    return "".join(parts)


def render_quarterly_html(
    *,
    label: str,
    start: date,
    end: date,
    graded: list[dict[str, Any]],
    summary: dict[str, Any],
    health_html: str = "",
    performance_html: str = "",
    options_html: str = "",
    plan_html: str = "",
) -> str:
    from .html import _wrap_html

    parts = [
        f"<p>Advice given {start:%b %d} – {end:%b %d, %Y}, measured from the day it was "
        f"given to today. A quarter is a short window for 3-5 year holdings — read it for "
        f"the pattern, not single names.</p>"
    ]
    if performance_html:
        parts.append(performance_html)
    if options_html:
        parts.append(options_html)
    if plan_html:
        parts.append(plan_html)
    if not graded:
        parts.append("<p>No suggestions were recorded last quarter.</p>")
    else:
        parts.append(
            '<section class="health"><h2>Last quarter at a glance</h2><ul>'
            + "".join(f"<li>{html.escape(line)}</li>" for line in headline(summary))
            + "</ul></section>"
        )
    for name, keep in GROUPS:
        rows = [g for g in graded if keep(g)]
        if not rows:
            continue
        parts.append(f"<h2>{html.escape(name)}</h2>")
        sells = name.startswith("Sell")
        header = ["", "Date", "Ticker", "Action", "Stock since", "SPY since"]
        header += ["Switch into", "Edge", "Acted?"] if sells else ["Edge", "Acted?"]
        table_rows = []
        for g in rows:
            badge = _badge(g["verdict"].upper()) if g["verdict"] != "—" else ""
            row = [
                badge,
                g["suggested_on"],
                html.escape(g["ticker"]),
                html.escape(g["action"].replace("_", " ").lower()),
                _pct(g["return_pct"]),
                _pct(g["spy_pct"]),
            ]
            if sells:
                row.append(
                    f"{html.escape(g['reinvest_into'])} {_pct(g['reinvest_pct'])}"
                    if g.get("reinvest_into")
                    else "—"
                )
            row += [_pts(g["edge_pct"]), g["acted"]]
            table_rows.append(row)
        parts.append(_table(header, table_rows))
    parts.append(
        '<p style="font-size:13px;color:#6b7280">Edge: for sell advice, how much the '
        "stock lagged its suggested replacement (or SPY when none was named) — positive "
        "means selling helped; for buys and holds, how much it beat SPY. Acted? compares "
        "today's position with the one held when the advice was given.</p>"
    )
    if health_html:
        parts.append(health_html)
    return _wrap_html(f"Quarterly review — {label}", "".join(parts))
