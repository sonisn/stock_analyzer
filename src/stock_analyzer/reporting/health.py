"""The "Portfolio health" block — the deterministic top of the daily email.

No LLM calls. It reuses the checks the discover/rebalance pipelines run,
applied to today's holdings, so the daily email flags what needs attention
between rebalance runs:

  - snapshot: market value and unrealized P/L;
  - stop-loss: positions at or past the rebalance pipeline's -20% hard
    stop (discover/rebalance_holdings.py), and ones within 5 points of it;
  - thesis check: holdings that were recent discover picks, re-checked
    against their own targets and trend (discover/thesis_tracker.py);
  - sector weight: any sector above the Sizer's book cap
    (DISCOVER_MAX_SECTOR_PCT);
  - tax-loss harvesting: taxable slices past the HARVEST_* thresholds
    (discover/tax_harvest.py; swaps are in the rebalance report, which has
    the peer data);
  - earnings in the next 7 days.

Each check is isolated: one that fails is listed as unavailable and the
rest still render.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..logging import get_logger

logger = get_logger(__name__)

STOP_LOSS_PCT = -20.0  # discover/rebalance_holdings.apply_stop_loss_overrides
NEAR_STOP_PTS = 5.0
EARNINGS_DAYS = 7


@dataclass
class PortfolioHealth:
    snapshot: dict[str, float] = field(default_factory=dict)
    stop_loss: list[dict[str, Any]] = field(default_factory=list)
    thesis: list[dict[str, Any]] = field(default_factory=list)
    sectors: list[dict[str, Any]] = field(default_factory=list)
    harvest: list[dict[str, Any]] = field(default_factory=list)
    earnings: list[dict[str, Any]] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)


def aggregate_positions(holdings: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    """{ticker: units, cost, value} across accounts, from brokerage rows."""
    out: dict[str, dict[str, float]] = {}
    for items in holdings.values():
        for h in items:
            ticker = h.get("ticker")
            units = float(h.get("units") or 0)
            if not ticker or not units:
                continue
            row = out.setdefault(ticker, {"units": 0.0, "cost": 0.0, "value": 0.0})
            row["units"] += units
            row["cost"] += units * float(h.get("average_purchase_price") or 0)
            row["value"] += units * float(h.get("price") or 0)
    return out


def build_portfolio_health(
    holdings: dict[str, list[dict[str, Any]]],
    *,
    max_sector_pct: float = 30.0,
    sector_of: Callable[[list[str]], dict[str, str]] | None = None,
    held_thesis_checks: Callable[[set[str]], list[dict[str, Any]]] | None = None,
    harvest: Callable[[], list[dict[str, Any]]] | None = None,
    earnings: Callable[[list[str]], dict[str, dict[str, Any]]] | None = None,
) -> PortfolioHealth:
    health = PortfolioHealth()
    positions = aggregate_positions(holdings)
    tickers = sorted(positions)

    def attempt(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — one check must not sink the email
            logger.warning("Portfolio health: %s check failed: %s", name, e)
            health.unavailable.append(name)

    def snapshot() -> None:
        value = sum(p["value"] for p in positions.values())
        cost = sum(p["cost"] for p in positions.values() if p["value"])
        health.snapshot = {
            "positions": len(positions),
            "value": value,
            "unrealized": value - cost,
            "unrealized_pct": (value / cost - 1) * 100 if cost else 0.0,
        }

    def stops() -> None:
        for t in tickers:
            p = positions[t]
            if not p["cost"] or not p["value"]:
                continue
            pnl = (p["value"] / p["cost"] - 1) * 100
            if pnl <= STOP_LOSS_PCT + NEAR_STOP_PTS:
                health.stop_loss.append(
                    {"ticker": t, "pnl_pct": pnl, "past_stop": pnl <= STOP_LOSS_PCT}
                )
        health.stop_loss.sort(key=lambda r: r["pnl_pct"])

    def sectors() -> None:
        if sector_of is None:
            return
        by_ticker = sector_of(tickers)
        total = sum(p["value"] for p in positions.values())
        weights: dict[str, float] = {}
        for t, p in positions.items():
            weights[by_ticker.get(t) or "Unknown"] = (
                weights.get(by_ticker.get(t) or "Unknown", 0.0) + p["value"]
            )
        for sector, value in sorted(weights.items(), key=lambda kv: -kv[1]):
            pct = value / total * 100 if total else 0.0
            health.sectors.append(
                {"sector": sector, "pct": pct, "over": sector != "Unknown" and pct > max_sector_pct}
            )

    def thesis() -> None:
        if held_thesis_checks is not None:
            health.thesis = [c for c in held_thesis_checks(set(tickers)) if c["status"] != "INTACT"]

    def harvesting() -> None:
        if harvest is not None:
            health.harvest = harvest()

    def upcoming() -> None:
        if earnings is not None:
            health.earnings = sorted(earnings(tickers).values(), key=lambda e: e["days_until"])

    attempt("snapshot", snapshot)
    attempt("stop-loss", stops)
    attempt("sector weights", sectors)
    attempt("thesis check", thesis)
    attempt("tax-loss harvesting", harvesting)
    attempt("earnings calendar", upcoming)
    return health


# --- rendering -------------------------------------------------------------------

_BADGE = {
    "BROKEN": ("#9c1010", "#fde4e4"),
    "TARGET HIT": ("#0e6432", "#e6f4ea"),
    "WATCH": ("#8a4a00", "#fff4e0"),
    "PAST STOP": ("#9c1010", "#fde4e4"),
    "NEAR STOP": ("#8a4a00", "#fff4e0"),
    "OVER CAP": ("#8a4a00", "#fff4e0"),
}


def _badge(label: str) -> str:
    fg, bg = _BADGE.get(label, ("#374151", "#f3f4f6"))
    return (
        f'<span style="display:inline-block;padding:1px 8px;border-radius:999px;'
        f'font-size:11px;font-weight:600;color:{fg};background:{bg}">{html.escape(label)}</span>'
    )


def _table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f'<th style="width:auto">{html.escape(h)}</th>' for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="health"><tr>{head}</tr>{body}</table>'


def _money(v: float) -> str:
    return f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}"


def render_health_html(h: PortfolioHealth) -> str:
    parts = ['<section class="health"><h2>Portfolio health</h2>']
    s = h.snapshot
    if s:
        color = "#0e6432" if s["unrealized"] >= 0 else "#9c1010"
        parts.append(
            '<p class="health-strip">'
            f"<b>{_money(s['value'])}</b> across {int(s['positions'])} positions · unrealized "
            f'<b style="color:{color}">{_money(s["unrealized"])} ({s["unrealized_pct"]:+.1f}%)</b></p>'
        )

    alerts = 0
    if h.stop_loss:
        alerts += len(h.stop_loss)
        parts.append("<h3>Stop-loss watch (−20% from cost)</h3>")
        parts.append(
            _table(
                ["", "Ticker", "From cost"],
                [
                    [
                        _badge("PAST STOP" if r["past_stop"] else "NEAR STOP"),
                        html.escape(r["ticker"]),
                        f"{r['pnl_pct']:+.1f}%",
                    ]
                    for r in h.stop_loss
                ],
            )
        )
    if h.thesis:
        alerts += len(h.thesis)
        parts.append("<h3>Held former picks: thesis check</h3>")
        parts.append(
            _table(
                ["", "Ticker", "Since pick", "vs SPY", "Why"],
                [
                    [
                        _badge(c["status"]),
                        html.escape(c["ticker"]),
                        f"{c['return_pct']:+.1f}%",
                        f"{c['excess_pct']:+.1f} pts" if c.get("excess_pct") is not None else "—",
                        html.escape(
                            "; ".join(x["text"] for x in c["signals"] if x["severity"] != "info")
                        ),
                    ]
                    for c in h.thesis
                ],
            )
        )
    over = [r for r in h.sectors if r["over"]]
    if over:
        alerts += len(over)
        parts.append("<h3>Sector weight above the cap</h3>")
        parts.append(
            _table(
                ["", "Sector", "Share of holdings"],
                [[_badge("OVER CAP"), html.escape(r["sector"]), f"{r['pct']:.1f}%"] for r in over],
            )
        )
    if h.harvest:
        alerts += len(h.harvest)
        parts.append("<h3>Tax-loss harvesting candidates</h3>")
        parts.append(
            _table(
                ["Ticker", "Account", "Loss", "Est. tax saving", "Note"],
                [
                    [
                        html.escape(c["ticker"]),
                        html.escape(c["account"]),
                        f"{_money(c['loss_usd'])} ({c['loss_pct']:+.1f}%)",
                        f"~{_money(c['est_tax_saving_usd'])}",
                        html.escape(
                            f"bought within 30 days — a loss sale before {c['wash_sale_until']} "
                            "may be a wash sale"
                            if c.get("wash_sale_until")
                            else f"no rebuy until {c['rebuy_ok_after']}"
                        ),
                    ]
                    for c in h.harvest
                ],
            )
        )
    if h.earnings:
        parts.append("<h3>Earnings in the next 7 days</h3>")
        parts.append(
            _table(
                ["Ticker", "Date", "In"],
                [
                    [
                        html.escape(e["ticker"]),
                        html.escape(e["earnings_date"]),
                        f"{e['days_until']}d",
                    ]
                    for e in h.earnings
                ],
            )
        )
    if not alerts:
        parts.append("<p>No stop-loss, thesis, sector or tax-loss alerts today.</p>")
    top = [r for r in h.sectors if r["sector"] != "Unknown"][:3]
    if top:
        parts.append(
            '<p style="font-size:13px;color:#6b7280">Largest sectors: '
            + ", ".join(f"{html.escape(r['sector'])} {r['pct']:.0f}%" for r in top)
            + "</p>"
        )
    if h.unavailable:
        parts.append(
            '<p style="font-size:13px;color:#6b7280">Unavailable today: '
            + html.escape(", ".join(h.unavailable))
            + "</p>"
        )
    parts.append("</section>")
    return "".join(parts)


# --- the short list ---------------------------------------------------------------

MAX_DECISIONS = 6


def decision_items(h: PortfolioHealth) -> list[dict[str, Any]]:
    """Everything above, reduced to one line per decision, most urgent first
    (priority 1 = act today). The email leads with the top MAX_DECISIONS."""
    items: list[dict[str, Any]] = []

    def add(priority: int, ticker: str | None, label: str, text: str) -> None:
        items.append({"priority": priority, "ticker": ticker, "label": label, "text": text})

    for r in h.stop_loss:
        if r["past_stop"]:
            add(
                1,
                r["ticker"],
                "PAST STOP",
                f"Review {r['ticker']}: {r['pnl_pct']:+.1f}% from cost, past the −20% stop "
                f"(the rebalance rule would trim 25%).",
            )
        else:
            add(
                3,
                r["ticker"],
                "NEAR STOP",
                f"Watch {r['ticker']}: {r['pnl_pct']:+.1f}% from cost, "
                f"{r['pnl_pct'] - STOP_LOSS_PCT:.1f} pts above the −20% stop.",
            )
    for c in h.thesis:
        reason = next((s["text"] for s in c["signals"] if s["severity"] != "info"), "")
        if c["status"] == "BROKEN":
            add(1, c["ticker"], "BROKEN", f"Review {c['ticker']}: thesis broken — {reason}.")
        elif c["status"] == "TARGET HIT":
            add(
                2,
                c["ticker"],
                "TARGET HIT",
                f"Consider taking profit on {c['ticker']}: {c['return_pct']:+.1f}% since the pick, "
                f"past its bull-case target.",
            )
        else:
            add(4, c["ticker"], "WATCH", f"Keep an eye on {c['ticker']}: {reason}.")
    for e in h.earnings:
        add(
            2 if e["days_until"] <= 2 else 4,
            e["ticker"],
            "EARNINGS",
            f"{e['ticker']} reports {e['earnings_date']} (in {e['days_until']}d) — decide "
            f"before the print whether to hold through it.",
        )
    for c in h.harvest:
        wash = (
            f"; recent purchase — a loss sale before {c['wash_sale_until']} may be a wash sale"
            if c.get("wash_sale_until")
            else ""
        )
        add(
            3,
            c["ticker"],
            "TAX LOSS",
            f"Tax-loss option: selling {c['ticker']} in {c['account']} realizes "
            f"{_money(c['loss_usd'])} (~{_money(c['est_tax_saving_usd'])} tax saved){wash}.",
        )
    for r in h.sectors:
        if r["over"]:
            add(
                4,
                None,
                "OVER CAP",
                f"Don't add to {r['sector']}: already {r['pct']:.0f}% of holdings (cap 30%).",
            )
    items.sort(key=lambda i: i["priority"])
    return items


def flagged_tickers(h: PortfolioHealth) -> list[str]:
    """Tickers with a decision item, most urgent first (for ordering the
    per-stock sections of the email)."""
    seen: list[str] = []
    for i in decision_items(h):
        if i["ticker"] and i["ticker"] not in seen:
            seen.append(i["ticker"])
    return seen


def render_decisions_html(h: PortfolioHealth) -> str:
    items = decision_items(h)
    if not items:
        return (
            '<section class="decide"><h2>Decide today</h2>'
            '<p class="health-strip">Nothing needs a decision today.</p></section>'
        )
    shown = items[:MAX_DECISIONS]
    lis = "".join(
        f'<li style="margin:6px 0">{_badge(i["label"])} {html.escape(i["text"])}</li>'
        for i in shown
    )
    more = (
        f'<p style="font-size:13px;color:#6b7280">+{len(items) - len(shown)} lower-priority '
        f"item(s) in Portfolio health below.</p>"
        if len(items) > len(shown)
        else ""
    )
    return f'<section class="decide"><h2>Decide today</h2><ol style="padding-left:20px">{lis}</ol>{more}</section>'


# Priority 4 items (thesis on watch, sector already over the cap) are
# standing guidance rather than something to act on today.
ACTION_PRIORITY = 3


def decision_count(h: PortfolioHealth) -> int:
    """How many items need action (priority 1-3) — the number in the subject."""
    return sum(1 for i in decision_items(h) if i["priority"] <= ACTION_PRIORITY)
