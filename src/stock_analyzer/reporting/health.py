"""The "Portfolio health" block — the deterministic top of the daily email.

No LLM calls. It reuses the checks the discover/rebalance pipelines run,
applied to today's holdings, so the daily email flags what needs attention
between rebalance runs:

  - snapshot: market value and unrealized P/L;
  - drawdown review: positions down 20%+ from cost. Holdings are long-term
    (3-5 year) investments, so this asks for a thesis re-check, not a sale
    (same threshold as discover/rebalance_holdings.flag_drawdown_reviews);
  - thesis check: holdings that were recent discover picks, re-checked
    against their own targets and trend (discover/thesis_tracker.py);
  - sector weight: any sector above the Sizer's book cap
    (DISCOVER_MAX_SECTOR_PCT);
  - tax-loss harvesting: taxable slices past the HARVEST_* thresholds
    (discover/tax_harvest.py; swaps are in the rebalance report, which has
    the peer data);
  - earnings in the next 7 days, and results of the last 7 days: the
    estimate direction since the report says whether the long-term case
    changed (discover/post_earnings.py);
  - dividend income: forward annual income and yield, and what the last
    12 months paid — reinvested automatically or left as cash
    (discover/income.py);
  - add on weakness: holdings 15%+ below their 52-week high with the
    long-term case intact — candidates for new money (discover/add_on.py);
  - reinvestment ideas: whenever a line suggests selling, it names where
    the money could go — a recent discover pick not held, outside any
    over-cap sector (discover/reinvest.py), or for a tax-loss sale a
    same-sector swap that keeps the exposure.

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

DRAWDOWN_REVIEW_PCT = -20.0  # discover/rebalance_holdings.flag_drawdown_reviews
EARNINGS_DAYS = 7


@dataclass
class PortfolioHealth:
    snapshot: dict[str, float] = field(default_factory=dict)
    drawdowns: list[dict[str, Any]] = field(default_factory=list)
    thesis: list[dict[str, Any]] = field(default_factory=list)
    sectors: list[dict[str, Any]] = field(default_factory=list)
    harvest: list[dict[str, Any]] = field(default_factory=list)
    earnings: list[dict[str, Any]] = field(default_factory=list)
    reinvest: list[dict[str, Any]] = field(default_factory=list)
    income: dict[str, Any] = field(default_factory=dict)
    earnings_results: list[dict[str, Any]] = field(default_factory=list)
    add_on: list[dict[str, Any]] = field(default_factory=list)
    sector_by_ticker: dict[str, str] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)  # ticker -> market value
    units: dict[str, float] = field(default_factory=dict)  # ticker -> shares held
    max_sector_pct: float = 30.0
    unavailable: list[str] = field(default_factory=list)
    # Data-quality notes (a stale account price, a missing cost basis):
    # things that make the numbers above worth a second look.
    data_notes: list[str] = field(default_factory=list)
    # Short calls already written against these holdings
    # (data/brokerage.fetch_covered_call_obligations). Shares backing a
    # call are promised: selling them turns the call naked, so no sale
    # suggestion here is free.
    covered_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Overnight and trailing moves on the exchanges that price this
    # portfolio's demand (data/world_markets.py). Context, never a
    # decision: these are 3-5 year holdings.
    world_markets: list[dict[str, Any]] = field(default_factory=list)
    # Accounts the broker has stopped syncing. Not a footnote like the
    # notes above — until the connection is restored every number for
    # that account describes the day it went dark, so it leads the email.
    stale_accounts: list[str] = field(default_factory=list)


def aggregate_positions(
    holdings: dict[str, list[dict[str, Any]]],
    prices: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """{ticker: units, cost, value} across accounts, from brokerage rows.

    `prices` (data/pricing.py) values every account's slice of a ticker at
    the same price; without it each account's own — possibly stale — price
    is used, which is how one holding ended up worth two different amounts
    in the same email.
    """
    prices = prices or {}
    out: dict[str, dict[str, float]] = {}
    for items in holdings.values():
        for h in items:
            ticker = h.get("ticker")
            units = float(h.get("units") or 0)
            if not ticker or not units:
                continue
            price = prices.get(str(ticker).upper()) or float(h.get("price") or 0)
            row = out.setdefault(ticker, {"units": 0.0, "cost": 0.0, "value": 0.0})
            row["units"] += units
            row["cost"] += units * float(h.get("average_purchase_price") or 0)
            row["value"] += units * price
    return out


def build_portfolio_health(
    holdings: dict[str, list[dict[str, Any]]],
    *,
    prices: dict[str, float] | None = None,
    data_notes: list[str] | None = None,
    stale_accounts: list[str] | None = None,
    covered_calls: dict[str, dict[str, Any]] | None = None,
    world_markets: list[dict[str, Any]] | None = None,
    max_sector_pct: float = 30.0,
    sector_of: Callable[[list[str]], dict[str, str]] | None = None,
    held_thesis_checks: Callable[[set[str]], list[dict[str, Any]]] | None = None,
    harvest: Callable[[], list[dict[str, Any]]] | None = None,
    earnings: Callable[[list[str]], dict[str, dict[str, Any]]] | None = None,
    reinvest: Callable[[set[str], set[str], int], list[dict[str, Any]]] | None = None,
    income: Callable[[dict[str, float], dict[str, float]], dict[str, Any]] | None = None,
    add_on: Callable[..., list[dict[str, Any]]] | None = None,
    earnings_results: Callable[[], list[dict[str, Any]]] | None = None,
) -> PortfolioHealth:
    """`reinvest(held, over_cap_sectors, n)` returns up to `n` ranked ideas
    for sale proceeds; it is only called when something suggests a sale."""
    health = PortfolioHealth(max_sector_pct=max_sector_pct)
    health.data_notes.extend(data_notes or [])
    health.stale_accounts.extend(stale_accounts or [])
    health.world_markets.extend(world_markets or [])
    health.covered_calls.update(covered_calls or {})
    positions = aggregate_positions(holdings, prices)
    tickers = sorted(positions)

    def attempt(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — one check must not sink the email
            logger.warning("Portfolio health: %s check failed: %s", name, e)
            health.unavailable.append(name)

    def snapshot() -> None:
        value = sum(p["value"] for p in positions.values())
        # Unrealized is measured only over positions that have BOTH a value
        # and a cost basis. Summing all the value against only the known
        # cost counted a position with no cost basis as pure profit.
        priced = [p for p in positions.values() if p["value"] and p["cost"]]
        cost = sum(p["cost"] for p in priced)
        matched = sum(p["value"] for p in priced)
        no_basis = [t for t, p in positions.items() if p["value"] and not p["cost"]]
        if no_basis:
            health.data_notes.append(
                "no cost basis for " + ", ".join(sorted(no_basis)) + " — left out of unrealized P/L"
            )
        health.values = {t: p["value"] for t, p in positions.items()}
        health.units = {t: p["units"] for t, p in positions.items()}
        health.snapshot = {
            "positions": len(positions),
            "value": value,
            "unrealized": matched - cost,
            "unrealized_pct": (matched / cost - 1) * 100 if cost else 0.0,
        }

    def drawdowns() -> None:
        for t in tickers:
            p = positions[t]
            if not p["cost"] or not p["value"]:
                continue
            pnl = (p["value"] / p["cost"] - 1) * 100
            if pnl <= DRAWDOWN_REVIEW_PCT:
                health.drawdowns.append({"ticker": t, "pnl_pct": pnl, "value": p["value"]})
        health.drawdowns.sort(key=lambda r: r["pnl_pct"])

    def sectors() -> None:
        if sector_of is None:
            return
        by_ticker = sector_of(tickers)
        health.sector_by_ticker = dict(by_ticker)
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
    attempt("drawdown review", drawdowns)
    attempt("sector weights", sectors)
    attempt("thesis check", thesis)
    attempt("tax-loss harvesting", harvesting)
    attempt("earnings calendar", upcoming)

    def dividends() -> None:
        if income is not None:
            health.income = income(health.units, health.values)

    def dips() -> None:
        if add_on is not None:
            health.add_on = add_on(
                values=health.values,
                sector_of=health.sector_by_ticker,
                over_cap_sectors={r["sector"] for r in health.sectors if r["over"]},
                # Anything already flagged for a thesis re-check or a loss
                # sale isn't also offered as a place for new money.
                thesis_flagged={c["ticker"] for c in health.thesis}
                | {r["ticker"] for r in health.drawdowns}
                | {c["ticker"] for c in health.harvest},
            )

    def results() -> None:
        if earnings_results is not None:
            health.earnings_results = earnings_results()

    attempt("earnings results", results)
    attempt("dividend income", dividends)
    attempt("add on weakness", dips)

    def ideas() -> None:
        sales = len(_sale_items(health)) + (1 if health.income.get("cash_12m") else 0)
        if reinvest is not None and sales:
            over = {r["sector"] for r in health.sectors if r["over"]}
            health.reinvest = reinvest(set(tickers), over, min(sales, 3))

    attempt("reinvestment ideas", ideas)
    return health


def _sale_items(h: PortfolioHealth) -> list[str]:
    """Tickers the email suggests selling (or may, after a thesis
    re-check) — each gets a destination for the money."""
    out = [c["ticker"] for c in h.thesis if c["status"] == "BROKEN"]
    out += [r["ticker"] for r in h.drawdowns if r["ticker"] not in out]
    out += [c["ticker"] for c in h.harvest if not c.get("swap_candidates")]
    return out


# --- rendering -------------------------------------------------------------------

_BADGE = {
    "BROKEN": ("#9c1010", "#fde4e4"),
    "TARGET HIT": ("#0e6432", "#e6f4ea"),
    "WATCH": ("#8a4a00", "#fff4e0"),
    "DRAWDOWN": ("#8a4a00", "#fff4e0"),
    "OVER CAP": ("#8a4a00", "#fff4e0"),
    "GOOD CALL": ("#0e6432", "#e6f4ea"),
    "MISSED": ("#9c1010", "#fde4e4"),
    "ADD ON DIP": ("#0e6432", "#e6f4ea"),
    "EARNINGS CUT": ("#9c1010", "#fde4e4"),
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
    if h.drawdowns:
        alerts += len(h.drawdowns)
        parts.append("<h3>Down 20%+ from cost: re-check the long-term thesis</h3>")
        parts.append(
            _table(
                ["", "Ticker", "From cost"],
                [
                    [_badge("DRAWDOWN"), html.escape(r["ticker"]), f"{r['pnl_pct']:+.1f}%"]
                    for r in h.drawdowns
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
    if h.add_on:
        parts.append("<h3>Add on weakness (long-term case intact)</h3>")
        parts.append(
            _table(
                ["Ticker", "Below 52-week high", "Share of portfolio", "Sector"],
                [
                    [
                        html.escape(a["ticker"]),
                        f"{a['off_high_pct']:+.1f}%",
                        f"{a['weight_pct']:.1f}%",
                        html.escape(a.get("sector") or "—"),
                    ]
                    for a in h.add_on
                ],
            )
        )
    inc = h.income
    if inc and (inc.get("forward_annual") or inc.get("received_12m")):
        parts.append("<h3>Dividend income</h3>")
        yld = f" ({inc['yield_pct']:.2f}% of holdings)" if inc.get("yield_pct") else ""
        line = (
            f"About <b>{_money(inc['forward_annual'])}/yr</b> at current rates{yld}; "
            f"{_money(inc['received_12m'])} received over the last 12 months"
        )
        if inc.get("reinvested_12m"):
            line += f", {_money(inc['reinvested_12m'])} of it reinvested automatically"
        parts.append(f"<p>{line}.</p>")
        if inc.get("cash_12m"):
            where = (
                f" — consider putting it to work in {html.escape(h.add_on[0]['ticker'])}"
                if h.add_on
                else ""
            )
            parts.append(
                f"<p>{_money(inc['cash_12m'])} of dividends arrived as cash in "
                f"{html.escape(', '.join(inc.get('cash_accounts') or []))}{where}.</p>"
            )
        payers = [r for r in inc.get("rows") or [] if r["annual"]][:5]
        if payers:
            parts.append(
                _table(
                    ["Ticker", "Per year", "Yield", "Last 12 months", "Reinvested"],
                    [
                        [
                            html.escape(r["ticker"]),
                            _money(r["annual"]),
                            f"{r['yield_pct']:.2f}%" if r.get("yield_pct") else "—",
                            _money(r["received_12m"]),
                            {True: "yes", False: "no", None: "—"}[r["reinvested"]],
                        ]
                        for r in payers
                    ],
                )
            )
    if h.reinvest:
        from ..discover.reinvest import format_idea

        parts.append("<h3>Where sale proceeds could go</h3>")
        parts.append(
            "<p>Recent discover picks you don't hold, outside any over-cap sector: "
            + ", ".join(html.escape(format_idea(i)) for i in h.reinvest)
            + ".</p>"
        )
    if h.earnings_results:
        from ..discover.post_earnings import result_text

        parts.append("<h3>Earnings results (last 7 days)</h3>")
        parts.append(
            "<ul>"
            + "".join(f"<li>{html.escape(result_text(r))}</li>" for r in h.earnings_results)
            + "</ul>"
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
        parts.append("<p>No drawdown, thesis, sector or tax-loss alerts today.</p>")
    top = [r for r in h.sectors if r["sector"] != "Unknown"][:3]
    if top:
        parts.append(
            '<p style="font-size:13px;color:#6b7280">Largest sectors: '
            + ", ".join(f"{html.escape(r['sector'])} {r['pct']:.0f}%" for r in top)
            + "</p>"
        )
    parts.append(render_covered_calls_html(h))
    parts.append(render_world_markets_html(h))
    if h.stale_accounts:
        parts.append(
            '<p style="font-size:13px;color:#9c1010"><b>Stale account data:</b> '
            + html.escape("; ".join(h.stale_accounts))
            + "</p>"
        )
    if h.data_notes:
        parts.append(
            '<p style="font-size:13px;color:#9c1010">Check the data: '
            + html.escape("; ".join(h.data_notes))
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


def render_covered_calls_html(h: PortfolioHealth) -> str:
    """What is already promised, and how close it is to being taken.

    A covered call is not visible anywhere in a holdings table — the
    shares still show as owned — so without this the report implies a
    freedom of action the position does not have."""
    if not h.covered_calls:
        return ""
    rows = []
    for ticker, rec in sorted(h.covered_calls.items()):
        units = h.units.get(ticker) or 0
        value = h.values.get(ticker)
        committed = float(rec.get("shares_committed") or 0)
        strike = rec.get("lowest_strike")
        price = (value / units) if value and units else None
        to_strike = (strike / price - 1) * 100 if strike and price else None
        share = min(committed / units, 1.0) * 100 if units else None
        colour = "#9c1010" if (to_strike is not None and to_strike <= ASSIGNMENT_WATCH_PCT) else ""
        rows.append(
            [
                html.escape(ticker),
                f"{rec['contracts']}",
                f"{committed:,.0f}" + (f" ({share:.0f}%)" if share is not None else ""),
                f"${strike:,.0f}" if strike else "—",
                (
                    f'<span style="color:{colour}">{to_strike:+.0f}%</span>'
                    if to_strike is not None
                    else "—"
                ),
                html.escape(str(rec.get("next_expiry") or "—")),
            ]
        )
    parts = ["<h3>Covered calls written</h3>"]
    parts.append(
        _table(["Ticker", "Calls", "Shares promised", "Strike", "To strike", "Expiry"], rows)
    )
    parts.append(
        '<p style="font-size:13px;color:#6b7280">Shares backing a call cannot be sold '
        "without buying it back first. A position at or above its strike gets called "
        "away at expiry.</p>"
    )
    return "".join(parts)


def render_world_markets_html(h: PortfolioHealth) -> str:
    """The exchanges that traded before New York, and what they say about
    what is held. Trailing columns lead; the overnight move is last,
    because one session is not a reason to touch a 3-5 year position."""
    rows = h.world_markets
    if not rows:
        return ""
    from ..data.world_markets import world_signals

    def cell(row: dict[str, Any], window: str) -> str:
        value = row.get(window)
        if value is None:
            return "—"
        color = "#166534" if value > 0 else "#9c1010" if value < 0 else "#6b7280"
        return f'<span style="color:{color}">{value:+.1f}%</span>'

    parts = ["<h3>World markets</h3>"]
    parts.append(
        _table(
            ["Market", "Region", "6mo", "1y", "1mo", "1d"],
            [
                [
                    html.escape(str(r["name"])),
                    html.escape(str(r["region"])),
                    cell(r, "6mo"),
                    cell(r, "1y"),
                    cell(r, "1mo"),
                    cell(r, "1d"),
                ]
                for r in rows
            ],
        )
    )
    signals = world_signals(rows, set(h.values))
    for line in signals.holdings_context:
        parts.append(f'<p style="font-size:13px;color:#374151">{html.escape(line)}</p>')
    if signals.regime_breaks:
        parts.append(
            '<p style="font-size:13px;color:#9c1010">Down more than 10% over the year: '
            + html.escape("; ".join(signals.regime_breaks))
            + "</p>"
        )
    return "".join(parts)


# --- the short list ---------------------------------------------------------------

MAX_DECISIONS = 6


# Within this much of the strike, assignment stops being hypothetical
# and the shares are likely to be called away at expiry.
ASSIGNMENT_WATCH_PCT = 15.0


def covered_call_clause(h: PortfolioHealth, ticker: str) -> str:
    """What an open short call adds to a decision to sell `ticker`.

    Empty when nothing is written against it. Otherwise it says how much
    of the position is promised and what closing it would take, because
    "sell NVDA" is a different instruction when all 401 shares back four
    calls.
    """
    rec = h.covered_calls.get(ticker)
    if not rec or not rec.get("contracts"):
        return ""
    committed = float(rec.get("shares_committed") or 0)
    held = float(h.units.get(ticker) or 0)
    share = f"{min(committed / held, 1.0) * 100:.0f}% of" if held else ""
    strike = rec.get("lowest_strike")
    where = f" at ${strike:,.0f}" if strike else ""
    expiry = rec.get("next_expiry")
    until = f" through {expiry}" if expiry else ""
    return (
        f" Note {rec['contracts']} covered call(s) on {ticker}: {committed:,.0f} shares "
        f"({share} the position){where}{until} are already promised, so selling means "
        f"buying those back first or waiting for assignment."
    )


def assignment_items(h: PortfolioHealth) -> list[dict[str, Any]]:
    """Positions close enough to a written strike to be called away.

    A 3-5 year holder's risk here is not a loss, it is losing the
    position: assignment ends the compounding and, in a taxable account,
    realizes the gain on someone else's schedule.
    """
    out = []
    for ticker, rec in sorted(h.covered_calls.items()):
        strike = rec.get("lowest_strike")
        units = h.units.get(ticker) or 0
        value = h.values.get(ticker)
        if not strike or not units or not value:
            continue
        price = value / units
        to_strike = (strike / price - 1) * 100
        if to_strike > ASSIGNMENT_WATCH_PCT or to_strike < 0:
            continue
        committed = float(rec.get("shares_committed") or 0)
        out.append(
            {
                "ticker": ticker,
                "to_strike_pct": to_strike,
                "text": (
                    f"{ticker} is {to_strike:.0f}% below your ${strike:,.0f} strike expiring "
                    f"{rec.get('next_expiry')}: a rally through it calls away {committed:,.0f} "
                    f"shares. Roll the call up or out if you mean to keep them."
                ),
            }
        )
    return out


def decision_items(h: PortfolioHealth) -> list[dict[str, Any]]:
    """Everything above, reduced to one line per decision, most urgent first
    (priority 1 = act today). The email leads with the top MAX_DECISIONS.

    Holdings are long-term (3-5 year) investments: only a broken business
    thesis is a reason to sell. Price drops ask for a thesis re-check, and
    upcoming earnings or a pick past its bull case are information, not
    something to trade on."""
    items: list[dict[str, Any]] = []

    def add(
        priority: int, ticker: str | None, label: str, text: str, reinvest_into: str | None = None
    ) -> None:
        items.append(
            {
                "priority": priority,
                "ticker": ticker,
                "label": label,
                "text": text,
                "reinvest_into": reinvest_into,
            }
        )

    from ..discover.reinvest import format_idea

    # Each sale gets its own idea while they last, then they repeat.
    dest = (
        {t: h.reinvest[i % len(h.reinvest)] for i, t in enumerate(_sale_items(h))}
        if (h.reinvest)
        else {}
    )

    def dest_ticker(ticker: str) -> str | None:
        idea = dest.get(ticker)
        return idea["ticker"] if idea else None

    def proceeds(ticker: str, amount: float | None, *, conditional: bool = False) -> str:
        idea = dest.get(ticker)
        if idea is None:
            return ""
        money = f"the ~{_money(amount)}" if amount else "the proceeds"
        lead = " If you do sell, reinvest" if conditional else " Reinvest"
        return f"{lead} {money} in {format_idea(idea)}."

    for note in h.stale_accounts:
        add(1, None, "STALE DATA", f"Reconnect the account: {note}.")
    for item in assignment_items(h):
        add(3, item["ticker"], "CALL ASSIGNMENT", item["text"])
    for r in h.drawdowns:
        add(
            2,
            r["ticker"],
            "DRAWDOWN",
            f"Re-check the long-term thesis for {r['ticker']}: {r['pnl_pct']:+.1f}% from cost. "
            f"A lower price alone isn't a reason to sell — sell only if the business case "
            f"has broken."
            + covered_call_clause(h, r["ticker"])
            + proceeds(r["ticker"], r.get("value"), conditional=True),
            dest_ticker(r["ticker"]),
        )
    for c in h.thesis:
        reason = next((s["text"] for s in c["signals"] if s["severity"] != "info"), "")
        if c["status"] == "BROKEN":
            add(
                1,
                c["ticker"],
                "BROKEN",
                f"Consider selling {c['ticker']}: long-term thesis broken — {reason}."
                + covered_call_clause(h, c["ticker"])
                + proceeds(c["ticker"], h.values.get(c["ticker"])),
                dest_ticker(c["ticker"]),
            )
        elif c["status"] == "TARGET HIT":
            add(
                4,
                c["ticker"],
                "TARGET HIT",
                f"{c['ticker']} is {c['return_pct']:+.1f}% since the pick, past its bull case — "
                f"re-check the valuation; trim only if it has grown too large a share."
                + covered_call_clause(h, c["ticker"]),
            )
        else:
            add(4, c["ticker"], "WATCH", f"Keep an eye on {c['ticker']}: {reason}.")
    from ..discover.post_earnings import result_text

    for r in h.earnings_results:
        lowering = r["direction"] == "lowering"
        add(
            2 if lowering else 4,
            r["ticker"],
            "EARNINGS CUT" if lowering else "EARNINGS",
            result_text(r),
        )
    for e in h.earnings:
        add(
            4,
            e["ticker"],
            "EARNINGS",
            f"{e['ticker']} reports {e['earnings_date']} (in {e['days_until']}d) — nothing to "
            f"do before the print; check the results against the long-term thesis.",
        )
    for c in h.harvest:
        wash = (
            f"; recent purchase — a loss sale before {c['wash_sale_until']} may be a wash sale"
            if c.get("wash_sale_until")
            else ""
        )
        slice_value = (c.get("units") or 0) * (c.get("price") or 0)
        swaps = c.get("swap_candidates") or []
        if swaps:
            where = (
                f" To keep similar exposure, buy {swaps[0]} (same sector, not the same "
                f"stock) with the ~{_money(slice_value)}."
            )
        else:
            where = proceeds(c["ticker"], slice_value)
        add(
            3,
            c["ticker"],
            "TAX LOSS",
            f"Tax-loss option: selling {c['ticker']} in {c['account']} realizes "
            f"{_money(c['loss_usd'])} (~{_money(c['est_tax_saving_usd'])} tax saved){wash}."
            + covered_call_clause(h, c["ticker"])
            + where,
            swaps[0] if swaps else dest_ticker(c["ticker"]),
        )
    for a in h.add_on:
        add(
            4,
            a["ticker"],
            "ADD ON DIP",
            f"{a['ticker']} is {-a['off_high_pct']:.0f}% below its 52-week high with its "
            f"long-term case intact — a candidate for new money.",
        )
    for r in h.sectors:
        if r["over"]:
            add(
                4,
                None,
                "OVER CAP",
                f"Don't add to {r['sector']}: already {r['pct']:.0f}% of holdings "
                f"(cap {h.max_sector_pct:.0f}%).",
            )
    items.sort(key=lambda i: i["priority"])
    return items


# Decision labels that are advice worth grading later, as ledger actions.
_LEDGER_ACTIONS = {
    "BROKEN": "SELL",
    "CALL ASSIGNMENT": "REVIEW",
    "DRAWDOWN": "REVIEW",
    "TAX LOSS": "TAX_LOSS",
    "EARNINGS CUT": "REVIEW",
}


def suggestion_rows(h: PortfolioHealth, *, today: str) -> list[dict[str, Any]]:
    """Today's actionable decision lines as `suggestions` rows, for the
    quarterly review to grade."""
    rows = []
    for i in decision_items(h):
        action = _LEDGER_ACTIONS.get(i["label"])
        t = i["ticker"]
        if action is None or not t:
            continue
        units = h.units.get(t)
        rows.append(
            {
                "suggested_on": today,
                "source": "daily",
                "action": action,
                "ticker": t,
                "detail": i["text"],
                "price": h.values[t] / units if units and t in h.values else None,
                "units_held": units,
                "reinvest_into": i["reinvest_into"],
            }
        )
    return rows


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


# Priority 4 items (thesis on watch, past the bull case, earnings coming,
# sector already over the cap) are information rather than something to
# act on today.
ACTION_PRIORITY = 3


def decision_count(h: PortfolioHealth) -> int:
    """How many items need action (priority 1-3) — the number in the subject."""
    return sum(1 for i in decision_items(h) if i["priority"] <= ACTION_PRIORITY)
