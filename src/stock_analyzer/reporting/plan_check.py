"""HTML for the plan check: asset location and the goal projection.

Decision first, like the other emails: each section opens with the one
line that says whether anything needs doing, then the table behind it.
"""

from __future__ import annotations

import html
from datetime import date

from ..discover.asset_location import AssetLocationReport
from ..discover.goal_projection import DEEP_DRAWDOWN, TARGET_ODDS, Projection
from .health import _badge, _money, _table

_KIND_LABEL = {"taxable": "taxable", "tax_deferred": "tax-deferred", "tax_free": "tax-free"}
_NOTE = 'style="font-size:13px;color:#6b7280"'


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _odds_badge(odds: float) -> str:
    if odds >= TARGET_ODDS:
        return _badge("ON TRACK")
    if odds >= 0.5:
        return _badge("AT RISK")
    return _badge("OFF TRACK")


def goal_headline(p: Projection | None, goal_date: date | None) -> str:
    if p is None:
        return "Goal projection unavailable (not enough price history)"
    when = f"by {goal_date:%b %Y}" if goal_date else f"in {p.months / 12:.0f} years"
    if p.odds is None:
        return f"Likely worth {_money(p.p50)} {when} (middle case)"
    return f"{_pct(p.odds)} odds of {_money(p.target or 0)} {when}"


def render_goal_html(
    p: Projection | None,
    *,
    goal_date: date | None,
    contribution_note: str,
    left_out: list[str] | tuple[str, ...] = (),
) -> str:
    if p is None:
        return (
            '<section class="health"><h2>Goal projection</h2><p>Not enough monthly price '
            "history to project from.</p></section>"
        )
    when = f"{goal_date:%b %Y}" if goal_date else f"{p.months / 12:.0f} years from now"
    parts = ['<section class="health"><h2>Goal projection</h2>']
    if p.odds is not None and p.target:
        gap = ""
        if p.odds < TARGET_ODDS and p.needed_contribution is not None:
            gap = (
                f" Adding <b>{_money(p.needed_contribution)}/month</b> (now "
                f"{_money(p.monthly_contribution)}) would make it {_pct(TARGET_ODDS)}."
            )
        parts.append(
            f"<p>{_odds_badge(p.odds)} <b>{_pct(p.odds)}</b> odds of reaching "
            f"<b>{_money(p.target)}</b> by {html.escape(when)}.{gap}</p>"
        )
    else:
        parts.append(
            f"<p>No target set (GOAL_TARGET_USD), so no odds — here is the likely range "
            f"{html.escape(when)}.</p>"
        )
    rows = [
        ["Bad case (1 in 10 worse)", _money(p.p10)],
        ["Middle case", _money(p.p50)],
        ["Good case (1 in 10 better)", _money(p.p90)],
        ["Same money in SPY, bad case", _money(p.spy_p10 or 0)],
        ["Same money in SPY, middle case", _money(p.spy_p50 or 0)],
    ]
    if p.spy_odds is not None:
        rows.append(["Same money in SPY, odds of the target", _pct(p.spy_odds)])
    rows.append(
        [f"Odds of a {_pct(DEEP_DRAWDOWN)}+ fall along the way", _pct(p.deep_drawdown_odds)]
    )
    parts.append(_table(["", "Value"], rows))
    concentration = (
        f"Your holdings swing {_pct(p.annual_volatility)} a year against SPY's "
        f"{_pct(p.spy_annual_volatility)}"
    )
    if p.spy_odds is not None and p.odds is not None:
        diff = p.odds - p.spy_odds
        concentration += (
            f". At the same average return that gives {abs(diff) * 100:.0f} points "
            f"{'better' if diff >= 0 else 'worse'} odds of the target than SPY"
        )
        if p.target and p.target > p.p50 and diff > 0:
            # A wider spread reaches a stretch target more often and falls
            # further in the bad case; saying only the first is a sales pitch.
            concentration += (
                ", because a stretch target needs a good outcome and a wider spread "
                "produces more of them; it produces more bad ones too, as the bad cases show"
            )
    parts.append(f"<p>{html.escape(concentration)}.</p>")
    filled = (
        f" Too young for the full window, so SPY's months stand in: "
        f"{html.escape(', '.join(p.filled_from_spy))}."
        if p.filled_from_spy
        else ""
    )
    if left_out:
        filled += f" Treated as cash (no price history, or a stable $1 fund) and left out: {html.escape(', '.join(left_out))}."
    parts.append(
        f"<p {_NOTE}>Starting from {_money(p.start_value)}, adding "
        f"{_money(p.monthly_contribution)}/month ({html.escape(contribution_note)}), over "
        f"{p.months} months. 10,000 futures built from 12-month blocks of your current "
        f"holdings' last {p.history_months} months, each month shifted to average "
        f"{_pct(p.expected_return)} a year (GOAL_EXPECTED_RETURN) — past returns of stocks "
        f"held because they rose are not a forecast; their swings are kept. Cash is assumed "
        f"invested like the rest. Nominal dollars.{filled}</p></section>"
    )
    return "".join(parts)


def asset_location_headline(r: AssetLocationReport | None) -> str:
    if r is None:
        return "Asset location unavailable"
    if r.swaps:
        saving = sum(s.yearly_saving for s in r.swaps)
        return f"{len(r.swaps)} account swap(s) would save ~{_money(saving)}/yr in tax"
    return f"Holdings are well placed (~{_money(r.taxable_drag)}/yr tax drag in taxable)"


def render_asset_location_html(r: AssetLocationReport | None) -> str:
    if r is None:
        return ""
    parts = ['<section class="health"><h2>Asset location</h2>']
    if not any(k == "taxable" for k in r.kinds.values()) or not any(
        k != "taxable" for k in r.kinds.values()
    ):
        parts.append(
            "<p>All accounts have the same tax treatment, so there is nothing to place.</p>"
            "</section>"
        )
        return "".join(parts)
    parts.append(
        f"<p>Holdings in taxable accounts cost about <b>{_money(r.taxable_drag)} a year</b> "
        f"in tax on dividends and option premium.</p>"
    )
    if r.swaps:
        rows = []
        for s in r.swaps:
            wash = (
                f"<br><span style='color:#9c1010'>Sold at a loss: do not buy "
                f"{html.escape(s.inefficient)} in {html.escape(s.into_account)} before "
                f"{s.wash_sale_until:%b %d} — a purchase in an IRA within 30 days loses the "
                f"loss for good.</span>"
                if s.wash_sale_until
                else ""
            )
            rows.append(
                [
                    f"<b>{_money(s.amount)}</b>",
                    f"In {html.escape(s.taxable_account)}: sell {html.escape(s.inefficient)}, "
                    f"buy {html.escape(s.efficient)}.<br>In {html.escape(s.into_account)}: sell "
                    f"{html.escape(s.efficient)}, buy {html.escape(s.inefficient)}.{wash}",
                    _money(s.yearly_saving) + "/yr",
                    _money(s.tax_on_sale),
                    "now" if not s.breakeven_years else f"{s.breakeven_years:.1f} yr",
                ]
            )
        parts.append(
            "<p><b>Swaps worth making</b> — the portfolio holds the same stocks after:</p>"
        )
        parts.append(_table(["Amount", "Trades", "Tax saved", "Tax to swap", "Pays back"], rows))
        parts.append(
            f"<p {_NOTE}>Tax to swap assumes long-term lots at your long-term rate; check "
            "lot dates before selling. IRA trades cost no tax.</p>"
        )
    else:
        parts.append(
            "<p>No swap pays for its own tax on sale within the payback limit — leave "
            "things where they are.</p>"
        )
    for ticker, (premium, account) in sorted(r.options_elsewhere.items(), key=lambda kv: -kv[1][0]):
        parts.append(
            f"<p>{_badge('TAX')} {_money(premium)} of premium was written on "
            f"{html.escape(ticker)} in a taxable account in the last year, taxed as income. "
            f"{html.escape(account)} holds 100+ {html.escape(ticker)} shares — calls written "
            f"there are not taxed.</p>"
        )
    rows = [
        [
            html.escape(p.ticker),
            html.escape(f"{p.account} ({_KIND_LABEL.get(p.kind, p.kind)})"),
            _money(p.value),
            f"{p.dividend_yield * 100:.1f}%" + (" (REIT)" if p.ordinary_dividends else ""),
            _money(p.option_premium) if p.option_premium else "—",
            _money(p.drag_if_taxable) + "/yr",
        ]
        for p in r.placements
        if p.drag_if_taxable >= 50 or p.kind == "taxable"
    ]
    if rows:
        parts.append(
            f"<p {_NOTE}>Tax each holding costs (or would cost) a year in a taxable account. "
            "Best placed: the costly ones in tax-deferred or tax-free accounts, the cheap "
            "ones in taxable.</p>"
        )
        parts.append(
            _table(["Ticker", "Account", "Value", "Yield", "Premium (1y)", "Tax if taxable"], rows)
        )
    parts.append(
        f"<p {_NOTE}>New money: income stocks and option writing go in the IRA, the "
        "highest-growth ideas in the HSA or a Roth, low-yield long-term holds in taxable.</p>"
        "</section>"
    )
    return "".join(parts)
