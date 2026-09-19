"""HTML for the December tax-planner email (discover/tax_planner.py)."""

from __future__ import annotations

import html
from datetime import date
from typing import Any

from .health import _money, _table


def render_tax_plan_html(
    *,
    year: int,
    taxable_accounts: list[str],
    realized: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    harvest: list[dict[str, Any]],
    soon: list[dict[str, Any]],
    last_day: date,
) -> str:
    from .html import _wrap_html

    esc = html.escape
    parts = [
        f"<p>Taxable accounts: {esc(', '.join(taxable_accounts) or 'none found')}. "
        f"Losses must be realized by <b>{last_day:%a %b %d}</b> (the year's last trading "
        "day) to count for this year. Estimates only — check them with your tax "
        "professional.</p>"
    ]

    parts.append("<h2>Realized so far this year (estimate)</h2>")
    s = summary
    parts.append(
        f"<p>Short-term <b>{_money(s['short_term'])}</b> · long-term "
        f"<b>{_money(s['long_term'])}</b> · net <b>{_money(s['net_gain'])}</b></p>"
    )
    sells = [r | {"account": a} for a, rec in realized.items() for r in rec["rows"]]
    if sells:
        parts.append(
            _table(
                ["Date", "Account", "Ticker", "Shares", "Proceeds", "Short-term", "Long-term"],
                [
                    [
                        f"{r['date']:%b %d}",
                        esc(r["account"]),
                        esc(r["ticker"]),
                        f"{r['units']:g}",
                        _money(r["proceeds"]),
                        _money(r["short_term"]),
                        _money(r["long_term"]),
                    ]
                    for r in sells
                ],
            )
        )
    options = sum(rec.get("options") or 0.0 for rec in realized.values())
    if options:
        parts.append(
            f"<p>Includes {_money(options)} net from option contracts closed this year "
            "(counted as short-term).</p>"
        )
    if s["basis_unknown_units"]:
        parts.append(
            f'<p style="font-size:13px;color:#6b7280">{s["basis_unknown_units"]:g} sold '
            "share(s) have no purchase on record (e.g. transferred in) — their gain isn't "
            "included.</p>"
        )

    parts.append("<h2>Losses you could harvest</h2>")
    if not harvest:
        parts.append("<p>No taxable position is far enough below its cost to harvest.</p>")
    else:
        parts.append(
            f"<p>Selling these realizes <b>{_money(-s['harvestable_loss'])}</b>: "
            f"{_money(s['offsets_gains'])} offsets this year's gains, "
            f"{_money(s['offsets_ordinary'])} offsets ordinary income (up to $3,000)"
            + (f", and {_money(s['carry_forward'])} carries forward" if s["carry_forward"] else "")
            + f" — roughly {_money(s['est_tax_saving'])} of tax.</p>"
        )
        parts.append(
            _table(
                ["Ticker", "Account", "Loss", "Keep the exposure with", "Watch out"],
                [
                    [
                        esc(c["ticker"]),
                        esc(c["account"]),
                        f"{_money(c['loss_usd'])} ({c['loss_pct']:+.1f}%)",
                        esc(", ".join(c.get("swap_candidates") or []) or "—"),
                        esc(
                            f"bought within 30 days — sell after {c['wash_sale_until']}"
                            if c.get("wash_sale_until")
                            else f"don't rebuy before {c['rebuy_ok_after']}"
                        ),
                    ]
                    for c in harvest
                ],
            )
        )
        parts.append(
            '<p style="font-size:13px;color:#6b7280">Wash-sale rule: buying the same '
            "stock (in any account, IRA included, dividend reinvestment included) within 30 "
            "days before or after the sale disallows the loss. Turn off dividend "
            "reinvestment on a stock you plan to harvest.</p>"
        )

    parts.append("<h2>Gains to leave alone for now</h2>")
    if not soon:
        parts.append("<p>No short-term lot in profit turns long-term in the next 60 days.</p>")
    else:
        parts.append(
            _table(
                ["Ticker", "Account", "Shares", "Gain", "Long-term from", "Tax saved by waiting"],
                [
                    [
                        esc(r["ticker"]),
                        esc(r["account"]),
                        f"{r['units']:g}",
                        _money(r["gain"]),
                        f"{r['long_term_on']:%b %d, %Y}",
                        f"~{_money(r['tax_saved_by_waiting'])}",
                    ]
                    for r in soon
                ],
            )
        )
    return _wrap_html(f"Tax planner — {year}", "".join(parts))
