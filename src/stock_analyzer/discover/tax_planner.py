"""Year-end tax planner — December email. No LLM.

For TAXABLE accounts only (a gain or loss inside an IRA/HSA has no tax
effect):

  1. Realized so far this year, estimated: each SELL is matched first-in
     first-out against that account's own purchase lots (BUY + dividend
     reinvestments) and split short-/long-term by `long_term_on`. SnapTrade
     sells carry no cost basis, so shares without a known purchase (e.g.
     transferred in) are counted as "basis unknown", never guessed.
  2. Losses available to harvest now (tax_harvest candidates) and how much
     of this year's gains they would offset (+ up to $3,000 of ordinary
     income; the rest carries forward).
  3. Gains to leave alone for now: short-term lots in profit that turn
     long-term within `SOON_DAYS` — selling after that date is taxed at the
     lower long-term rate.

Rates are the same rough estimates the rest of the app uses; this is a
prompt for a conversation with a tax professional, not tax advice.
"""

from __future__ import annotations

from collections import deque
from datetime import date, timedelta
from typing import Any

from ..data.transactions import _coerce_date, _extract_ticker, is_option_activity
from ..models.portfolio import long_term_on
from .tax_lot_helper import long_term_rate, short_term_rate

SOON_DAYS = 60
ORDINARY_OFFSET_LIMIT = 3000.0


def _num(v: Any) -> float:
    try:
        return float(v or 0)
    except TypeError, ValueError:
        return 0.0


def realized_this_year(activities: list[dict[str, Any]], *, year: int) -> dict[str, Any]:
    """FIFO realized gains for one account's activity list."""
    lots: dict[str, deque[list[Any]]] = {}
    rows: list[dict[str, Any]] = []
    unknown_units = 0.0
    events = []
    for a in activities:
        if is_option_activity(a):
            continue
        kind = (a.get("type") or "").upper()
        day = _coerce_date(a.get("trade_date") or a.get("settlement_date"))
        ticker = _extract_ticker(a)
        if day is None or not ticker or kind not in ("BUY", "REI", "SELL"):
            continue
        events.append((day, 0 if kind != "SELL" else 1, kind, ticker, a))
    for day, _, kind, ticker, a in sorted(events, key=lambda e: (e[0], e[1])):
        units, price = abs(_num(a.get("units"))), _num(a.get("price"))
        if units <= 0 or price <= 0:
            continue
        queue = lots.setdefault(ticker, deque())
        if kind in ("BUY", "REI"):
            queue.append([day, units, price])
            continue
        remaining, st, lt = units, 0.0, 0.0
        while remaining > 1e-9 and queue:
            lot = queue[0]
            take = min(lot[1], remaining)
            gain = take * (price - lot[2])
            if day >= long_term_on(lot[0]):
                lt += gain
            else:
                st += gain
            lot[1] -= take
            remaining -= take
            if lot[1] <= 1e-9:
                queue.popleft()
        if day.year != year:
            continue
        fee = _num(a.get("fee"))
        rows.append(
            {
                "date": day,
                "ticker": ticker,
                "units": units,
                "proceeds": units * price,
                "short_term": st - (fee if st else 0.0),
                "long_term": lt - (fee if lt and not st else 0.0),
                "basis_unknown_units": remaining if remaining > 1e-9 else 0.0,
            }
        )
        unknown_units += remaining if remaining > 1e-9 else 0.0
    options = option_realized(activities, year=year)
    return {
        "short_term": sum(r["short_term"] for r in rows) + options,
        "long_term": sum(r["long_term"] for r in rows),
        "basis_unknown_units": unknown_units,
        "options": options,
        "rows": rows,
    }


def option_realized(activities: list[dict[str, Any]], *, year: int) -> float:
    """Net cash of option contracts CLOSED this year (bought and sold, or
    expired/assigned), counted as short-term. A contract is identified by
    its option_symbol; ones still open are left out."""
    by_contract: dict[str, dict[str, Any]] = {}
    for a in activities:
        opt = a.get("option_symbol")
        if not opt:
            continue
        key = str(opt.get("id") or opt.get("ticker")) if isinstance(opt, dict) else str(opt)
        rec = by_contract.setdefault(key, {"units": 0.0, "cash": 0.0, "last": None})
        kind = (a.get("type") or "").upper()
        if kind in ("BUY", "SELL"):
            rec["units"] += _num(a.get("units"))
        rec["cash"] += _num(a.get("amount"))
        day = _coerce_date(a.get("trade_date") or a.get("settlement_date"))
        closes = kind in ("OPTIONEXPIRATION", "OPTIONASSIGNMENT", "OPTIONEXERCISE")
        if day and (rec["last"] is None or day > rec["last"]):
            rec["last"] = day
        if closes:
            rec["units"] = 0.0
    return sum(
        r["cash"]
        for r in by_contract.values()
        if abs(r["units"]) < 1e-9 and r["last"] is not None and r["last"].year == year
    )


def gains_turning_long_term(
    tax_lots: dict[str, dict[str, Any]],
    prices: dict[str, float | None],
    taxable_accounts: set[str],
    *,
    today: date,
    days: int = SOON_DAYS,
) -> list[dict[str, Any]]:
    """Short-term lots in profit that become long-term within `days`."""
    out = []
    for ticker, payload in tax_lots.items():
        price = prices.get(ticker)
        if not price:
            continue
        for lot in (payload or {}).get("lots") or []:
            if lot.get("account") not in taxable_accounts or lot.get("treatment") == "long_term":
                continue
            lt_on = lot.get("long_term_on")
            if not lt_on:
                continue
            when = date.fromisoformat(lt_on)
            gain = _num(lot.get("units")) * (price - _num(lot.get("price_per_share")))
            if gain > 0 and today < when <= today + timedelta(days=days):
                out.append(
                    {
                        "ticker": ticker,
                        "account": lot["account"],
                        "units": _num(lot.get("units")),
                        "gain": gain,
                        "long_term_on": when,
                        "tax_saved_by_waiting": gain * (short_term_rate() - long_term_rate()),
                    }
                )
    return sorted(out, key=lambda r: r["long_term_on"])


def plan_summary(
    realized: dict[str, dict[str, Any]], harvest: list[dict[str, Any]]
) -> dict[str, Any]:
    """Totals across taxable accounts and how far the harvestable losses
    go against this year's gains."""
    st = sum(r["short_term"] for r in realized.values())
    lt = sum(r["long_term"] for r in realized.values())
    net_gain = st + lt
    losses = -sum(c["loss_usd"] for c in harvest)  # positive
    offset_gains = min(losses, max(net_gain, 0.0))
    ordinary = min(max(losses - offset_gains, 0.0), ORDINARY_OFFSET_LIMIT)
    return {
        "short_term": st,
        "long_term": lt,
        "net_gain": net_gain,
        "harvestable_loss": losses,
        "offsets_gains": offset_gains,
        "offsets_ordinary": ordinary,
        "carry_forward": max(losses - offset_gains - ordinary, 0.0),
        "est_tax_saving": sum(c.get("est_tax_saving_usd") or 0.0 for c in harvest),
        "basis_unknown_units": sum(r["basis_unknown_units"] for r in realized.values()),
    }


def last_trading_day_of_year(year: int) -> date:
    """Dec 31 unless it falls on a weekend (the NYSE is open on Dec 31)."""
    d = date(year, 12, 31)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d
