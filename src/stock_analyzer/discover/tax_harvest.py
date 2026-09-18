"""Tax-loss harvesting candidates for the rebalance report (no LLM).

For every position slice in a TAXABLE account trading meaningfully below
its cost basis, this computes the loss that selling would realize, how it
splits between short- and long-term lots, a rough tax-saving estimate,
whether a recent purchase makes selling now a wash sale, and peers that
keep similar exposure without being "substantially identical".

The loss itself comes from the brokerage's per-account cost basis, which
already nets out past sells. The purchase lots (from transaction history)
are only used to split it into short/long-term and to name the specific
lots to sell: highest-cost lots first, capped at the units still held.
Tax-advantaged accounts never qualify — a loss inside an IRA/HSA has no
tax value. Rates are the same rough estimates the Reviewer sees
(tax_lot_helper); this is a prompt for a decision, not tax advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .tax_lot_helper import _LONG_TERM_RATE, _SHORT_TERM_RATE

WASH_SALE_DAYS = 30


@dataclass(frozen=True)
class HarvestLot:
    date: str
    units: float
    cost_per_share: float
    loss_usd: float
    long_term: bool


@dataclass
class HarvestCandidate:
    ticker: str
    account: str
    units: float
    avg_cost: float
    price: float
    loss_usd: float  # negative
    short_term_loss_usd: float
    long_term_loss_usd: float
    est_tax_saving_usd: float
    rebuy_ok_after: date  # first day the ticker can be bought back if sold today
    lots: list[HarvestLot] = field(default_factory=list)
    wash_sale_until: date | None = None
    swap_candidates: list[str] = field(default_factory=list)
    plan_conflict: str | None = None

    @property
    def loss_pct(self) -> float:
        return (self.price / self.avg_cost - 1) * 100


def _loss_lots(
    lots: list[dict[str, Any]], account: str, price: float, units_held: float
) -> list[HarvestLot]:
    """Lots in `account` bought above `price`, highest cost first, capped
    at the units still held (the lot history does not net out sells)."""
    candidates = sorted(
        (
            lot
            for lot in lots
            if lot.get("account") == account and float(lot.get("price_per_share") or 0) > price
        ),
        key=lambda lot: float(lot.get("price_per_share") or 0),
        reverse=True,
    )
    out: list[HarvestLot] = []
    remaining = units_held
    for lot in candidates:
        if remaining <= 0:
            break
        units = min(float(lot.get("units") or 0), remaining)
        if units <= 0:
            continue
        cost = float(lot["price_per_share"])
        out.append(
            HarvestLot(
                date=str(lot.get("date") or ""),
                units=units,
                cost_per_share=cost,
                loss_usd=units * (price - cost),
                long_term=lot.get("treatment") == "long_term",
            )
        )
        remaining -= units
    return out


def _wash_sale_until(lots: list[dict[str, Any]], today: date) -> date | None:
    """A purchase in ANY account (IRAs included) within the last 30 days
    makes a loss sale today a wash sale; returns the first clean sell date."""
    latest: date | None = None
    for lot in lots:
        try:
            bought = date.fromisoformat(str(lot.get("date"))[:10])
        except ValueError:
            continue
        if 0 <= (today - bought).days <= WASH_SALE_DAYS and (latest is None or bought > latest):
            latest = bought
    return latest + timedelta(days=WASH_SALE_DAYS + 1) if latest else None


def find_harvest_candidates(
    position_splits: dict[str, dict[str, Any]],
    prices: dict[str, float | None],
    tax_lots: dict[str, dict[str, Any]],
    peers: dict[str, Any] | None = None,
    *,
    min_loss_usd: float = 1000.0,
    min_loss_pct: float = 10.0,
    today: date | None = None,
) -> list[HarvestCandidate]:
    today = today or date.today()
    peers = peers or {}
    out: list[HarvestCandidate] = []
    for ticker, info in position_splits.items():
        price = prices.get(ticker)
        if not price or price <= 0:
            continue
        lots = (tax_lots.get(ticker) or {}).get("lots") or []
        swaps = [
            p for p in ((peers.get(ticker) or {}).get("peers") or {}) if p.upper() != ticker.upper()
        ][:3]
        for split in info.get("splits") or []:
            if split.get("tax_status") != "taxable":
                continue
            units = float(split.get("units") or 0)
            avg = float(split.get("avg_buy_price") or 0)
            if units <= 0 or avg <= 0:
                continue
            loss = units * (price - avg)
            if loss > -min_loss_usd or (price / avg - 1) * 100 > -min_loss_pct:
                continue
            account = str(split.get("account") or "")
            loss_lots = _loss_lots(lots, account, price, units)
            st = sum(lot.loss_usd for lot in loss_lots if not lot.long_term)
            lt = sum(lot.loss_usd for lot in loss_lots if lot.long_term)
            # Blend the rate by the lots' short/long mix; with no lot detail
            # assume long-term, the smaller (conservative) saving.
            if st + lt < 0:
                rate = (st * _SHORT_TERM_RATE + lt * _LONG_TERM_RATE) / (st + lt)
            else:
                rate = _LONG_TERM_RATE
            out.append(
                HarvestCandidate(
                    ticker=ticker,
                    account=account,
                    units=units,
                    avg_cost=avg,
                    price=price,
                    loss_usd=loss,
                    short_term_loss_usd=st,
                    long_term_loss_usd=lt,
                    est_tax_saving_usd=-loss * rate,
                    rebuy_ok_after=today + timedelta(days=WASH_SALE_DAYS + 1),
                    lots=loss_lots,
                    wash_sale_until=_wash_sale_until(lots, today),
                    swap_candidates=swaps,
                )
            )
    return sorted(out, key=lambda c: c.loss_usd)


def flag_plan_conflicts(candidates: list[HarvestCandidate], plan: Any) -> list[HarvestCandidate]:
    """Mark candidates the rebalance plan would buy more of — an ADD/BUY
    within 30 days of a loss sale would disallow the loss."""
    buys = {
        a.ticker.upper(): a.action
        for a in getattr(plan, "actions", None) or []
        if a.action in ("ADD", "BUY")
    }
    for c in candidates:
        action = buys.get(c.ticker.upper())
        if action:
            c.plan_conflict = (
                f"plan says {action} {c.ticker}: buying within 30 days of a loss sale "
                f"is a wash sale, so harvest only if you skip that {action}"
            )
    return candidates


def format_harvest_block(candidates: list[HarvestCandidate]) -> str:
    """Rebalancer prompt input: deterministic candidates, one line each."""
    lines = []
    for c in candidates:
        wash = (
            f"; bought within 30 days, so a loss sale before "
            f"{c.wash_sale_until.isoformat()} may be a wash sale unless those shares "
            f"are sold too (an IRA purchase disallows the loss permanently)"
            if c.wash_sale_until
            else ""
        )
        swaps = (
            f"; similar-exposure swaps: {', '.join(c.swap_candidates)}" if c.swap_candidates else ""
        )
        lines.append(
            f"  {c.ticker} in {c.account}: {c.units:g} sh, basis ${c.avg_cost:,.2f} vs "
            f"${c.price:,.2f} ({c.loss_pct:+.1f}%), loss ${-c.loss_usd:,.0f} "
            f"(short-term ${-c.short_term_loss_usd:,.0f}, long-term ${-c.long_term_loss_usd:,.0f}), "
            f"est. tax saving ~${c.est_tax_saving_usd:,.0f}{wash}{swaps}"
        )
    return "\n".join(lines)


def harvest_report_data(candidates: list[HarvestCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "ticker": c.ticker,
            "account": c.account,
            "units": c.units,
            "avg_cost": c.avg_cost,
            "price": c.price,
            "loss_usd": c.loss_usd,
            "loss_pct": c.loss_pct,
            "short_term_loss_usd": c.short_term_loss_usd,
            "long_term_loss_usd": c.long_term_loss_usd,
            "est_tax_saving_usd": c.est_tax_saving_usd,
            "lots": [
                {
                    "date": lot.date,
                    "units": lot.units,
                    "cost_per_share": lot.cost_per_share,
                    "loss_usd": lot.loss_usd,
                    "long_term": lot.long_term,
                }
                for lot in c.lots
            ],
            "wash_sale_until": c.wash_sale_until.isoformat() if c.wash_sale_until else None,
            "rebuy_ok_after": c.rebuy_ok_after.isoformat(),
            "swap_candidates": c.swap_candidates,
            "plan_conflict": c.plan_conflict,
        }
        for c in candidates
    ]
