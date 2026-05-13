"""Deterministic per-lot tax-impact pre-computation.

Hybrid pattern: the LLM (reviewer / rebalancer) decides WHICH lots to
sell with reasoning; this helper does the arithmetic — realized P&L
per lot, short-term vs long-term treatment, tax-advantaged free pass,
rough tax-dollar estimate. The LLM reads the pre-computed
`if_sold_today` field instead of fabricating numbers.

Used by `cli/rebalance.py` to enrich each holding's `tax_lots` payload
before passing it to the reviewer agent. The reviewer's prompt
explicitly references the pre-computed numbers — eliminates the
hallucination class where Sonnet invented realized gains that don't
match the actual lot data.
"""
from __future__ import annotations

from typing import Any

# Federal tax rate assumptions — rough enough that the LLM can use them
# as ranking signal but explicit enough to flag in prose. The user's
# actual bracket may differ, so the LLM cites these as estimates.
_LONG_TERM_RATE = 0.18    # mid-bracket long-term cap gains + state
_SHORT_TERM_RATE = 0.32   # high-bracket ordinary income


def _compute_lot_impact(
    lot: dict[str, Any],
    current_price: float,
    account_tax_status: str,
) -> dict[str, Any]:
    """Pre-compute what selling THIS lot today would yield.

    Returns a dict the LLM can read directly — no math required of it.
    `account_tax_status` should be 'taxable' or 'tax_advantaged' per
    the brokerage.classify_tax_status helper.
    """
    units = float(lot.get("units") or 0)
    cost_per_share = float(lot.get("price_per_share") or 0)
    total_cost = float(lot.get("total_cost") or 0) or units * cost_per_share
    treatment = lot.get("treatment", "short_term")  # 'short_term' | 'long_term'

    proceeds = units * current_price
    realized = proceeds - total_cost

    # In a tax-advantaged account, gains and losses inside the account
    # are tax-shielded. Treat the realized amount as "moves cash within
    # the wrapper" — no tax bill, no harvest benefit (no taxable gain
    # elsewhere to offset).
    if account_tax_status == "tax_advantaged":
        return {
            "proceeds": round(proceeds, 2),
            "realized_gain_loss": round(realized, 2),
            "treatment": treatment,
            "account_tax_status": "tax_advantaged",
            "estimated_tax_dollars": 0.0,
            "harvest_benefit_or_cost": "none_tax_advantaged",
            "free_to_trim": True,
            "note": (
                "Tax-advantaged account — no tax consequence on sale; "
                "no harvest benefit (account is already tax-shielded)."
            ),
        }

    # Taxable account.
    if realized >= 0:
        # Gain: tax owed at the appropriate rate.
        rate = _LONG_TERM_RATE if treatment == "long_term" else _SHORT_TERM_RATE
        estimated_tax = realized * rate
        return {
            "proceeds": round(proceeds, 2),
            "realized_gain_loss": round(realized, 2),
            "treatment": treatment,
            "account_tax_status": "taxable",
            "estimated_tax_dollars": round(estimated_tax, 2),
            "harvest_benefit_or_cost": "cost",
            "free_to_trim": False,
            "note": (
                f"Taxable gain — realizing ${realized:.0f} at "
                f"~{int(rate*100)}% rate = ~${estimated_tax:.0f} tax owed."
            ),
        }

    # Taxable LOSS: harvest benefit.
    # Loss offsets gains elsewhere (long-term offsets long-term first,
    # then crosses to short-term; up to $3,000 against ordinary income
    # per year with unlimited carryforward). For ranking purposes we
    # estimate the marginal tax saving at the matching rate.
    rate = _LONG_TERM_RATE if treatment == "long_term" else _SHORT_TERM_RATE
    harvest_savings = abs(realized) * rate
    return {
        "proceeds": round(proceeds, 2),
        "realized_gain_loss": round(realized, 2),
        "treatment": treatment,
        "account_tax_status": "taxable",
        "estimated_tax_dollars": round(-harvest_savings, 2),  # negative = benefit
        "harvest_benefit_or_cost": "benefit",
        "free_to_trim": False,
        "note": (
            f"Taxable LOSS — selling crystallizes ~${abs(realized):.0f} "
            f"loss usable against gains elsewhere; estimated tax "
            f"saving ~${harvest_savings:.0f}. Wash-sale window: do not "
            f"re-buy this security or substantially identical for 30 "
            f"days after sale."
        ),
    }


def enrich_tax_lots_with_impact(
    tax_payload: dict[str, Any],
    current_price: float,
    account_meta_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Annotate each lot in the tax payload with `if_sold_today` field.

    The reviewer's prompt directs Sonnet to read this field instead of
    fabricating realized-gain numbers. Falls through cleanly if the
    payload is missing or empty.
    """
    if not tax_payload or not isinstance(tax_payload, dict):
        return tax_payload
    lots = tax_payload.get("lots") or []
    if not lots or current_price is None or current_price <= 0:
        return tax_payload

    enriched: list[dict[str, Any]] = []
    aggregate_proceeds = 0.0
    aggregate_realized = 0.0
    aggregate_tax = 0.0
    n_free = 0
    n_harvest = 0
    n_taxable_gain = 0

    for lot in lots:
        account_name = lot.get("account") or ""
        meta = account_meta_by_name.get(account_name) or {}
        account_tax_status = meta.get("tax_status") or "taxable"
        impact = _compute_lot_impact(lot, current_price, account_tax_status)

        enriched.append({**lot, "if_sold_today": impact})
        aggregate_proceeds += impact["proceeds"]
        aggregate_realized += impact["realized_gain_loss"]
        aggregate_tax += impact["estimated_tax_dollars"]
        if impact["free_to_trim"]:
            n_free += 1
        elif impact["harvest_benefit_or_cost"] == "benefit":
            n_harvest += 1
        elif impact["harvest_benefit_or_cost"] == "cost":
            n_taxable_gain += 1

    new_payload = {**tax_payload, "lots": enriched}
    new_payload["if_all_sold_today"] = {
        "total_proceeds": round(aggregate_proceeds, 2),
        "total_realized_gain_loss": round(aggregate_realized, 2),
        "total_estimated_tax": round(aggregate_tax, 2),
        "n_lots_free_to_trim": n_free,
        "n_lots_taxable_loss_harvest": n_harvest,
        "n_lots_taxable_gain": n_taxable_gain,
    }
    return new_payload


__all__ = ["enrich_tax_lots_with_impact"]
