"""Tax-impact pre-computation — the highest-stakes math in the system.

Wrong numbers here mean wrong sell-ordering advice and real-dollar tax
mistakes. These tests pin down the gain / loss / tax-advantaged paths
plus the aggregate `if_all_sold_today` rollup.
"""
from __future__ import annotations

from stock_analyzer.discover.tax_lot_helper import (
    _compute_lot_impact,
    enrich_tax_lots_with_impact,
)

# A 100-share lot bought at $50 ($5000 cost basis).
_LOT_100SH_50 = {"units": 100, "price_per_share": 50.0, "total_cost": 5000.0}


# --- gains / losses, taxable accounts -------------------------------------


def test_long_term_gain_estimates_18pct_tax():
    """100 sh × $50 cost, sold at $80 = $3000 LT gain × 18% = $540 tax."""
    impact = _compute_lot_impact(
        {**_LOT_100SH_50, "treatment": "long_term"},
        current_price=80.0,
        account_tax_status="taxable",
    )
    assert impact["realized_gain_loss"] == 3000.0
    assert impact["estimated_tax_dollars"] == 540.0
    assert impact["harvest_benefit_or_cost"] == "cost"
    assert impact["free_to_trim"] is False


def test_short_term_loss_is_harvest_benefit_at_32pct():
    """100 sh × $50 cost, sold at $40 = $1000 ST loss → harvest benefit
    $1000 × 32% = $320 (negative tax = benefit)."""
    impact = _compute_lot_impact(
        {**_LOT_100SH_50, "treatment": "short_term"},
        current_price=40.0,
        account_tax_status="taxable",
    )
    assert impact["realized_gain_loss"] == -1000.0
    assert impact["estimated_tax_dollars"] == -320.0  # negative = benefit
    assert impact["harvest_benefit_or_cost"] == "benefit"
    # Wash-sale guidance must be present on loss lots so the reviewer
    # surfaces it to the user.
    assert "wash-sale" in impact["note"].lower()


def test_tax_advantaged_zeroes_tax_regardless_of_gain():
    """IRA / 401(k) / HSA: gain or loss inside the wrapper is shielded.
    Must report $0 tax, free_to_trim=True."""
    impact = _compute_lot_impact(
        {**_LOT_100SH_50, "treatment": "long_term"},
        current_price=200.0,  # huge gain — irrelevant inside an IRA
        account_tax_status="tax_advantaged",
    )
    assert impact["realized_gain_loss"] == 15000.0  # math still computed
    assert impact["estimated_tax_dollars"] == 0.0
    assert impact["free_to_trim"] is True
    assert impact["harvest_benefit_or_cost"] == "none_tax_advantaged"


def test_long_term_gain_lower_rate_than_short_term_gain():
    """Same $ gain — LT (18%) must produce less estimated tax than ST (32%).
    Direct calibration of the rate constants — a typo flipping the rates
    would advise the user to sell the wrong lot first."""
    lt = _compute_lot_impact(
        {**_LOT_100SH_50, "treatment": "long_term"},
        current_price=80.0, account_tax_status="taxable",
    )
    st = _compute_lot_impact(
        {**_LOT_100SH_50, "treatment": "short_term"},
        current_price=80.0, account_tax_status="taxable",
    )
    assert lt["estimated_tax_dollars"] < st["estimated_tax_dollars"]


# --- aggregate rollup ------------------------------------------------------


def test_aggregate_if_all_sold_today_sums_per_status():
    """Mix of taxable gain + IRA gain + taxable loss should report:
      - n_lots_taxable_gain = 1
      - n_lots_taxable_loss_harvest = 1
      - n_lots_free_to_trim = 1
      - total_estimated_tax = gain_tax + 0 + (-benefit)
    """
    tax_payload = {
        "lots": [
            {"account": "Taxable", "units": 100, "price_per_share": 50.0,
             "total_cost": 5000.0, "treatment": "long_term"},
            {"account": "IRA", "units": 100, "price_per_share": 50.0,
             "total_cost": 5000.0, "treatment": "long_term"},
            {"account": "Taxable", "units": 100, "price_per_share": 50.0,
             "total_cost": 5000.0, "treatment": "short_term"},
        ],
    }
    accounts = {
        "Taxable": {"tax_status": "taxable"},
        "IRA":     {"tax_status": "tax_advantaged"},
    }
    # Current price $80: taxable LT gain = +$3000, IRA gain irrelevant,
    # taxable ST gain = +$3000 (same $80 vs $50). Make the third a LOSS
    # by changing its cost basis higher than current price.
    tax_payload["lots"][2] = {
        "account": "Taxable", "units": 100, "price_per_share": 100.0,
        "total_cost": 10000.0, "treatment": "short_term",
    }
    out = enrich_tax_lots_with_impact(tax_payload, current_price=80.0,
                                      account_meta_by_name=accounts)
    summary = out["if_all_sold_today"]
    assert summary["n_lots_taxable_gain"] == 1     # taxable LT gain lot
    assert summary["n_lots_free_to_trim"] == 1     # IRA lot
    assert summary["n_lots_taxable_loss_harvest"] == 1  # taxable ST loss lot

    # total_estimated_tax = $540 (LT gain) + $0 (IRA) + -$640 (ST loss × 32%)
    assert summary["total_estimated_tax"] == 540.0 + 0.0 + -640.0


def test_missing_or_empty_lots_is_a_no_op():
    """No lots, no current price, or empty payload must pass through
    unchanged — never crash a rebalance run."""
    assert enrich_tax_lots_with_impact({}, 100.0, {}) == {}
    assert enrich_tax_lots_with_impact({"lots": []}, 100.0, {}) == {"lots": []}
    payload = {"lots": [_LOT_100SH_50]}
    # current_price <= 0 → no enrichment
    assert enrich_tax_lots_with_impact(payload, 0.0, {}) == payload
