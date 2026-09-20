"""A dead listing is the one loss the harvester can never find."""

from __future__ import annotations

from datetime import date

from stock_analyzer.discover.tax_harvest import find_harvest_candidates
from stock_analyzer.discover.worthless import find_worthless_positions, worthless_report_data
from stock_analyzer.reporting.tax_plan import render_tax_plan_html

# The two positions the Schwab reconnect surfaced, as the broker reports
# them: by CUSIP, at $0.00, with the basis intact.
TARONIS = {
    "Individual ...004": [
        {
            "ticker": "876214206",
            "kind": "equity",
            "units": 144,
            "price": 0.0,
            "average_purchase_price": 12.29,
        },
        {
            "ticker": "87621P209",
            "kind": "equity",
            "units": 9,
            "price": 0.0,
            "average_purchase_price": 207.22,
        },
    ]
}
META = {"Individual ...004": {"tax_status": "taxable"}}


def test_the_harvester_drops_exactly_these():
    """Not a regression to fix in the harvester — it needs a price to sell
    at. This pins WHY the worthless module has to exist."""
    splits = {
        "876214206": {
            "splits": [
                {
                    "account": "Individual ...004",
                    "tax_status": "taxable",
                    "units": 144,
                    "avg_buy_price": 12.29,
                }
            ]
        }
    }
    assert find_harvest_candidates(splits, {"876214206": 0.0}, {}) == []


def test_worthless_positions_are_found_biggest_basis_first():
    found = find_worthless_positions(TARONIS, META)
    assert [(p.symbol, p.cost_basis_usd) for p in found] == [
        ("87621P209", 1864.98),
        ("876214206", 1769.76),
    ]
    assert [p.loss_usd for p in found] == [-1864.98, -1769.76]
    assert all("CUSIP" in p.reason for p in found)


def test_a_listed_ticker_at_zero_is_a_broken_feed():
    """The whole guard: never call a real company worthless because a
    quote failed."""
    holdings = {
        "Individual ...004": [
            {
                "ticker": "NVDA",
                "kind": "equity",
                "units": 100,
                "price": 0.0,
                "average_purchase_price": 150.0,
            }
        ]
    }
    assert find_worthless_positions(holdings, META) == []


def test_tax_advantaged_accounts_and_small_stubs_are_skipped():
    # A 401(k) commingled pool is unlisted and unquoted, but a loss there
    # has no tax value.
    plan = {
        "Broadcom U.S. 401(k) Plan": [
            {
                "ticker": "FGCCPS",
                "kind": "other",
                "units": 40,
                "price": 0.0,
                "average_purchase_price": 110.0,
            }
        ]
    }
    assert (
        find_worthless_positions(
            plan, {"Broadcom U.S. 401(k) Plan": {"tax_status": "tax_advantaged"}}
        )
        == []
    )
    # A $12 stub is not worth an amended return.
    stub = {
        "Individual ...004": [
            {
                "ticker": "123456789",
                "kind": "equity",
                "units": 1,
                "price": 0.0,
                "average_purchase_price": 12.0,
            }
        ]
    }
    assert find_worthless_positions(stub, META) == []
    # A position still worth something is not worthless.
    alive = {
        "Individual ...004": [
            {
                "ticker": "123456789",
                "kind": "equity",
                "units": 100,
                "price": 4.0,
                "average_purchase_price": 12.0,
            }
        ]
    }
    assert find_worthless_positions(alive, META) == []


def test_the_report_says_whose_call_the_year_is():
    body = render_tax_plan_html(
        year=2026,
        taxable_accounts=["Individual ...004"],
        realized={},
        summary={
            "short_term": 0.0,
            "long_term": 0.0,
            "net_gain": 0.0,
            "harvestable_loss": 0.0,
            "offsets_gains": 0.0,
            "offsets_ordinary": 0.0,
            "carry_forward": 0.0,
            "est_tax_saving": 0.0,
            "basis_unknown_units": 0,
        },
        harvest=[],
        soon=[],
        last_day=date(2026, 12, 31),
        worthless=worthless_report_data(find_worthless_positions(TARONIS, META)),
    )
    assert "Possibly worthless" in body
    assert "$3,635" in body  # the whole basis, not a per-line figure
    assert "165(g)" in body and "amended return" in body
    assert "87621P209" in body and "876214206" in body
    # Absent the section entirely when there is nothing to say.
    assert "Possibly worthless" not in render_tax_plan_html(
        year=2026,
        taxable_accounts=[],
        realized={},
        summary={
            "short_term": 0.0,
            "long_term": 0.0,
            "net_gain": 0.0,
            "harvestable_loss": 0.0,
            "offsets_gains": 0.0,
            "offsets_ordinary": 0.0,
            "carry_forward": 0.0,
            "est_tax_saving": 0.0,
            "basis_unknown_units": 0,
        },
        harvest=[],
        soon=[],
        last_day=date(2026, 12, 31),
        worthless=[],
    )
