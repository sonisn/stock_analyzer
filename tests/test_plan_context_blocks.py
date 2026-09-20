"""Facts a sell decision needs, which the rebalancer could not see.

Two were wired into the daily email and the discover run and never
reached the rebalancer: what a holding has under contract, and what a
part-lot would earn if it were completed instead of sold. On 2026-09-20
that produced a plan to trim the 61 AVGO shares sitting 38 short of a
second writable lot, on the name with a book up 552%.
"""

from __future__ import annotations

from dataclasses import dataclass

from stock_analyzer.data.backlog import backlog_block
from stock_analyzer.discover.rebalance_cc import stub_income_block


def test_the_book_that_argues_against_a_sale():
    block = backlog_block(
        {
            "AVGO": {
                "value": 179.2e9,
                "yoy_pct": 552.0,
                "qoq_pct": 8.9,
                "period_end": "2026-08-03",
            },
            "GOOGL": {"value": 519.5e9, "yoy_pct": 380.0},
        }
    )
    assert "AVGO: $179.2B contracted, +552% YoY, +9% QoQ, as of 2026-08-03" in block
    assert "GOOGL: $519.5B contracted, +380% YoY" in block
    assert "signed orders" in block


def test_absence_of_a_book_is_never_evidence_against_a_name():
    """Only ~36% of the index tags the concept — a missing row must not
    read as an empty order book."""
    block = backlog_block({"AVGO": {"value": 179.2e9, "yoy_pct": 552.0}, "BE": {}, "LLY": {}})
    assert "BE" not in block and "LLY" not in block
    assert "must NOT be treated as having no backlog" in block
    assert backlog_block({}) == ""
    assert backlog_block({"BE": {"value": 0}}) == ""


@dataclass
class _Cov:
    stub_shares: float
    stub_dollar_value: float
    to_next_lot_shares: float
    to_next_lot_cost: float


def test_what_completing_a_lot_costs_and_pays():
    """The live case: BE is 16 shares short of a fourth lot, and those
    shares unlock a contract worth more than a quarter of their price."""
    block = stub_income_block(
        {"BE": _Cov(84.36, 22_402.0, 15.64, 4154.0)},
        {"BE": (1325.0, 420.0, "2027-01-15")},
        min_stub_usd=1000.0,
    )
    assert "16 more share(s) (~$4,154) completes a writable lot" in block
    assert "one call at $420 expiring 2027-01-15 pays ~$1,325" in block
    assert "(32% of the cost)" in block
    assert "selling it" in block and "ends the premium permanently" in block


def test_a_stub_with_no_quote_still_shows_its_cost():
    block = stub_income_block(
        {"AVGO": _Cov(61.55, 21_814.0, 38.45, 13_750.0)}, {}, min_stub_usd=1000.0
    )
    assert "38 more share(s) (~$13,750)" in block
    assert "pays" not in block


def test_stubs_too_small_to_bother_with_are_left_out():
    assert (
        stub_income_block({"OKLO": _Cov(3.0, 114.0, 97.0, 3686.0)}, {}, min_stub_usd=1000.0) == ""
    )
    # And a position already on a round lot has no shortfall to report.
    assert stub_income_block({"NVDA": _Cov(0.0, 0.0, 0.0, 0.0)}, {}, min_stub_usd=1000.0) == ""
