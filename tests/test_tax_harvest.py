"""Tax-loss harvesting candidates for the rebalance report (no LLM)."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from stock_analyzer.discover.rebalance_sections import append_harvest_section
from stock_analyzer.discover.report_pdf import render_pdf
from stock_analyzer.discover.tax_harvest import (
    find_harvest_candidates,
    flag_plan_conflicts,
    format_harvest_block,
    harvest_report_data,
)

TODAY = date(2026, 9, 18)


def _split(account: str, units: float, avg: float, status: str = "taxable") -> dict:
    return {"account": account, "tax_status": status, "units": units, "avg_buy_price": avg}


def _lot(d: str, units: float, px: float, account: str = "Brokerage", lt: bool = False) -> dict:
    return {
        "date": d,
        "units": units,
        "price_per_share": px,
        "account": account,
        "treatment": "long_term" if lt else "short_term",
    }


def _find(splits, price, lots=(), peers=None, **kw):
    return find_harvest_candidates(
        {"XYZ": {"splits": splits}},
        {"XYZ": price},
        {"XYZ": {"lots": list(lots)}},
        {"XYZ": {"peers": peers or {}}},
        today=TODAY,
        **kw,
    )


def test_taxable_loss_is_a_candidate_with_lot_split_and_blended_saving():
    lots = [
        _lot("2025-01-10", 60, 100.0, lt=True),  # LT, loss 60 x -40
        _lot("2026-03-02", 40, 90.0),  # ST, loss 40 x -30
        _lot("2026-05-01", 50, 55.0),  # bought below price: not a loss lot
    ]
    [c] = _find(
        [_split("Brokerage", 100, 96.0)], 60.0, lots, peers={"ABC": {}, "XYZ": {}, "DEF": {}}
    )
    assert c.loss_usd == pytest.approx(-3600)
    assert (c.long_term_loss_usd, c.short_term_loss_usd) == (pytest.approx(-2400), -1200)
    rate = (1200 * 0.32 + 2400 * 0.18) / 3600
    assert c.est_tax_saving_usd == pytest.approx(3600 * rate)
    assert [lot.date for lot in c.lots] == ["2025-01-10", "2026-03-02"]
    assert c.swap_candidates == ["ABC", "DEF"]
    assert c.wash_sale_until is None
    assert c.rebuy_ok_after == date(2026, 10, 19)


def test_loss_lots_are_capped_at_units_still_held():
    # History shows 200 shares bought above the price, but only 50 remain.
    lots = [_lot("2026-01-05", 100, 120.0), _lot("2026-02-05", 100, 110.0)]
    [c] = _find([_split("Brokerage", 50, 115.0)], 80.0, lots)
    assert [(lot.date, lot.units) for lot in c.lots] == [("2026-01-05", 50)]


def test_tax_advantaged_and_small_losses_are_skipped():
    assert _find([_split("IRA", 100, 100.0, "tax_advantaged")], 50.0) == []
    assert _find([_split("Brokerage", 10, 100.0)], 50.0) == []  # $500 < $1,000
    assert _find([_split("Brokerage", 1000, 100.0)], 95.0) == []  # -5% < 10%


def test_recent_purchase_in_any_account_sets_wash_sale_date():
    lots = [_lot("2024-01-01", 100, 100.0, lt=True), _lot("2026-09-01", 5, 70.0, "IRA")]
    [c] = _find([_split("Brokerage", 100, 100.0)], 60.0, lots)
    assert c.wash_sale_until == date(2026, 10, 2)
    assert "2026-10-02 may be a wash sale" in format_harvest_block([c])


def test_no_lot_detail_assumes_long_term_rate():
    [c] = _find([_split("Brokerage", 100, 100.0)], 50.0)
    assert c.est_tax_saving_usd == pytest.approx(5000 * 0.18)


def test_plan_add_on_a_harvest_ticker_is_flagged():
    [c] = _find([_split("Brokerage", 100, 100.0)], 50.0)
    plan = SimpleNamespace(actions=[SimpleNamespace(action="ADD", ticker="XYZ")])
    [c] = flag_plan_conflicts([c], plan)
    assert c.plan_conflict.startswith("plan says ADD XYZ")


def test_report_section_renders():
    lots = [_lot("2025-01-10", 100, 100.0, lt=True)]
    cands = _find([_split("Brokerage", 100, 100.0)], 60.0, lots, peers={"ABC": {}})
    sections = []
    append_harvest_section(sections, harvest_report_data(cands))
    assert sections[0].text == "Tax-loss harvesting candidates"
    assert "$4,000 of losses" in sections[1].text
    row = sections[2].table_rows[0]
    assert row[:6] == ["XYZ", "Brokerage", "-$4,000 (-40.0%)", "$0 / $4,000", "~$720", "ABC"]
    assert "sell lots 2025-01-10 (100 sh @ $100.00, LT)" in row[6]
    assert render_pdf(sections, {}).startswith(b"%PDF")
