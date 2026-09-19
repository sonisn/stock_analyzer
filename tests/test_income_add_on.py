"""Dividend income and add-on-weakness (daily health + rebalance block)."""

from __future__ import annotations

from datetime import date

import pytest

from stock_analyzer.discover.add_on import add_on_candidates, format_add_on_block
from stock_analyzer.discover.income import dividend_income
from stock_analyzer.models.llm import HoldingReview
from stock_analyzer.reporting.health import (
    build_portfolio_health,
    decision_items,
    render_health_html,
)

TODAY = date(2026, 9, 18)


def test_dividend_income_forward_and_received():
    inc = dividend_income(
        units={"KO": 100.0, "NVDA": 10.0},
        values={"KO": 7000.0, "NVDA": 1800.0, "ARM": 1200.0},
        rates={"KO": 2.04, "NVDA": 0.04},
        received=[
            {
                "date": date(2026, 9, 1),
                "ticker": "KO",
                "amount": 51.0,
                "account": "IRA",
                "reinvested": True,
            },
            {
                "date": date(2026, 6, 1),
                "ticker": "POWL",
                "amount": 3.69,
                "account": "Taxable",
                "reinvested": False,
            },
            {
                "date": date(2025, 6, 1),
                "ticker": "KO",
                "amount": 48.0,
                "account": "IRA",
                "reinvested": True,
            },  # older than 12 months
        ],
        today=TODAY,
    )
    assert inc["forward_annual"] == pytest.approx(204.4)
    assert inc["yield_pct"] == pytest.approx(204.4 / 10000 * 100)
    assert inc["received_12m"] == pytest.approx(54.69)
    assert inc["reinvested_12m"] == 51.0 and inc["cash_12m"] == pytest.approx(3.69)
    assert inc["cash_accounts"] == ["Taxable"]
    assert [r["ticker"] for r in inc["rows"]] == ["KO", "POWL", "NVDA"]


def test_add_on_candidates_filters():
    got = add_on_candidates(
        values={
            "DIP": 1000.0,
            "BIG": 5000.0,
            "CUT": 1000.0,
            "FLAG": 1000.0,
            "HOT": 1000.0,
            "CAP": 1000.0,
        },
        highs={
            "DIP": (80.0, 100.0),  # -20%: qualifies
            "BIG": (70.0, 100.0),  # 50% of the portfolio: no room
            "CUT": (70.0, 100.0),  # estimates being cut
            "FLAG": (70.0, 100.0),  # thesis on watch
            "HOT": (95.0, 100.0),  # only -5%
            "CAP": (70.0, 100.0),  # sector over the cap
        },
        sector_of={"CAP": "Technology"},
        over_cap_sectors={"Technology"},
        thesis_flagged={"FLAG"},
        estimates_cut=lambda ts: {"CUT"} & set(ts),
    )
    assert [(c["ticker"], round(c["off_high_pct"])) for c in got] == [("DIP", -20)]


def _review(verdict: str, conf: int) -> HoldingReview:
    return HoldingReview.model_construct(ticker="X", verdict=verdict, confidence=conf)


def test_rebalance_add_on_block():
    block = format_add_on_block(
        {
            "DIP": _review("HOLD", 8),
            "LOW": _review("HOLD", 5),
            "TRIM": _review("TRIM", 8),
            "FLAT": _review("HOLD", 9),
        },
        {t: {"dist_from_52w_high": -0.22} for t in ("DIP", "LOW", "TRIM")}
        | {"FLAT": {"dist_from_52w_high": -0.05}},
        {"DIP": 1000.0, "LOW": 1000.0, "TRIM": 1000.0, "FLAT": 7000.0},
    )
    assert block == "  DIP: HOLD-8, -22% vs 52-week high, 10.0% of holdings"


def test_health_shows_income_and_add_ons():
    holdings = {
        "IRA": [
            {"ticker": "KO", "units": 100, "average_purchase_price": 60.0, "price": 70.0},
            {"ticker": "DIP", "units": 10, "average_purchase_price": 90.0, "price": 80.0},
        ]
    }
    h = build_portfolio_health(
        holdings,
        income=lambda units, values: dividend_income(
            units=units,
            values=values,
            rates={"KO": 2.04},
            received=[
                {
                    "date": TODAY,
                    "ticker": "KO",
                    "amount": 3.0,
                    "account": "Taxable",
                    "reinvested": False,
                }
            ],
            today=TODAY,
        ),
        add_on=lambda **kw: [
            {"ticker": "DIP", "off_high_pct": -20.0, "weight_pct": 10.3, "sector": "Tech"}
        ],
    )
    body = render_health_html(h)
    assert "Dividend income" in body and "$204/yr" in body
    assert "arrived as cash in Taxable — consider putting it to work in DIP" in body
    assert "Add on weakness" in body
    item = next(i for i in decision_items(h) if i["label"] == "ADD ON DIP")
    assert item["priority"] == 4 and "20% below its 52-week high" in item["text"]


def test_flagged_holdings_are_not_add_on_candidates():
    seen = {}

    def add_on(**kw):
        seen.update(kw)
        return []

    build_portfolio_health(
        {"IRA": [{"ticker": "POWL", "units": 10, "average_purchase_price": 100.0, "price": 70.0}]},
        harvest=lambda: [{"ticker": "LOSS"}],
        add_on=add_on,
    )
    assert {"POWL", "LOSS"} <= seen["thesis_flagged"]  # drawdown + tax-loss names
