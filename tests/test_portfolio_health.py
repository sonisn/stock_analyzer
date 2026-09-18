"""The daily email's "Decide today" list and Portfolio health block."""

from __future__ import annotations

from stock_analyzer.cli.portfolio import build_email
from stock_analyzer.reporting.health import (
    build_portfolio_health,
    decision_items,
    flagged_tickers,
    render_decisions_html,
    render_health_html,
)

HOLDINGS = {
    "Brokerage": [
        {"ticker": "DOWN", "units": 10, "average_purchase_price": 100.0, "price": 75.0},
        {"ticker": "NEAR", "units": 10, "average_purchase_price": 100.0, "price": 82.0},
        {"ticker": "OK", "units": 10, "average_purchase_price": 100.0, "price": 130.0},
    ],
    "IRA": [{"ticker": "OK", "units": 10, "average_purchase_price": 90.0, "price": 130.0}],
}

REPORT = """\
Social/Economic Sentiment:
Markets were calm.
----------------------------------------
OK - Okay Corp
Price: 130
----------------------------------------
NEAR - Near Inc
Price: 82
----------------------------------------
DOWN - Down Co
Price: 75
"""


def _health(**overrides):
    kwargs = dict(
        sector_of=lambda t: {"DOWN": "Technology", "NEAR": "Technology", "OK": "Energy"},
        held_thesis_checks=lambda held: [
            {
                "ticker": "OK",
                "status": "TARGET HIT",
                "return_pct": 44.0,
                "excess_pct": 30.0,
                "signals": [{"severity": "target", "text": "past its bull-case target"}],
            }
        ],
        harvest=lambda: [
            {
                "ticker": "DOWN",
                "account": "Brokerage",
                "loss_usd": -250.0,
                "loss_pct": -25.0,
                "est_tax_saving_usd": 80.0,
                "wash_sale_until": None,
                "rebuy_ok_after": "2026-10-19",
            }
        ],
        earnings=lambda t: {
            "NEAR": {"ticker": "NEAR", "earnings_date": "2026-09-19", "days_until": 1}
        },
    )
    kwargs.update(overrides)
    return build_portfolio_health(HOLDINGS, max_sector_pct=30.0, **kwargs)


def test_snapshot_and_alerts():
    h = _health()
    assert h.snapshot["positions"] == 3 and h.snapshot["value"] == 750 + 820 + 2600
    assert [(r["ticker"], r["past_stop"]) for r in h.stop_loss] == [("DOWN", True), ("NEAR", False)]
    tech = next(r for r in h.sectors if r["sector"] == "Technology")
    assert round(tech["pct"], 1) == 37.6 and tech["over"]
    assert h.unavailable == []


def test_decisions_are_ranked_by_urgency():
    items = decision_items(_health())
    assert [i["label"] for i in items] == [
        "PAST STOP",
        "TARGET HIT",
        "EARNINGS",
        "NEAR STOP",
        "TAX LOSS",
        "OVER CAP",  # standing guidance ranks last
        "OVER CAP",
    ]
    assert items[0]["text"].startswith("Review DOWN: -25.0% from cost")
    assert flagged_tickers(_health()) == ["DOWN", "OK", "NEAR"]


def test_a_failing_check_is_reported_not_raised():
    def boom(*_):
        raise RuntimeError("SnapTrade down")

    h = _health(harvest=boom)
    assert h.unavailable == ["tax-loss harvesting"]
    assert "Unavailable today: tax-loss harvesting" in render_health_html(h)


def test_quiet_day_says_so():
    calm = build_portfolio_health(
        {"A": [{"ticker": "OK", "units": 1, "average_purchase_price": 1.0, "price": 2.0}]}
    )
    assert decision_items(calm) == []
    assert "Nothing needs a decision today" in render_decisions_html(calm)


def test_email_leads_with_decisions_and_orders_flagged_holdings_first():
    subject, body = build_email(REPORT, _health(), {})
    assert subject.endswith(": 5 to decide")  # the two sector notes don't count
    assert body.index("Decide today") < body.index("Portfolio health") < body.index("Sentiment")
    assert body.index("<h2>DOWN") < body.index("<h2>OK") < body.index("<h2>NEAR")

    subject, body = build_email(REPORT, None, {})
    assert subject.startswith("Portfolio Analysis - ") and "Decide today" not in body
