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
    # Only -20% or worse asks for a thesis re-check; NEAR (-18%) is not flagged.
    assert [(r["ticker"], round(r["pnl_pct"])) for r in h.drawdowns] == [("DOWN", -25)]
    tech = next(r for r in h.sectors if r["sector"] == "Technology")
    assert round(tech["pct"], 1) == 37.6 and tech["over"]
    assert h.unavailable == []


def test_decisions_are_ranked_by_urgency():
    items = decision_items(_health())
    assert [i["label"] for i in items] == [
        "DRAWDOWN",
        "TAX LOSS",
        "TARGET HIT",  # long-term: information, not a trade
        "EARNINGS",
        "OVER CAP",
        "OVER CAP",
    ]
    assert items[0]["text"].startswith("Re-check the long-term thesis for DOWN: -25.0% from cost")
    assert "isn't a reason to sell" in items[0]["text"]
    assert "nothing to do before the print" in items[3]["text"]
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
    assert subject.endswith(": 2 to decide")  # information items don't count
    assert body.index("Decide today") < body.index("Portfolio health") < body.index("Sentiment")
    assert body.index("<h2>DOWN") < body.index("<h2>OK") < body.index("<h2>NEAR")

    subject, body = build_email(REPORT, None, {})
    assert subject.startswith("Portfolio Analysis - ") and "Decide today" not in body


def test_every_stock_keeps_its_chart_when_decisions_reorder_the_email():
    cids = {t: f"chart-{t}" for t in ("OK", "NEAR", "DOWN")}
    _, body = build_email(REPORT, _health(), cids)
    for t in cids:
        img = f'src="cid:chart-{t}"'
        assert body.count(img) == 1
        # The chart sits inside its own stock's section, after that heading.
        assert body.index(f"<h2>{t}") < body.index(img)
    assert body.index('src="cid:chart-DOWN"') < body.index('src="cid:chart-OK"')
