"""Sale decisions when the shares are already promised.

On 2026-09-20 twelve short calls were open in the Traditional IRA, all
expiring 2026-12-18: NVDA 4 (400 of 401 shares), GOOGL 2 (200 of 200),
TSLA 2 (200 of 200), BE 3 (300 of 384), AVGO 1 (100 of 162). Three
positions were fully committed, and nothing outside the rebalancer knew.
"""

from __future__ import annotations

from stock_analyzer.reporting.health import (
    assignment_items,
    build_portfolio_health,
    covered_call_clause,
    decision_items,
    render_covered_calls_html,
    suggestion_rows,
)

HOLDINGS = {"Traditional IRA": [{"ticker": "NVDA", "units": 401, "price": 222.27}]}

CALLS = {
    "NVDA": {
        "contracts": 4,
        "shares_committed": 400.0,
        "by_account": {"Traditional IRA": 4},
        "legs": [
            {"account": "Traditional IRA", "contracts": 4, "strike": 285.0, "expiry": "2026-12-18"}
        ],
        "next_expiry": "2026-12-18",
        "lowest_strike": 285.0,
    }
}


def _health(**kwargs):
    return build_portfolio_health(HOLDINGS, covered_calls=CALLS, **kwargs)


def test_the_clause_says_what_selling_would_take():
    clause = covered_call_clause(_health(), "NVDA")
    assert "4 covered call(s)" in clause
    assert "400 shares" in clause and "100% of the position" in clause
    assert "$285" in clause and "2026-12-18" in clause
    assert "buying those back" in clause


def test_a_holding_with_no_calls_gets_no_clause():
    assert covered_call_clause(_health(), "AMD") == ""
    assert covered_call_clause(build_portfolio_health(HOLDINGS), "NVDA") == ""


def test_a_broken_thesis_sale_carries_the_clause():
    health = _health(
        held_thesis_checks=lambda held: [
            {
                "ticker": "NVDA",
                "status": "BROKEN",
                "return_pct": -32.0,
                "signals": [{"text": "estimates cut", "severity": "warning"}],
            }
        ]
    )
    broken = next(i for i in decision_items(health) if i["label"] == "BROKEN")
    assert "covered call(s)" in broken["text"]


def test_a_drawdown_review_carries_it_too():
    health = build_portfolio_health(
        {
            "Traditional IRA": [
                {"ticker": "NVDA", "units": 401, "price": 222.27, "average_purchase_price": 400.0}
            ]
        },
        covered_calls=CALLS,
    )
    drawdown = next(i for i in decision_items(health) if i["label"] == "DRAWDOWN")
    assert "covered call(s)" in drawdown["text"]


def test_assignment_is_flagged_only_when_the_strike_is_close():
    # NVDA at $222.27 against a $285 strike is 28% away: not yet.
    assert assignment_items(_health()) == []

    near = {
        **CALLS,
        "TSLA": {
            **CALLS["NVDA"],
            "lowest_strike": 400.0,
            "shares_committed": 200.0,
            "contracts": 2,
        },
    }
    health = build_portfolio_health(
        {"IRA": [{"ticker": "TSLA", "units": 200, "price": 364.27}]}, covered_calls=near
    )
    items = assignment_items(health)
    assert len(items) == 1 and items[0]["ticker"] == "TSLA"
    assert "10% below" in items[0]["text"] and "200 shares" in items[0]["text"]
    assert "Roll the call" in items[0]["text"]


def test_a_position_already_above_its_strike_is_not_a_warning():
    # Past the strike it is not a risk to watch, it is done.
    above = {"X": {**CALLS["NVDA"], "lowest_strike": 100.0}}
    health = build_portfolio_health(
        {"IRA": [{"ticker": "X", "units": 400, "price": 150.0}]}, covered_calls=above
    )
    assert assignment_items(health) == []


def test_the_assignment_warning_is_graded_as_a_review_not_a_trade():
    near = {"TSLA": {**CALLS["NVDA"], "lowest_strike": 400.0, "shares_committed": 200.0}}
    health = build_portfolio_health(
        {"IRA": [{"ticker": "TSLA", "units": 200, "price": 364.27}]}, covered_calls=near
    )
    rows = suggestion_rows(health, today="2026-09-20")
    assert [r["action"] for r in rows] == ["REVIEW"]


def test_the_email_shows_what_is_promised():
    html = render_covered_calls_html(_health())
    assert "Covered calls written" in html
    assert "400 (100%)" in html and "$285" in html and "2026-12-18" in html
    assert "+28%" in html  # distance to the strike


def test_no_calls_means_no_section():
    assert render_covered_calls_html(build_portfolio_health(HOLDINGS)) == ""
