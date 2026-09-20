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


# --- where new money would unlock another call -----------------------------------


def test_headroom_is_measured_per_account():
    # 60 uncovered shares in one account and 60 in another are not a
    # contract: a call is written against shares in one place.
    from stock_analyzer.reporting.health import call_headroom

    health = build_portfolio_health(
        {
            "Traditional IRA": [{"ticker": "AVGO", "units": 60, "price": 357.61}],
            "Robinhood Individual": [{"ticker": "AVGO", "units": 60, "price": 357.61}],
        }
    )
    rows = {r["account"]: r for r in call_headroom(health)}
    assert rows["Traditional IRA"]["writable_contracts"] == 0
    assert rows["Traditional IRA"]["shares_to_next_lot"] == 40


def test_shares_already_promised_do_not_count_as_headroom():
    # AVGO on 2026-09-20: 162 shares, one call written, 62 free.
    health = build_portfolio_health(
        {"Traditional IRA": [{"ticker": "AVGO", "units": 162, "price": 357.61}]},
        covered_calls={
            "AVGO": {
                "contracts": 1,
                "shares_committed": 100.0,
                "by_account": {"Traditional IRA": 1},
                "legs": [],
                "next_expiry": "2026-12-18",
                "lowest_strike": 480.0,
            }
        },
    )
    from stock_analyzer.reporting.health import call_headroom, headroom_clause

    [row] = call_headroom(health)
    assert row["uncovered"] == 62
    assert row["writable_contracts"] == 0 and row["shares_to_next_lot"] == 38
    clause = headroom_clause(health, "AVGO")
    assert "38 more shares" in clause and "Traditional IRA" in clause
    assert "13,589" in clause  # 38 x $357.61


def test_a_full_uncovered_lot_can_be_written_today():
    from stock_analyzer.reporting.health import call_headroom, headroom_clause

    health = build_portfolio_health(
        {"Robinhood Individual": [{"ticker": "MRVL", "units": 240, "price": 244.24}]}
    )
    [row] = call_headroom(health)
    assert row["writable_contracts"] == 2 and row["uncovered"] == 240
    assert "enough to write 2 more call(s)" in headroom_clause(health, "MRVL")

    item = next(i for i in decision_items(health) if i["label"] == "CALL HEADROOM")
    assert "240 MRVL shares" in item["text"] and item["priority"] == 4


def test_a_fully_covered_position_has_no_headroom():
    from stock_analyzer.reporting.health import call_headroom, headroom_clause

    health = _health()  # NVDA 401 shares, 400 promised
    [row] = call_headroom(health)
    assert row["uncovered"] == 401 - 400
    assert row["writable_contracts"] == 0
    # 99 short of another lot, so it is still worth naming
    assert "99 more shares" in headroom_clause(health, "NVDA")


def test_an_add_on_idea_names_what_it_would_unlock():
    health = build_portfolio_health(
        {"Robinhood Individual": [{"ticker": "ARM", "units": 55, "price": 275.61}]},
        add_on=lambda **kwargs: [{"ticker": "ARM", "off_high_pct": -22.0}],
    )
    item = next(i for i in decision_items(health) if i["label"] == "ADD ON DIP")
    assert "45 more shares" in item["text"]
    assert "covered call" in item["text"]


def test_an_unapproved_account_is_a_blocked_opportunity_not_a_silent_one():
    # The Schwab HSA holds 73 uncovered BE shares and cannot trade
    # options until a form is filed. Saying nothing would read as
    # "nothing to do here".
    from stock_analyzer.reporting.health import (
        blocked_headroom,
        call_headroom,
        headroom_clause,
    )

    health = build_portfolio_health(
        {
            "HSA Brokerage ...263": [{"ticker": "BE", "units": 73, "price": 265.63}],
            "Traditional IRA": [{"ticker": "BE", "units": 300, "price": 265.63}],
        },
        options_accounts=("Traditional IRA",),
    )
    rows = {r["account"]: r for r in call_headroom(health)}
    assert rows["HSA Brokerage ...263"]["options_approved"] is False
    assert rows["Traditional IRA"]["options_approved"] is True

    # The add-on clause only names accounts that could act on it.
    assert "Traditional IRA" in headroom_clause(health, "BE")
    assert "HSA" not in headroom_clause(health, "BE")

    [blocked] = blocked_headroom(health)
    assert "not approved for options" in blocked["text"]
    assert "27 shares" in blocked["text"] and "options application" in blocked["text"]

    item = next(i for i in decision_items(health) if i["label"] == "OPTIONS NOT APPROVED")
    assert item["priority"] == 4


def test_a_writable_lot_in_an_unapproved_account_is_not_offered_as_income():
    from stock_analyzer.reporting.health import blocked_headroom

    health = build_portfolio_health(
        {"HSA Brokerage ...263": [{"ticker": "BE", "units": 200, "price": 265.63}]},
        options_accounts=("Traditional IRA",),
    )
    assert [i for i in decision_items(health) if i["label"] == "CALL HEADROOM"] == []
    [blocked] = blocked_headroom(health)
    assert "2 contract(s) could be written" in blocked["text"]


def test_one_line_per_account_not_per_holding():
    from stock_analyzer.reporting.health import blocked_headroom

    health = build_portfolio_health(
        {
            "HSA Brokerage ...263": [
                {"ticker": "BE", "units": 73, "price": 265.63},
                {"ticker": "OKLO", "units": 60, "price": 38.0},
            ]
        },
        options_accounts=("Traditional IRA",),
    )
    assert len(blocked_headroom(health)) == 1  # one form to file, one line


def test_no_allowlist_means_every_account_is_approved():
    from stock_analyzer.reporting.health import blocked_headroom, call_headroom

    health = build_portfolio_health(
        {"HSA Brokerage ...263": [{"ticker": "BE", "units": 73, "price": 265.63}]}
    )
    assert call_headroom(health)[0]["options_approved"] is True
    assert blocked_headroom(health) == []


# --- what the market is rewarding --------------------------------------------------


ROTATION = {
    "lookback_months": 6,
    # Fractions, the way fetch_sector_returns actually returns them —
    # the first version of this fixture used percents and let a renderer
    # that printed "+0.2%" for a sector up a fifth pass.
    "returns_by_sector": {
        "Technology": 0.214,
        "Healthcare": 0.128,
        "Financial Services": 0.096,
        "Industrials": 0.041,
        "Utilities": -0.062,
    },
    "leaders": ["Technology", "Healthcare", "Financial Services"],
    "laggards": ["Utilities", "Communication Services", "Consumer Cyclical"],
}


def test_the_email_shows_leaders_laggards_and_where_you_sit():
    from stock_analyzer.reporting.health import (
        build_portfolio_health,
        render_sector_rotation_html,
    )

    health = build_portfolio_health(
        {"IRA": [{"ticker": "NVDA", "units": 10, "price": 222.27}]},
        sector_rotation=ROTATION,
        sector_of=lambda tickers: {"NVDA": "Technology"},
    )
    html = render_sector_rotation_html(health)
    assert "Sector rotation (6 months)" in html
    # ordered by return, best first
    assert html.index("Technology") < html.index("Healthcare") < html.index("Utilities")
    assert "+21.4%" in html and "-6.2%" in html
    assert "leading" in html and "lagging" in html
    # the sector actually held is marked
    tech_row = html[html.index("Technology") : html.index("Healthcare")]
    assert "yes" in tech_row
    utilities_row = html[html.index("Utilities") :]
    assert "yes" not in utilities_row


def test_no_rotation_data_renders_nothing():
    from stock_analyzer.reporting.health import (
        build_portfolio_health,
        render_sector_rotation_html,
    )

    assert render_sector_rotation_html(build_portfolio_health({})) == ""


def test_a_cash_sweep_is_never_covered_call_headroom():
    # SPAXX held 20,846 "shares" at $1.00 and was offered as 208
    # writable contracts. The quote type catches it, but only when a
    # caller passes one — the quarterly review does not.
    from stock_analyzer.reporting.health import build_portfolio_health, call_headroom, is_cash_like

    health = build_portfolio_health(
        {
            "Traditional IRA": [
                {"ticker": "SPAXX", "units": 20846, "price": 1.0},
                {"ticker": "NVDA", "units": 401, "price": 222.27},
            ]
        }
    )
    assert [r["ticker"] for r in call_headroom(health)] == ["NVDA"]

    # by name, and by a NAV pinned to a dollar for one not on the list
    assert is_cash_like("SPAXX", 1.0) and is_cash_like("FDRXX", None)
    assert is_cash_like("XXXXX", 0.999)
    assert not is_cash_like("NVDA", 222.27)
    assert not is_cash_like("SOFI", None)
