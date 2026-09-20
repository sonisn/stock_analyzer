"""Rolling a call the stock has run at.

On 2026-09-20 TSLA sat at $364.27 against a $400 strike expiring
2026-12-18, delta 0.39, with all 200 shares committed. "Roll it up and
out" is the right instruction and useless without the arithmetic: which
strike, which expiry, and does it still pay after buying the near call
back at the ask.
"""

from __future__ import annotations

from datetime import date, timedelta

from stock_analyzer.discover.cc_roll import (
    best_roll,
    roll_candidates,
    roll_suggestion,
)
from stock_analyzer.models.market import OptionChain, OptionQuote

TODAY = date(2026, 9, 20)
NEAR = date(2026, 12, 18)
SPOT = 364.27


def _q(strike, expiry, bid, ask, delta=0.20):
    return OptionQuote(
        strike=strike,
        expiry=expiry,
        bid=bid,
        ask=ask,
        iv=0.43,
        delta=delta,
        open_interest=1000,
        volume=100,
    )


def _chain(calls):
    return OptionChain(
        ticker="TSLA",
        spot=SPOT,
        asof=f"{TODAY}T16:00:00",
        calls=calls,
        source="tradier",
    )


CURRENT = _q(400.0, NEAR, 19.00, 19.55, delta=0.39)


def _args(chain, **over):
    base = {
        "ticker": "TSLA",
        "account": "Traditional IRA",
        "contracts": 2,
        "from_strike": 400.0,
        "from_expiry": NEAR,
        "chain": chain,
        "min_upside_pct": 15.0,
        "delta_max": 0.25,
        "today": TODAY,
    }
    return {**base, **over}


def test_a_paying_roll_is_priced_at_the_sides_you_can_trade():
    chain = _chain([CURRENT, _q(600.0, date(2027, 9, 17), 20.10, 21.00)])
    [candidate] = roll_candidates(**_args(chain))
    # buy back at the ASK, sell at the BID
    assert candidate.buyback_per_share == 19.55
    assert candidate.credit_per_share == 20.10
    assert round(candidate.net_usd, 2) == 110.00  # 0.55 x 100 x 2
    assert round(candidate.upside_pct) == 65


def test_a_strike_too_close_to_spot_is_not_a_roll():
    # $410 is only 13% above spot, inside the 15% floor.
    chain = _chain([CURRENT, _q(410.0, date(2027, 3, 19), 30.0, 31.0)])
    assert roll_candidates(**_args(chain)) == []


def test_a_barely_higher_strike_is_churn():
    # $410 clears nothing meaningful over the $400 already written.
    chain = _chain([CURRENT, _q(415.0, date(2027, 3, 19), 28.0, 29.0)])
    assert roll_candidates(**_args(chain)) == []


def test_a_replacement_above_the_delta_ceiling_is_rejected():
    chain = _chain([CURRENT, _q(600.0, date(2027, 9, 17), 20.10, 21.0, delta=0.44)])
    assert roll_candidates(**_args(chain)) == []


def test_without_a_quote_for_the_open_call_nothing_can_be_costed():
    chain = _chain([_q(600.0, date(2027, 9, 17), 20.10, 21.00)])
    assert roll_candidates(**_args(chain)) == []


def test_the_soonest_paying_roll_wins_not_the_biggest_credit():
    # Credit grows with time to expiry, so ranking on it alone always
    # answers "sell a 2028 call" — the most money and the longest
    # surrender of decisions.
    chain = _chain(
        [
            CURRENT,
            _q(600.0, date(2027, 9, 17), 20.10, 21.00),  # +$110
            _q(680.0, date(2028, 1, 21), 21.65, 22.50),  # +$420
        ]
    )
    best = best_roll(roll_candidates(**_args(chain)))
    assert best.to_expiry == date(2027, 9, 17)
    assert round(best.net_usd, 2) == 110.0


def test_a_roll_that_costs_money_is_only_offered_when_nothing_pays():
    chain = _chain([CURRENT, _q(600.0, date(2027, 3, 19), 15.00, 16.00)])
    best = best_roll(roll_candidates(**_args(chain)))
    assert best.net_per_share < 0


def test_the_window_is_preferred_and_the_lock_in_is_stated():
    chain = _chain([CURRENT, _q(600.0, date(2027, 9, 17), 20.10, 21.00)])
    text = roll_suggestion(
        ticker="TSLA",
        obligation={
            "legs": [
                {
                    "account": "Traditional IRA",
                    "contracts": 2,
                    "strike": 400.0,
                    "expiry": "2026-12-18",
                }
            ]
        },
        chain=chain,
        min_upside_pct=15.0,
        delta_max=0.25,
        dte_max=120,
        today=TODAY,
    )
    assert "Nothing in the usual 120-day window pays" in text
    assert "net credit of $110" in text
    assert "capped until 2027-09-17" in text
    assert "keeps the shares" in text


def test_a_roll_inside_the_window_is_named_plainly():
    soon = TODAY + timedelta(days=100)
    chain = _chain([CURRENT, _q(500.0, soon, 21.00, 22.00)])
    text = roll_suggestion(
        ticker="TSLA",
        obligation={
            "legs": [
                {
                    "account": "Traditional IRA",
                    "contracts": 2,
                    "strike": 400.0,
                    "expiry": "2026-12-18",
                }
            ]
        },
        chain=chain,
        min_upside_pct=15.0,
        delta_max=0.25,
        dte_max=120,
        today=TODAY,
    )
    assert text.startswith("Roll the 2 TSLA $400 call(s)")
    assert "Nothing in" not in text


def test_the_assignment_warning_carries_the_roll():
    from stock_analyzer.reporting.health import build_portfolio_health, decision_items

    health = build_portfolio_health(
        {"Traditional IRA": [{"ticker": "TSLA", "units": 200, "price": SPOT}]},
        covered_calls={
            "TSLA": {
                "contracts": 2,
                "shares_committed": 200.0,
                "by_account": {"Traditional IRA": 2},
                "legs": [],
                "next_expiry": "2026-12-18",
                "lowest_strike": 400.0,
            }
        },
        roll_ideas={"TSLA": "Roll the 2 TSLA $400 call(s) up to $600 for a net credit of $110."},
    )
    item = next(i for i in decision_items(health) if i["label"] == "CALL ASSIGNMENT")
    assert "net credit of $110" in item["text"]
    assert "Roll the call up or out if you mean to keep them" not in item["text"]
