"""Core and satellite: an index-fund core the rebalancer builds in steps."""

from __future__ import annotations

from types import SimpleNamespace

from stock_analyzer.discover import goal_projection as gp
from stock_analyzer.discover.core_satellite import check_core, core_block, core_status
from stock_analyzer.discover.rebalancer import _decide_prompt
from stock_analyzer.reporting.plan_check import render_goal_html
from tests.test_plan_check import _returns

POSITIONS = {
    "AVGO": {"value": 300_000.0},
    "NVDA": {"value": 150_000.0},
    "FXAIX": {"value": 20_000.0},  # an S&P 500 fund held in the 401(k)
    "SPAXX": {"value": 0.0},
}


def test_core_is_off_by_default_and_counts_every_index_fund():
    assert core_status(POSITIONS, 30_000, fund="VOO", target_pct=0) is None
    s = core_status(POSITIONS, 30_000, fund="voo", target_pct=40)
    assert s.fund == "VOO" and s.holdings == ("FXAIX",)
    assert s.total_value == 500_000 and round(s.core_pct, 1) == 4.0
    assert s.shortfall_usd == 180_000
    # At most 10% of the portfolio per rebalance.
    assert s.step_usd(10) == 50_000 and s.step_usd(50) == 180_000


def test_block_asks_for_the_step_funded_tax_free_first():
    s = core_status(POSITIONS, 30_000, fund="VOO", target_pct=40)
    block = core_block(s, step_pct=10)
    assert "~$50,000 into VOO" in block and "before any satellite BUY" in block
    assert "tax-advantaged account" in block and "never a sale in a taxable account" in block
    assert "single-position cap does not apply" in block
    met = core_status({"VOO": {"value": 100.0}}, 0, fund="VOO", target_pct=40)
    assert "Target met" in core_block(met, step_pct=10)
    assert core_block(None, step_pct=10) == ""


def test_the_block_reaches_the_rebalancer_prompt():
    s = core_status(POSITIONS, 30_000, fund="VOO", target_pct=40)
    prompt, _ = _decide_prompt(
        holdings_reviews={},
        picks_text="",
        cash_available=30_000,
        core_block=core_block(s, step_pct=10),
    )
    assert "CORE INDEX FUND" in prompt


def test_a_plan_that_skips_a_due_core_step_is_reported():
    s = core_status(POSITIONS, 30_000, fund="VOO", target_pct=40)
    skipped = SimpleNamespace(actions=[SimpleNamespace(action="BUY", ticker="NVDA")])
    (warning,) = check_core(skipped, s, step_pct=10)
    assert "does not buy VOO" in warning and "~$50,000" in warning
    done = SimpleNamespace(actions=[SimpleNamespace(action="ADD", ticker="voo")])
    assert check_core(done, s, step_pct=10) == []
    assert check_core(skipped, None, step_pct=10) == []


def test_more_core_means_calmer_and_better_odds_for_a_hot_portfolio():
    mixes = gp.core_mix(
        weights={"AAA": 100_000.0},
        returns=_returns(),
        start_value=100_000,
        months=60,
        monthly_contribution=500,
        expected_return=0.07,
        target=150_000,
    )
    assert [m["core_share"] for m in mixes] == list(gp.CORE_SHARES)
    # Not monotone in between: an uncorrelated mix can swing less than SPY alone.
    assert mixes[0]["volatility"] > mixes[2]["volatility"] > mixes[-1]["volatility"] * 0.5
    assert mixes[-1]["odds"] > mixes[0]["odds"]


def test_plan_check_shows_the_mixes_and_marks_the_target():
    r = _returns()
    kw = dict(
        returns=r,
        start_value=100_000,
        months=60,
        monthly_contribution=500,
        expected_return=0.07,
        target=150_000,
    )
    p = gp.project(weights={"AAA": 1.0}, **kw)
    mixes = gp.core_mix(weights={"AAA": 1.0}, **kw)
    body = render_goal_html(
        p, goal_date=None, contribution_note="x", core_mixes=mixes, core_target_pct=50
    )
    assert "With part of it in an index fund" in body
    assert "50% in the index fund — your target" in body
    assert "CORE_TARGET_PCT (now 50%)" in body
    off = render_goal_html(p, goal_date=None, contribution_note="x", core_mixes=mixes)
    assert "CORE_TARGET_PCT is 0 (off)" in off
