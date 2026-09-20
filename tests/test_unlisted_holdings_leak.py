"""Held, valued and taxed — but not analysed, and not writable.

A revoked CUSIP and a money-market sweep are real positions. They have no
price, no fundamentals and no news, so asking an LLM about them buys a
paragraph saying so, and offering to write calls on them is nonsense. The
daily email learned this; the rebalance path had not.
"""

from __future__ import annotations

from stock_analyzer.data.brokerage import is_listed_symbol
from stock_analyzer.reporting.health import is_cash_like

# As the broker reports them after the 2026-09-20 Schwab reconnect.
TARONIS = "876214206"
PLAN_FUND = "FGCCPS"


def test_what_counts_as_unwritable():
    assert not is_listed_symbol(TARONIS)
    assert not is_listed_symbol(PLAN_FUND, "other")
    assert is_listed_symbol("NVDA")
    assert is_cash_like("SPAXX", 1.0)
    assert is_cash_like("VMFXX", None)
    assert not is_cash_like("NVDA", 222.27)


def test_round_lot_coverage_excludes_sweeps_and_dead_listings():
    """SPAXX offered 208 round lots and a revoked CUSIP offered one more."""
    from stock_analyzer.discover.rebalance_cc import writable_positions

    positions = {
        "NVDA": {"units": 401.46},
        "SPAXX": {"units": 20845.0},
        TARONIS: {"units": 144.0},
        PLAN_FUND: {"units": 40.244},
    }
    spots = {"NVDA": 222.27, "SPAXX": 1.0, TARONIS: 0.0, PLAN_FUND: 0.0}
    assert sorted(writable_positions(positions, spots)) == ["NVDA"]
    # FGCCPS is a plain-looking six-letter symbol, so the ticker pattern
    # alone lets it through — it is the missing price that stops it.
    assert is_listed_symbol(PLAN_FUND)
    assert writable_positions({PLAN_FUND: {"units": 40.244}}, {PLAN_FUND: 0.0}) == {}
    assert "AMD" in writable_positions({"AMD": {"units": 31.0}}, {"AMD": 162.0})


def test_only_analyzable_holdings_are_sent_to_the_reviewer():
    """The filter existed in state as `holdings_tickers` and the reviewer
    read the unfiltered `holdings_positions` right past it."""
    import inspect

    from stock_analyzer.cli.rebalance import RebalancePipeline

    source = inspect.getsource(RebalancePipeline.step_review_holdings)
    assert "holdings_tickers" in source, (
        "step_review_holdings must review only the analyzable tickers, or a "
        "revoked CUSIP costs an LLM call to say it has no data"
    )
    assert 'positions=self.state["holdings_positions"],' not in source
