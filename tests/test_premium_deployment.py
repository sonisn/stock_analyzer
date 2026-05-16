"""Tests for deterministic CC reporting math."""
from __future__ import annotations

from stock_analyzer.discover.cc_render import (
    compute_premium_deployment,
    compute_premium_income,
    compute_round_lot_summary,
)
from stock_analyzer.models.portfolio import RoundLotCoverage
from stock_analyzer.models.rebalance import (
    OptionWrite,
    RebalanceAction,
    RebalancePlan,
)


def _ow(ticker: str, contracts: int, premium: float,
        account: str = "Test Account") -> OptionWrite:
    return OptionWrite(
        ticker=ticker, account=account,
        strike=200.0, expiry="2026-06-20",
        contracts=contracts, est_premium_per_share=premium,
        delta=0.4, assignment_probability=0.4,
    )


def test_premium_income_totals():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        option_writes=[
            _ow("NVDA", 3, 2.40),
            _ow("AAPL", 2, 3.20),
        ],
        full_text="…",
    )
    out = compute_premium_income(plan, slippage_buffer=0.10)
    assert out["gross_premium_usd"] == 1360.0
    assert out["slippage_buffer_usd"] == 136.0
    assert out["deployable_premium_usd"] == 1224.0
    assert len(out["rows"]) == 2


def test_premium_deployment_full_flow():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[
            RebalanceAction(action="WRITE_CALL", ticker="NVDA", sizing="3 contracts $260C"),
            RebalanceAction(action="ADD", ticker="AMZN", sizing="$1,400"),
            RebalanceAction(action="BUY", ticker="PLTR", sizing="$600"),
        ],
        option_writes=[_ow("NVDA", 3, 2.40)],
        full_text="…",
    )
    out = compute_premium_deployment(
        plan, cash_balance=850.0, slippage_buffer=0.10,
        stub_consolidation_usd=0.0,
    )
    assert out["deployable_premium_usd"] == 648.0
    assert out["existing_cash_usd"] == 850.0
    assert out["stub_consolidation_usd"] == 0.0
    assert out["total_dry_powder_usd"] == 1498.0
    assert {"ticker": "AMZN", "action": "ADD", "sizing": "$1,400"} in out["deployments"]


def test_premium_deployment_with_stub_consolidation_row():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[],
        option_writes=[_ow("NVDA", 3, 2.40)],
        full_text="…",
    )
    out = compute_premium_deployment(
        plan, cash_balance=850.0, slippage_buffer=0.10,
        stub_consolidation_usd=10500.0,
    )
    assert out["stub_consolidation_usd"] == 10500.0
    assert out["total_dry_powder_usd"] == 11_998.0


def test_premium_income_row_carries_account():
    from stock_analyzer.discover.cc_render import compute_premium_income
    from stock_analyzer.models.rebalance import (
        OptionWrite,
        RebalancePlan,
    )

    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[],
        option_writes=[OptionWrite(
            ticker="NVDA", account="Fidelity IRA",
            strike=260.0, expiry="2026-06-20",
            contracts=2, est_premium_per_share=2.30,
            delta=0.36, assignment_probability=0.36,
        )],
        full_text="…",
    )
    out = compute_premium_income(plan, slippage_buffer=0.10)
    assert len(out["rows"]) == 1
    assert out["rows"][0]["account"] == "Fidelity IRA"


def test_round_lot_summary():
    coverage = {
        "TSLA": RoundLotCoverage(
            ticker="TSLA", shares=335, round_lots=3, stub_shares=35,
            stub_dollar_value=10500.0, to_next_lot_shares=65,
            to_next_lot_cost=19500.0,
        ),
        "AAPL": RoundLotCoverage(
            ticker="AAPL", shares=215, round_lots=2, stub_shares=15,
            stub_dollar_value=3225.0, to_next_lot_shares=85,
            to_next_lot_cost=18275.0,
        ),
        "NVDA": RoundLotCoverage(
            ticker="NVDA", shares=200, round_lots=2, stub_shares=0,
            stub_dollar_value=0.0, to_next_lot_shares=0, to_next_lot_cost=0.0,
        ),
    }
    out = compute_round_lot_summary(coverage)
    tickers = [r["ticker"] for r in out["rows"]]
    assert "NVDA" not in tickers
    assert tickers == ["TSLA", "AAPL"]
    assert out["stub_pool_total_usd"] == 13725.0
