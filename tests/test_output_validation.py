"""Sanity-checking Ranker scenarios against price/volatility hard data."""

from __future__ import annotations

from stock_analyzer.discover.output_validation import validate_pick_scenarios
from stock_analyzer.models.llm import RankerPick, Scenario
from stock_analyzer.models.market import RealizedVolatility


def _pick(scenarios: list[Scenario], time_horizon: str = "6-12 months") -> RankerPick:
    return RankerPick.model_construct(
        rank=1,
        ticker="NVDA",
        one_liner="x",
        why_over_alternatives="x",
        conviction=8,
        time_horizon=time_horizon,
        sector_concentration_check="x",
        bull_thesis="x",
        what_youre_betting_on="x",
        scenarios=scenarios,
    )


def test_plausible_bull_target_produces_no_warning():
    # ~35% annualized vol, 9mo horizon -> ~30% horizon sigma. A +40% bull
    # target is well under the 4x-sigma flag threshold.
    pick = _pick([Scenario(label="bull", probability=0.4, target_return_pct=40.0, rationale="x")])
    hv = RealizedVolatility(ticker="NVDA", hv_annualized=0.35, sample_size=252)
    assert validate_pick_scenarios(pick, price=100.0, hv=hv) == []


def test_extreme_bull_target_relative_to_volatility_is_flagged():
    pick = _pick([Scenario(label="bull", probability=0.4, target_return_pct=400.0, rationale="x")])
    hv = RealizedVolatility(ticker="NVDA", hv_annualized=0.20, sample_size=252)
    warnings = validate_pick_scenarios(pick, price=100.0, hv=hv)
    assert any("bull target" in w for w in warnings)


def test_implausible_return_pct_flagged_regardless_of_volatility_data():
    pick = _pick([Scenario(label="base", probability=0.4, target_return_pct=250.0, rationale="x")])
    warnings = validate_pick_scenarios(pick, price=100.0, hv=None)
    assert any("implausibly large" in w for w in warnings)


def test_missing_volatility_data_skips_the_sigma_check_but_not_unit_check():
    pick = _pick([Scenario(label="bull", probability=0.4, target_return_pct=40.0, rationale="x")])
    assert validate_pick_scenarios(pick, price=100.0, hv=None) == []


def test_missing_price_skips_the_unit_check():
    pick = _pick([Scenario(label="base", probability=0.4, target_return_pct=250.0, rationale="x")])
    assert validate_pick_scenarios(pick, price=None, hv=None) == []
