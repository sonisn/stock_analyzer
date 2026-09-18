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


def test_annualized_3_to_5_year_targets():
    # 3-5 years (4 yr midpoint), 40% vol -> a CAGR's sigma is ~20%/yr, so
    # +30%/yr is fine and +90%/yr is >4 sigma; >100%/yr is a unit error.
    hv = RealizedVolatility(ticker="NVDA", hv_annualized=0.40, sample_size=252)

    def check(target: float) -> list[str]:
        scen = Scenario(label="bull", probability=0.4, target_return_pct=target, rationale="x")
        return validate_pick_scenarios(_pick([scen], "3-5 years"), price=100.0, hv=hv)

    assert check(30.0) == []
    assert any("+90%/yr over 3-5 years" in w for w in check(90.0))
    assert any("implausibly large" in w for w in check(120.0))


def test_calibration_grades_annualized_forecasts_at_one_year():
    from stock_analyzer.discover.calibration import _Forecast

    def horizon(time_horizon: str | None) -> int:
        return _Forecast(
            ticker="X",
            pick_date="2026-01-01",
            age_days=0,
            conviction=7,
            ev_pct=10.0,
            entry_price=1.0,
            scenarios={},
            time_horizon=time_horizon,
        ).ev_horizon_days

    assert horizon("3-5 years") == 365
    assert horizon("6-12 months") == 270
    assert horizon(None) == 270
