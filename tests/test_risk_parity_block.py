"""_format_risk_parity_block: inverse-volatility sizing input for the Sizer."""

from __future__ import annotations

from stock_analyzer.cli.discover_steps.helpers import _format_risk_parity_block
from stock_analyzer.models.llm import RankerOutput, RankerPick
from stock_analyzer.models.market import RealizedVolatility


def _pick(ticker: str) -> RankerPick:
    return RankerPick.model_construct(
        rank=1,
        ticker=ticker,
        one_liner="x",
        why_over_alternatives="x",
        conviction=8,
        time_horizon="6-12 months",
        sector_concentration_check="x",
        bull_thesis="x",
        what_youre_betting_on="x",
        scenarios=[],
    )


def _output(tickers: list[str]) -> RankerOutput:
    return RankerOutput.model_construct(
        picks=[_pick(t) for t in tickers],
        pairs_not_to_hold_together=[],
        full_text="original text",
    )


def _hv(ticker: str, hv_annualized: float) -> RealizedVolatility:
    return RealizedVolatility(ticker=ticker, hv_annualized=hv_annualized, sample_size=252)


def test_normalizes_to_100_pct_inverse_volatility():
    output = _output(["LOWVOL", "HIVOL"])
    hv_data = {"LOWVOL": _hv("LOWVOL", 0.10), "HIVOL": _hv("HIVOL", 0.40)}
    block = _format_risk_parity_block(output, hv_data)
    assert "LOWVOL" in block
    assert "HIVOL" in block
    # inverse-vol weights: 1/0.10=10, 1/0.40=2.5 -> 80%/20%
    assert "80%" in block
    assert "20%" in block


def test_missing_hv_for_a_pick_skips_it():
    output = _output(["A", "B", "C"])
    hv_data = {"A": _hv("A", 0.20), "B": _hv("B", 0.20)}  # C missing
    block = _format_risk_parity_block(output, hv_data)
    assert "A" in block
    assert "B" in block
    assert "C" not in block


def test_fewer_than_two_usable_picks_returns_empty():
    output = _output(["A", "B"])
    hv_data = {"A": _hv("A", 0.20)}  # only one usable
    assert _format_risk_parity_block(output, hv_data) == ""


def test_zero_hv_is_skipped_not_divided_by():
    output = _output(["A", "B"])
    hv_data = {"A": _hv("A", 0.0), "B": _hv("B", 0.20)}
    assert _format_risk_parity_block(output, hv_data) == ""


def test_non_ranker_output_returns_empty():
    assert _format_risk_parity_block(None, {}) == ""
