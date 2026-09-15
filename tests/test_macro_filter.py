"""Macro/regime veto: suppress high-momentum picks in a risk-off regime."""

from __future__ import annotations

from stock_analyzer.discover.macro_filter import apply_macro_veto
from stock_analyzer.models.llm import RankerOutput, RankerPick

_RISK_OFF = {"yield_spread_10y_2y": -0.25, "vix": 35.0}
_RISK_ON = {"yield_spread_10y_2y": 0.80, "vix": 14.0}


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


def test_risk_on_regime_never_suppresses_anything():
    output = _output(["HYPE"])
    technicals = {"HYPE": {"rs_6mo": 0.50}}
    result, reasons = apply_macro_veto(output, _RISK_ON, technicals)
    assert reasons == []
    assert result is output


def test_risk_off_regime_suppresses_high_momentum_picks():
    output = _output(["HYPE", "STEADY"])
    technicals = {"HYPE": {"rs_6mo": 0.50}, "STEADY": {"rs_6mo": 0.02}}
    result, reasons = apply_macro_veto(output, _RISK_OFF, technicals)
    tickers = [p.ticker for p in result.picks]
    assert tickers == ["STEADY"]
    assert len(reasons) == 1
    assert "HYPE" in reasons[0]
    assert "MACRO VETO" in result.full_text


def test_risk_off_regime_with_no_momentum_picks_is_a_noop():
    output = _output(["STEADY"])
    technicals = {"STEADY": {"rs_6mo": 0.02}}
    result, reasons = apply_macro_veto(output, _RISK_OFF, technicals)
    assert reasons == []
    assert result is output


def test_missing_macro_data_is_treated_as_not_risk_off():
    output = _output(["HYPE"])
    technicals = {"HYPE": {"rs_6mo": 0.50}}
    result, reasons = apply_macro_veto(output, None, technicals)
    assert reasons == []
    assert result is output


def test_missing_technicals_for_a_ticker_does_not_suppress_it():
    output = _output(["UNKNOWN"])
    result, reasons = apply_macro_veto(output, _RISK_OFF, {})
    assert reasons == []
    assert [p.ticker for p in result.picks] == ["UNKNOWN"]
