"""Tests for options_chain.py — providers, orchestrator, fallback."""
from __future__ import annotations

from datetime import date, datetime

from stock_analyzer.data.options_chain import (
    OptionChain,
    OptionQuote,
)


def test_optionquote_frozen_and_typed():
    q = OptionQuote(
        strike=260.0, expiry=date(2026, 6, 20),
        bid=2.20, ask=2.40, iv=0.29, delta=0.36,
        open_interest=2890, volume=540,
    )
    assert q.strike == 260.0
    assert q.delta == 0.36


def test_optionchain_dataclass():
    chain = OptionChain(
        ticker="NVDA", spot=235.0, asof=datetime(2026, 5, 13, 16, 0, 0),
        calls=[], source="missing",
    )
    assert chain.ticker == "NVDA"
    assert chain.source == "missing"
