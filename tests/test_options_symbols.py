"""Tests for OCC option-symbol parser."""
from __future__ import annotations

from datetime import date

import pytest

from stock_analyzer.data.options_symbols import (
    OCCParseError,
    is_option_symbol,
    parse_occ,
)


def test_parse_call():
    p = parse_occ("NVDA  260620C00250000")
    assert p.ticker == "NVDA"
    assert p.expiry == date(2026, 6, 20)
    assert p.option_type == "C"
    assert p.strike == 250.0


def test_parse_put():
    p = parse_occ("AAPL  260718P00200500")
    assert p.option_type == "P"
    assert p.strike == 200.5


def test_parse_long_underlying():
    # 6-char underlying with no padding required.
    p = parse_occ("BRKB  260117C00450000")
    assert p.ticker == "BRKB"


def test_is_option_symbol_true():
    assert is_option_symbol("NVDA  260620C00250000")


def test_is_option_symbol_false_for_equity():
    assert not is_option_symbol("NVDA")
    assert not is_option_symbol("BRK.B")


def test_parse_rejects_garbage():
    with pytest.raises(OCCParseError):
        parse_occ("not-an-option")


def test_parse_rejects_wrong_length():
    with pytest.raises(OCCParseError):
        parse_occ("NVDA260620C00250000")  # missing padding


def test_six_char_root_with_no_padding_rejected():
    """OCC requires at least one space between root and date, even for
    6-char roots. Brokers that strip the padding entirely are not
    supported — repad before parsing."""
    with pytest.raises(OCCParseError):
        parse_occ("BRKBXX260117C00450000")  # 6 chars, no space → rejected
