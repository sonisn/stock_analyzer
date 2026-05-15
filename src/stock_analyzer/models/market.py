"""Pydantic models for market data — OCC option symbols, option chains,
and realized volatility.

These are the value objects produced by the ``data/`` providers
(``options_symbols``, ``options_chain``, ``historical_volatility``) and
consumed across the discover + rebalance pipelines. Each model is
frozen so they're safe to share across threads and to use as dict keys
or set members downstream.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OptionType = Literal["C", "P"]


class OCCParseError(ValueError):
    """Raised when a string does not look like an OCC option symbol."""


class ParsedOCC(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    expiry: date
    option_type: OptionType
    strike: float


class OptionQuote(BaseModel):
    """One option strike/expiry row (calls only — puts not supported)."""

    model_config = ConfigDict(frozen=True)

    strike: float
    expiry: date
    bid: float
    ask: float
    iv: float | None
    delta: float | None
    open_interest: int | None
    volume: int | None


class OptionChain(BaseModel):
    """A ticker's filtered OTM call chain.

    ``source`` records which provider answered. ``"missing"`` is a valid
    state that downstream code handles — it does NOT raise.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    spot: float
    asof: datetime
    calls: list[OptionQuote] = Field(default_factory=list)
    source: Literal["tradier", "yfinance", "missing"] = "missing"


class RealizedVolatility(BaseModel):
    """Annualized realized volatility for one ticker."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    hv_annualized: float           # e.g. 0.27 = 27%
    sample_size: int               # number of daily returns used


__all__ = [
    "OptionType",
    "OCCParseError",
    "ParsedOCC",
    "OptionQuote",
    "OptionChain",
    "RealizedVolatility",
]
