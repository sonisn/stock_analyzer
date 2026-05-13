"""Options chain fetching: SnapTrade primary, yfinance fallback.

The orchestrator (`fetch_chains`) tries SnapTrade per-ticker and falls
back to yfinance on None/error. Both providers return a normalized
`OptionChain` containing only OTM calls within the requested DTE band.

Failure of either provider for a given ticker is non-fatal — the
returned `OptionChain.source` is set to `"missing"` and the rebalancer
context just reads `Option chain: UNAVAILABLE` for that ticker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Protocol


@dataclass(frozen=True)
class OptionQuote:
    """One option strike/expiry row (calls only — puts not supported)."""
    strike: float
    expiry: date
    bid: float
    ask: float
    iv: float | None
    delta: float | None
    open_interest: int | None
    volume: int | None


@dataclass(frozen=True)
class OptionChain:
    """A ticker's filtered OTM call chain.

    `source` records which provider answered. `"missing"` is a valid
    state that downstream code handles — it does NOT raise.
    """
    ticker: str
    spot: float
    asof: datetime
    calls: list[OptionQuote] = field(default_factory=list)
    source: Literal["snaptrade", "yfinance", "missing"] = "missing"


class OptionChainProvider(Protocol):
    """Minimal contract every chain provider implements.

    Implementations MUST:
      - filter to OTM calls only (strike > spot)
      - filter to expiries within [today+dte_min, today+dte_max]
      - return None on any error (graceful degradation)
    """
    def fetch(
        self, ticker: str, dte_min: int, dte_max: int
    ) -> OptionChain | None:
        ...
