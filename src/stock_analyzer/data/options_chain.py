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
from datetime import date, datetime, timedelta
from typing import Literal, Protocol

import yfinance as yf

from ..logging import get_logger

logger = get_logger(__name__)


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


class YFinanceChain:
    """yfinance-backed options chain provider.

    yfinance does not expose Greeks; `delta` is always None. The
    rebalancer's prompt is robust to that — it falls back to comparing
    strike vs spot when delta is missing.
    """

    def fetch(
        self, ticker: str, dte_min: int, dte_max: int
    ) -> OptionChain | None:
        try:
            t = yf.Ticker(ticker)
            spot = float(t.fast_info.last_price)
        except Exception as e:
            logger.info("yfinance chain miss for %s (%s)", ticker, e)
            return None

        today = date.today()
        lo = today + timedelta(days=dte_min)
        hi = today + timedelta(days=dte_max)
        calls: list[OptionQuote] = []
        try:
            expiries = tuple(t.options)
        except Exception as e:
            logger.info("yfinance no expiries for %s (%s)", ticker, e)
            return OptionChain(
                ticker=ticker, spot=spot, asof=datetime.now(),
                calls=[], source="yfinance",
            )

        for e_str in expiries:
            try:
                expiry = date.fromisoformat(e_str)
            except ValueError:
                continue
            if expiry < lo or expiry > hi:
                continue
            try:
                df = t.option_chain(e_str).calls
            except Exception as ex:
                logger.info("yfinance chain row miss %s@%s (%s)", ticker, e_str, ex)
                continue
            for _, row in df.iterrows():
                strike = float(row["strike"])
                if strike <= spot:  # OTM calls only
                    continue
                calls.append(OptionQuote(
                    strike=strike,
                    expiry=expiry,
                    bid=float(row.get("bid") or 0.0),
                    ask=float(row.get("ask") or 0.0),
                    iv=float(row["impliedVolatility"]) if row.get("impliedVolatility") is not None else None,
                    delta=None,  # yfinance does not provide Greeks
                    open_interest=int(row.get("openInterest") or 0),
                    volume=int(row.get("volume") or 0),
                ))

        return OptionChain(
            ticker=ticker, spot=spot, asof=datetime.now(),
            calls=calls, source="yfinance",
        )
