"""Options chain fetching: Tradier primary, yfinance fallback.

The orchestrator (`fetch_chains`) tries Tradier per-ticker and falls
back to yfinance on None/error. Both providers return a normalized
`OptionChain` containing only OTM calls within the requested DTE band.

Failure of either provider for a given ticker is non-fatal — the
returned `OptionChain.source` is set to `"missing"` and the rebalancer
context just reads `Option chain: UNAVAILABLE` for that ticker.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Protocol

import requests
import yfinance as yf

from ..config import Settings
from ..logging import get_logger
from ..models.market import OptionChain, OptionQuote

logger = get_logger(__name__)

# Re-export the model classes here so legacy import paths continue
# working during Phase 1. Group C strips this shim once every callsite
# has been migrated.
__all__ = [
    "OptionChain", "OptionQuote", "OptionChainProvider",
    "YFinanceChain", "TradierChain", "fetch_chains",
]


def _safe_float(v: object) -> float | None:
    """Coerce to float, returning None for None / NaN / Inf / unparseable.
    Used at provider boundary because yfinance returns NaN for low-volume
    strikes, which crashes downstream arithmetic and int() conversion."""
    if v is None:
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _safe_int(v: object) -> int:
    """Coerce to int, returning 0 for None / NaN / Inf / unparseable."""
    f = _safe_float(v)
    return int(f) if f is not None else 0


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
            spot = _safe_float(t.fast_info.last_price)
        except Exception as e:
            logger.info("yfinance chain miss for %s (%s)", ticker, e)
            return None
        if spot is None or spot <= 0:
            logger.info(
                "yfinance returned invalid spot for %s (NaN / 0 / negative); "
                "skipping ticker", ticker,
            )
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
                strike = _safe_float(row.get("strike"))
                # NaN / None / 0 / negative strikes are nonsense; skip them.
                if strike is None or strike <= 0:
                    continue
                if strike <= spot:  # OTM calls only
                    continue
                calls.append(OptionQuote(
                    strike=strike,
                    expiry=expiry,
                    bid=_safe_float(row.get("bid")) or 0.0,
                    ask=_safe_float(row.get("ask")) or 0.0,
                    iv=_safe_float(row.get("impliedVolatility")),
                    delta=None,  # yfinance does not provide Greeks
                    open_interest=_safe_int(row.get("openInterest")),
                    volume=_safe_int(row.get("volume")),
                ))

        return OptionChain(
            ticker=ticker, spot=spot, asof=datetime.now(),
            calls=calls, source="yfinance",
        )


class TradierChain:
    """Tradier-backed options chain provider.

    Two-step fetch: GET expirations → GET chain per in-band expiry with
    greeks=true. Returns OptionChain with source="tradier" populated with
    accurate delta/iv from Tradier (ORATS-backed).

    Returns None on any failure — auth missing, network error, payload
    shape unexpected. The orchestrator falls back to yfinance on None.
    """

    _TIMEOUT_SECONDS = 10

    def __init__(self) -> None:
        # Cache "is provider configured" once per instance so we don't
        # spam logs across multiple ticker fetches.
        self._configured: bool | None = None

    def fetch(
        self, ticker: str, dte_min: int, dte_max: int
    ) -> OptionChain | None:
        s = Settings()  # type: ignore[call-arg]
        if not s.tradier_api_key:
            if self._configured is None:
                logger.info(
                    "Tradier chain provider not configured "
                    "(TRADIER_API_KEY unset). Falling back to yfinance."
                )
                self._configured = False
            return None
        self._configured = True

        headers = {
            "Authorization": f"Bearer {s.tradier_api_key}",
            "Accept": "application/json",
        }

        # Step 1: expirations
        try:
            resp = requests.get(
                f"{s.tradier_base_url}/markets/options/expirations",
                params={"symbol": ticker, "includeAllRoots": "true"},
                headers=headers,
                timeout=self._TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            payload = resp.json() or {}
        except Exception as e:
            logger.warning("Tradier expirations fetch failed for %s: %s", ticker, e)
            return None

        expirations = self._extract_expirations(payload)
        if not expirations:
            logger.info("Tradier returned no expirations for %s", ticker)
            return OptionChain(
                ticker=ticker, spot=0.0, asof=datetime.now(),
                calls=[], source="tradier",
            )

        today = date.today()
        lo = today + timedelta(days=dte_min)
        hi = today + timedelta(days=dte_max)
        in_band: list[date] = []
        for d_str in expirations:
            try:
                d = date.fromisoformat(d_str)
            except (ValueError, TypeError):
                continue
            if lo <= d <= hi:
                in_band.append(d)

        if not in_band:
            return OptionChain(
                ticker=ticker, spot=0.0, asof=datetime.now(),
                calls=[], source="tradier",
            )

        # Step 2: fetch spot for filtering ITM calls
        spot = self._fetch_spot(ticker, headers, s.tradier_base_url) or 0.0

        calls: list[OptionQuote] = []
        for expiry in in_band:
            chain_rows = self._fetch_chain_for_expiry(
                ticker, expiry, headers, s.tradier_base_url,
            )
            for row in chain_rows:
                if row.get("option_type") != "call":
                    continue
                strike = _safe_float(row.get("strike")) or 0.0
                if strike <= 0 or (spot > 0 and strike <= spot):
                    continue  # OTM calls only when spot known; else keep all
                greeks = row.get("greeks") or {}
                calls.append(OptionQuote(
                    strike=strike,
                    expiry=expiry,
                    bid=_safe_float(row.get("bid")) or 0.0,
                    ask=_safe_float(row.get("ask")) or 0.0,
                    iv=_safe_float(greeks.get("mid_iv")),
                    delta=_safe_float(greeks.get("delta")),
                    open_interest=_safe_int(row.get("open_interest")),
                    volume=_safe_int(row.get("volume")),
                ))

        return OptionChain(
            ticker=ticker, spot=spot, asof=datetime.now(),
            calls=calls, source="tradier",
        )

    @staticmethod
    def _extract_expirations(payload: dict) -> list[str]:
        exp_node = payload.get("expirations")
        if not isinstance(exp_node, dict):
            return []
        date_node = exp_node.get("date")
        if isinstance(date_node, str):
            return [date_node]
        if isinstance(date_node, list):
            return [d for d in date_node if isinstance(d, str)]
        return []

    @staticmethod
    def _normalize_chain_options(payload: dict) -> list[dict]:
        opt_node = payload.get("options")
        if not isinstance(opt_node, dict):
            return []
        rows = opt_node.get("option")
        if isinstance(rows, dict):
            return [rows]
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
        return []

    def _fetch_chain_for_expiry(
        self, ticker: str, expiry: date,
        headers: dict[str, str], base_url: str,
    ) -> list[dict]:
        try:
            resp = requests.get(
                f"{base_url}/markets/options/chains",
                params={
                    "symbol": ticker,
                    "expiration": expiry.isoformat(),
                    "greeks": "true",
                },
                headers=headers,
                timeout=self._TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            payload = resp.json() or {}
        except Exception as e:
            logger.warning(
                "Tradier chain fetch failed for %s @ %s: %s",
                ticker, expiry, e,
            )
            return []
        return self._normalize_chain_options(payload)

    @staticmethod
    def _fetch_spot(
        ticker: str, headers: dict[str, str], base_url: str,
    ) -> float | None:
        try:
            resp = requests.get(
                f"{base_url}/markets/quotes",
                params={"symbols": ticker, "greeks": "false"},
                headers=headers,
                timeout=TradierChain._TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            payload = resp.json() or {}
            quotes = (payload.get("quotes") or {}).get("quote")
            if isinstance(quotes, list):
                quotes = quotes[0] if quotes else None
            if isinstance(quotes, dict):
                return _safe_float(quotes.get("last"))
        except Exception as e:
            logger.info("Tradier spot fetch failed for %s: %s", ticker, e)
        return None



def fetch_chains(
    tickers: list[str],
    *,
    dte_min: int,
    dte_max: int,
) -> dict[str, OptionChain]:
    """Per-ticker chain fetch with Tradier → yfinance fallback.

    Always returns a chain object for every input ticker. When all
    providers fail, the returned `OptionChain.source` is `"missing"`.
    """
    if not tickers:
        return {}
    tradier = TradierChain()
    yfin = YFinanceChain()
    out: dict[str, OptionChain] = {}
    for t in tickers:
        chain = tradier.fetch(t, dte_min, dte_max)
        if chain is None:
            chain = yfin.fetch(t, dte_min, dte_max)
        if chain is None:
            chain = OptionChain(
                ticker=t, spot=0.0, asof=datetime.now(),
                calls=[], source="missing",
            )
            logger.warning("chain unavailable for %s (all providers failed)", t)
        out[t] = chain
    return out
