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
from typing import Any, Literal, Protocol

import yfinance as yf

from ..config import Settings
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


def _snaptrade_client() -> Any:
    """Lazy SnapTrade client builder. Returns None when creds are missing
    so callers can degrade gracefully rather than blow up."""
    s = Settings()  # type: ignore[call-arg]
    if not all([
        s.snaptrade_client_id, s.snaptrade_consumer_key,
        s.snaptrade_user_id, s.snaptrade_user_secret,
    ]):
        logger.info(
            "SnapTrade chain provider unavailable: missing one or more of "
            "SNAPTRADE_CLIENT_ID/CONSUMER_KEY/USER_ID/USER_SECRET — "
            "falling back to yfinance only."
        )
        return None
    try:
        from snaptrade_client import SnapTrade
    except ImportError:
        logger.warning(
            "SnapTrade SDK not installed — falling back to yfinance only."
        )
        return None
    client = SnapTrade(
        client_id=s.snaptrade_client_id,
        consumer_key=s.snaptrade_consumer_key,
    )
    # Bind user creds to the convenience attrs the rest of the codebase uses.
    client.user_id = s.snaptrade_user_id
    client.user_secret = s.snaptrade_user_secret
    return client


def _first_account_id(client: Any) -> str | None:
    try:
        accts = client.account_information.list_user_accounts(
            user_id=client.user_id, user_secret=client.user_secret,
        ).body
    except Exception as e:
        logger.info("SnapTrade list_user_accounts failed: %s", e)
        return None
    if not accts:
        return None
    first = accts[0]
    return first.get("id") if isinstance(first, dict) else getattr(first, "id", None)


def _parse_snaptrade_options_payload(
    ticker: str, payload: dict[str, Any], dte_min: int, dte_max: int,
) -> OptionChain | None:
    """Translate SnapTrade's chain shape into OptionChain. Returns None
    when the shape is unrecognized (so we fall back to yfinance)."""
    try:
        spot = float(payload["underlying_price"])
        rows = payload["options"]
    except (KeyError, TypeError, ValueError):
        return None

    today = date.today()
    lo = today + timedelta(days=dte_min)
    hi = today + timedelta(days=dte_max)
    calls: list[OptionQuote] = []
    for r in rows:
        try:
            strike = float(r["strike_price"])
        except (KeyError, TypeError, ValueError):
            continue
        if strike <= spot:  # OTM calls only
            continue
        for entry in r.get("option_chain") or []:
            try:
                expiry = date.fromisoformat(entry["expiration_date"])
            except (KeyError, TypeError, ValueError):
                continue
            if expiry < lo or expiry > hi:
                continue
            call = entry.get("call") or {}
            calls.append(OptionQuote(
                strike=strike, expiry=expiry,
                bid=float(call.get("bid_price") or 0.0),
                ask=float(call.get("ask_price") or 0.0),
                iv=(float(call["implied_volatility"])
                    if call.get("implied_volatility") is not None else None),
                delta=(float(call["delta"]) if call.get("delta") is not None else None),
                open_interest=int(call.get("open_interest") or 0),
                volume=int(call.get("volume") or 0),
            ))

    return OptionChain(
        ticker=ticker, spot=spot, asof=datetime.now(),
        calls=calls, source="snaptrade",
    )


class SnapTradeChain:
    """SnapTrade-backed options chain provider.

    Returns None on any failure — auth missing, account list empty,
    endpoint not supported on the user's tier, payload shape mismatch.
    The orchestrator falls back to yfinance on None.
    """

    def fetch(
        self, ticker: str, dte_min: int, dte_max: int
    ) -> OptionChain | None:
        client = _snaptrade_client()
        if client is None:
            return None
        account_id = _first_account_id(client)
        if account_id is None:
            logger.info("SnapTrade: no account_id available for chain fetch")
            return None
        try:
            resp = client.trading.get_options_chain(
                account_id=account_id, symbol=ticker,
                user_id=client.user_id, user_secret=client.user_secret,
            )
        except Exception as e:
            logger.info("SnapTrade chain fetch failed for %s: %s", ticker, e)
            return None
        body = getattr(resp, "body", None)
        if not isinstance(body, dict):
            return None
        return _parse_snaptrade_options_payload(ticker, body, dte_min, dte_max)


def fetch_chains(
    tickers: list[str],
    *,
    dte_min: int,
    dte_max: int,
) -> dict[str, OptionChain]:
    """Per-ticker chain fetch with SnapTrade → yfinance fallback.

    Always returns a chain object for every input ticker. When both
    providers fail, the returned `OptionChain.source` is `"missing"` and
    `calls` is empty — the rebalancer prompt is told to show
    `UNAVAILABLE` for these tickers, and Opus will simply not recommend
    a WRITE_CALL on them.
    """
    if not tickers:
        return {}
    snap = SnapTradeChain()
    yfin = YFinanceChain()
    out: dict[str, OptionChain] = {}
    for t in tickers:
        chain = snap.fetch(t, dte_min, dte_max)
        if chain is None:
            chain = yfin.fetch(t, dte_min, dte_max)
        if chain is None:
            chain = OptionChain(
                ticker=t, spot=0.0, asof=datetime.now(),
                calls=[], source="missing",
            )
            logger.warning("chain unavailable for %s (both providers failed)", t)
        out[t] = chain
    return out
