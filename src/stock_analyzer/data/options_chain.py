"""Options chain fetching: Tradier primary, yfinance fallback.

The orchestrator (`fetch_chains`) tries Tradier per-ticker and falls
back to yfinance on None/error. Both providers return a normalized
`OptionChain` containing only OTM options within the requested DTE band:
calls above spot (covered calls), puts below it (cash-secured puts), or
both, per the `kind` argument.

Failure of either provider for a given ticker is non-fatal — the
returned `OptionChain.source` is set to `"missing"` and the rebalancer
context just reads `Option chain: UNAVAILABLE` for that ticker.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Literal, Protocol

from ..config import Settings
from ..http_client import HttpClient, RetryPolicy
from ..logging import get_logger
from ..models.market import OptionChain, OptionQuote
from . import yf_gateway

logger = get_logger(__name__)

# Re-export the model classes here so legacy import paths continue
# working during Phase 1. Group C strips this shim once every callsite
# has been migrated.
__all__ = [
    "OptionChain",
    "OptionQuote",
    "OptionChainProvider",
    "YFinanceChain",
    "TradierChain",
    "ChainKind",
    "fetch_chains",
]

ChainKind = Literal["calls", "puts", "both"]


def _wants(kind: ChainKind) -> tuple[bool, bool]:
    """(want_calls, want_puts) for a `kind`."""
    return kind in ("calls", "both"), kind in ("puts", "both")


def _is_otm(option_type: str, strike: float, spot: float) -> bool:
    """OTM test; with an unknown spot (<= 0) every strike is kept."""
    if spot <= 0:
        return True
    return strike > spot if option_type == "call" else strike < spot


def _safe_float(v: object) -> float | None:
    """Coerce to float, returning None for None / NaN / Inf / unparseable.
    Used at provider boundary because yfinance returns NaN for low-volume
    strikes, which crashes downstream arithmetic and int() conversion."""
    if v is None:
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except TypeError, ValueError:
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
      - filter to OTM options only (calls: strike > spot, puts: strike < spot)
      - populate only the side(s) `kind` asks for
      - filter to expiries within [today+dte_min, today+dte_max]
      - return None on any error (graceful degradation)
    """

    def fetch(
        self, ticker: str, dte_min: int, dte_max: int, kind: ChainKind = "calls"
    ) -> OptionChain | None: ...


class YFinanceChain:
    """yfinance-backed options chain provider.

    yfinance does not expose Greeks; `delta` is always None. The
    rebalancer's prompt is robust to that — it falls back to comparing
    strike vs spot when delta is missing.
    """

    def fetch(
        self, ticker: str, dte_min: int, dte_max: int, kind: ChainKind = "calls"
    ) -> OptionChain | None:
        want_calls, want_puts = _wants(kind)
        # Every Yahoo touch below goes through the gateway, which builds
        # (and caches) the Ticker inside its own retry/pacing envelope.
        spot = _safe_float(
            yf_gateway.ticker_call(ticker, "chain.spot", lambda tk: tk.fast_info.last_price)
        )
        if spot is None or spot <= 0:
            logger.info(
                "yfinance returned invalid spot for %s (NaN / 0 / negative); skipping ticker",
                ticker,
            )
            return None

        today = date.today()
        lo = today + timedelta(days=dte_min)
        hi = today + timedelta(days=dte_max)
        calls: list[OptionQuote] = []
        puts: list[OptionQuote] = []
        expiries = yf_gateway.ticker_call(
            ticker, "chain.expiries", lambda tk: tuple(tk.options), default=()
        )
        if not expiries:
            logger.info("yfinance has no expiries for %s", ticker)
            return OptionChain(
                ticker=ticker,
                spot=spot,
                asof=datetime.now(),
                calls=[],
                source="yfinance",
            )

        for e_str in expiries:
            try:
                expiry = date.fromisoformat(e_str)
            except ValueError:
                continue
            if expiry < lo or expiry > hi:
                continue
            sides = [
                (side, out)
                for side, out, want in (("call", calls, want_calls), ("put", puts, want_puts))
                if want
            ]
            for option_type, out in sides:
                df = yf_gateway.ticker_call(
                    ticker,
                    f"chain.{e_str}.{option_type}s",
                    lambda tk, expiry_str=e_str, attr=f"{option_type}s": getattr(
                        tk.option_chain(expiry_str), attr
                    ),
                )
                if df is None:
                    continue
                for _, row in df.iterrows():
                    strike = _safe_float(row.get("strike"))
                    # NaN / None / 0 / negative strikes are nonsense; skip them.
                    if strike is None or strike <= 0:
                        continue
                    if not _is_otm(option_type, strike, spot):
                        continue
                    out.append(
                        OptionQuote(
                            strike=strike,
                            expiry=expiry,
                            bid=_safe_float(row.get("bid")) or 0.0,
                            ask=_safe_float(row.get("ask")) or 0.0,
                            iv=_safe_float(row.get("impliedVolatility")),
                            delta=None,  # yfinance does not provide Greeks
                            open_interest=_safe_int(row.get("openInterest")),
                            volume=_safe_int(row.get("volume")),
                        )
                    )

        return OptionChain(
            ticker=ticker,
            spot=spot,
            asof=datetime.now(),
            calls=calls,
            puts=puts,
            source="yfinance",
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
    # Tradier's documented sandbox/production limit is 120 req/min; stay
    # under it so a wide eligible-ticker set doesn't trip a 429 mid-run.
    _RATE_LIMIT_PER_MIN = 100

    def __init__(self) -> None:
        # Cache "is provider configured" once per instance so we don't
        # spam logs across multiple ticker fetches.
        self._configured: bool | None = None

    def _client(self, base_url: str, api_key: str) -> HttpClient:
        """A shared-client instance for this provider.

        Tradier used to call `requests.get` directly, which (a) depended on
        a package this project never declared — it only resolved
        transitively via yfinance — and (b) skipped the retry/backoff and
        rate limiting every other HTTP integration here gets. A transient
        429 therefore degraded silently to delayed yfinance data with no
        Greeks, which is exactly the case the delta-band strike rule needs
        real data for.
        """
        return HttpClient(
            base_url=base_url.rstrip("/"),
            default_headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            timeout=self._TIMEOUT_SECONDS,
            rate_limit_per_min=self._RATE_LIMIT_PER_MIN,
            retry_policy=RetryPolicy(max_attempts=3),
            name="tradier",
        )

    def fetch(
        self, ticker: str, dte_min: int, dte_max: int, kind: ChainKind = "calls"
    ) -> OptionChain | None:
        want_calls, want_puts = _wants(kind)
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

        with self._client(s.tradier_base_url, s.tradier_api_key) as client:
            # Step 1: expirations
            try:
                payload = (
                    client.get_json(
                        "/markets/options/expirations",
                        params={"symbol": ticker, "includeAllRoots": "true"},
                    )
                    or {}
                )
            except Exception as e:
                logger.warning("Tradier expirations fetch failed for %s: %s", ticker, e)
                return None

            expirations = self._extract_expirations(payload)
            if not expirations:
                logger.info("Tradier returned no expirations for %s", ticker)
                return OptionChain(
                    ticker=ticker,
                    spot=0.0,
                    asof=datetime.now(),
                    calls=[],
                    source="tradier",
                )

            today = date.today()
            lo = today + timedelta(days=dte_min)
            hi = today + timedelta(days=dte_max)
            in_band: list[date] = []
            for d_str in expirations:
                try:
                    d = date.fromisoformat(d_str)
                except ValueError, TypeError:
                    continue
                if lo <= d <= hi:
                    in_band.append(d)

            if not in_band:
                return OptionChain(
                    ticker=ticker,
                    spot=0.0,
                    asof=datetime.now(),
                    calls=[],
                    source="tradier",
                )

            # Step 2: fetch spot for filtering ITM strikes
            spot = self._fetch_spot(ticker, client) or 0.0

            calls: list[OptionQuote] = []
            puts: list[OptionQuote] = []
            for expiry in in_band:
                chain_rows = self._fetch_chain_for_expiry(ticker, expiry, client)
                for row in chain_rows:
                    option_type = row.get("option_type")
                    if option_type == "call" and want_calls:
                        out = calls
                    elif option_type == "put" and want_puts:
                        out = puts
                    else:
                        continue
                    strike = _safe_float(row.get("strike")) or 0.0
                    if strike <= 0 or not _is_otm(option_type, strike, spot):
                        continue  # OTM only when spot known; else keep all
                    greeks = row.get("greeks") or {}
                    out.append(
                        OptionQuote(
                            strike=strike,
                            expiry=expiry,
                            bid=_safe_float(row.get("bid")) or 0.0,
                            ask=_safe_float(row.get("ask")) or 0.0,
                            iv=_safe_float(greeks.get("mid_iv")),
                            delta=_safe_float(greeks.get("delta")),
                            open_interest=_safe_int(row.get("open_interest")),
                            volume=_safe_int(row.get("volume")),
                        )
                    )

            return OptionChain(
                ticker=ticker,
                spot=spot,
                asof=datetime.now(),
                calls=calls,
                puts=puts,
                source="tradier",
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
        self,
        ticker: str,
        expiry: date,
        client: HttpClient,
    ) -> list[dict]:
        try:
            payload = (
                client.get_json(
                    "/markets/options/chains",
                    params={
                        "symbol": ticker,
                        "expiration": expiry.isoformat(),
                        "greeks": "true",
                    },
                )
                or {}
            )
        except Exception as e:
            logger.warning(
                "Tradier chain fetch failed for %s @ %s: %s",
                ticker,
                expiry,
                e,
            )
            return []
        return self._normalize_chain_options(payload)

    @staticmethod
    def _fetch_spot(ticker: str, client: HttpClient) -> float | None:
        try:
            payload = (
                client.get_json(
                    "/markets/quotes",
                    params={"symbols": ticker, "greeks": "false"},
                )
                or {}
            )
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
    kind: ChainKind = "calls",
) -> dict[str, OptionChain]:
    """Per-ticker chain fetch with Tradier → yfinance fallback.

    `kind` picks the side(s) populated: "calls" (the covered-call default),
    "puts" (cash-secured puts) or "both".

    Always returns a chain object for every input ticker. When all
    providers fail, the returned `OptionChain.source` is `"missing"`.
    """
    if not tickers:
        return {}
    tradier = TradierChain()
    yfin = YFinanceChain()
    out: dict[str, OptionChain] = {}
    for t in tickers:
        chain = tradier.fetch(t, dte_min, dte_max, kind)
        if chain is None:
            chain = yfin.fetch(t, dte_min, dte_max, kind)
        if chain is None:
            chain = OptionChain(
                ticker=t,
                spot=0.0,
                asof=datetime.now(),
                calls=[],
                source="missing",
            )
            logger.warning("chain unavailable for %s (all providers failed)", t)
        out[t] = chain
    return out
