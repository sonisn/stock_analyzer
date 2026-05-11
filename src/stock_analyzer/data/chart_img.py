"""chart-img.com v2 client — fetches TradingView-style PNG charts per ticker.

Used to embed a daily chart for each holding in the portfolio digest email.
Charts are rendered with a dark theme, 1D interval, RSI(14) and 50/200 SMA.

chart-img requires symbols in TradingView's `EXCHANGE:SYMBOL` form (e.g.,
`NASDAQ:AAPL`). We resolve the exchange via yfinance and persist the mapping
to disk so repeated runs are instant.

Requests are serialized with a 2-second pacing gap — that is the free-tier
limit (parallel fetches were getting 429-throttled).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from ..cache import CACHE_DIR
from ..logging import get_logger

logger = get_logger(__name__)

_ENDPOINT = "https://api.chart-img.com/v2/tradingview/advanced-chart"
_TIMEOUT_SECONDS = 20
_REQUEST_INTERVAL_SECONDS = 2.0

_EXCHANGE_CACHE_PATH = CACHE_DIR / "chart_img_exchanges.json"

# yfinance exchange codes → TradingView exchange prefixes accepted by chart-img.
_YF_EXCHANGE_TO_TV = {
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NAS": "NASDAQ",
    "NSM": "NASDAQ",
    "NYQ": "NYSE",
    "ASE": "AMEX",
    "PCX": "AMEX",
    "BATS": "CBOE",
}

_SKIP_QUOTE_TYPES = {"MUTUALFUND", "MONEYMARKET"}


def _load_exchange_cache() -> dict[str, str]:
    if not _EXCHANGE_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_EXCHANGE_CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_exchange_cache(cache: dict[str, str]) -> None:
    _EXCHANGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _EXCHANGE_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _resolve_symbol(ticker: str, cache: dict[str, str]) -> str | None:
    """Return `EXCHANGE:SYMBOL` for `ticker`, or None if unresolvable
    (e.g., money-market mutual funds with no listed exchange).

    Results are written into `cache` (an empty string sentinel means
    "known-unresolvable" so we don't re-query yfinance on every run)."""
    if ticker in cache:
        return cache[ticker] or None

    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
    except Exception as e:
        logger.warning("yfinance lookup failed for %s: %s", ticker, e)
        return None

    quote_type = (info.get("quoteType") or "").upper()
    if quote_type in _SKIP_QUOTE_TYPES:
        cache[ticker] = ""
        return None

    yf_exchange = info.get("exchange") or ""
    tv_exchange = _YF_EXCHANGE_TO_TV.get(yf_exchange)
    if not tv_exchange:
        logger.warning(
            "Unknown yfinance exchange %r for %s; falling back to NASDAQ",
            yf_exchange,
            ticker,
        )
        tv_exchange = "NASDAQ"

    resolved = f"{tv_exchange}:{ticker}"
    cache[ticker] = resolved
    return resolved


def _build_payload(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "interval": "1D",
        "theme": "dark",
        "timezone": "America/New_York",
        "studies": [
            {"name": "Volume", "forceOverlay": False},
            {"name": "Moving Average", "input": {
                "length": 50,
                "source": "close",
                "offset": 0,
                "smoothingLine": "SMA",
                "smoothingLength": 50
            }},
            {"name": "Moving Average", "input": {
                "length": 200,
                "source": "close",
                "offset": 0,
                "smoothingLine": "SMA",
                "smoothingLength": 200
            }},
        ],
    }


def fetch_chart(symbol: str, *, api_key: str | None = None) -> bytes | None:
    """Fetch a single chart PNG by `EXCHANGE:SYMBOL`. Returns None on any
    failure (never raises). Callers should resolve plain tickers first."""
    key = api_key or os.getenv("CHART_IMG_API_KEY")
    if not key:
        logger.warning("CHART_IMG_API_KEY not set; skipping chart for %s", symbol)
        return None

    body = json.dumps(_build_payload(symbol)).encode("utf-8")
    req = urllib.request.Request(
        _ENDPOINT,
        data=body,
        method="POST",
        headers={
            "x-api-key": key,
            "content-type": "application/json",
            "accept": "image/png",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:200]
        logger.warning("chart-img HTTP %d for %s: %s", e.code, symbol, detail)
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning("chart-img request failed for %s: %s", symbol, e)
    return None


def fetch_charts(tickers: list[str]) -> dict[str, bytes]:
    """Fetch charts for many tickers sequentially with a 2-second pacing gap.
    Returned dict is keyed by the raw ticker (not the resolved EXCHANGE:SYMBOL),
    so callers can match it to their existing per-ticker rendering."""
    if not tickers:
        return {}
    key = os.getenv("CHART_IMG_API_KEY")
    if not key:
        logger.warning("CHART_IMG_API_KEY not set; skipping all charts")
        return {}

    exchange_cache = _load_exchange_cache()
    out: dict[str, bytes] = {}
    issued = 0
    try:
        for ticker in tickers:
            symbol = _resolve_symbol(ticker, exchange_cache)
            if not symbol:
                logger.info("Skipping chart for %s — no listed exchange", ticker)
                continue
            if issued > 0:
                time.sleep(_REQUEST_INTERVAL_SECONDS)
            issued += 1
            data = fetch_chart(symbol, api_key=key)
            if data:
                out[ticker] = data
    finally:
        _save_exchange_cache(exchange_cache)
    logger.info("Fetched %d/%d charts from chart-img", len(out), len(tickers))
    return out
