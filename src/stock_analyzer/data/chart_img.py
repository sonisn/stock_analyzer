"""chart-img.com v2 client — fetches TradingView-style PNG charts per ticker.

Used to embed a daily chart for each holding in the portfolio digest email.
Charts are rendered with a dark theme, 1D interval, RSI(14) and 50/200 SMA.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ..logging import get_logger

logger = get_logger(__name__)

_ENDPOINT = "https://api.chart-img.com/v2/tradingview/advanced-chart"
_TIMEOUT_SECONDS = 20
_MAX_WORKERS = 8


def _build_payload(ticker: str) -> dict:
    return {
        "symbol": ticker,
        "interval": "1D",
        "theme": "dark",
        "studies": [
            {"name": "Relative Strength Index", "input": {"in_0": 14}},
            {"name": "Moving Average", "input": {"in_0": 50}},
            {"name": "Moving Average", "input": {"in_0": 200}},
        ],
    }


def fetch_chart(ticker: str, *, api_key: str | None = None) -> bytes | None:
    """Fetch a single chart PNG. Returns None on any failure (never raises)."""
    key = api_key or os.getenv("CHART_IMG_API_KEY")
    if not key:
        logger.warning("CHART_IMG_API_KEY not set; skipping chart for %s", ticker)
        return None

    body = json.dumps(_build_payload(ticker)).encode("utf-8")
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
        logger.warning(
            "chart-img HTTP %d for %s: %s", e.code, ticker, detail
        )
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning("chart-img request failed for %s: %s", ticker, e)
    return None


def fetch_charts(tickers: list[str]) -> dict[str, bytes]:
    """Fetch charts for many tickers in parallel. Tickers with no chart are omitted."""
    if not tickers:
        return {}
    key = os.getenv("CHART_IMG_API_KEY")
    if not key:
        logger.warning("CHART_IMG_API_KEY not set; skipping all charts")
        return {}

    out: dict[str, bytes] = {}
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(tickers))) as ex:
        futures = {ex.submit(fetch_chart, t, api_key=key): t for t in tickers}
        for fut in futures:
            ticker = futures[fut]
            data = fut.result()
            if data:
                out[ticker] = data
    logger.info("Fetched %d/%d charts from chart-img", len(out), len(tickers))
    return out
