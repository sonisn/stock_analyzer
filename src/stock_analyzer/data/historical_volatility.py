"""Realized volatility computation from yfinance close prices.

Two consumers:
  1. Covered-call writing — a free proxy for IV-rank when no paid
     IV-rank provider (ORATS, tastytrade, IBKR) is wired up. The IV/HV
     ratio captures the same "is implied vol elevated?" signal that IVR
     does, just in a different space.
  2. `discover/output_validation.py` — sanity-checks a Ranker pick's
     stated bull-scenario return against the ticker's own realized
     volatility, scaled to the pick's horizon.

This is a synchronous wrapper around yfinance — one network call per
ticker. Batched callers should expect ~1 sec per ticker.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from ..logging import get_logger
from ..models.market import RealizedVolatility

logger = get_logger(__name__)

LOOKBACK_DAYS = 252  # one trading year
ANNUALIZATION_FACTOR = math.sqrt(252)

# Re-export RealizedVolatility from the canonical models package so
# legacy import paths continue working during Phase 1. Group C removes
# this shim.
__all__ = [
    "RealizedVolatility",
    "_annualized_vol_from_closes",
    "fetch_realized_volatility",
]


def _annualized_vol_from_closes(closes: list[float]) -> float | None:
    """Compute annualized stdev of log returns. Returns None if too
    few observations or all closes equal (zero variance)."""
    if len(closes) < 30:
        return None
    # log returns
    rets: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev <= 0 or cur <= 0:
            continue
        rets.append(math.log(cur / prev))
    if len(rets) < 20:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    if var <= 0:
        return None
    sd = math.sqrt(var)
    return sd * ANNUALIZATION_FACTOR


def fetch_realized_volatility(
    tickers: list[str],
    *,
    lookback_days: int = LOOKBACK_DAYS,
) -> dict[str, RealizedVolatility]:
    """Per-ticker realized vol via yfinance. Returns {} for tickers
    that fail (network, no data, too few observations). Never raises.
    """
    if not tickers:
        return {}

    from . import yf_gateway

    # `lookback_days` counts trading days (it was Yahoo's `period="300d"`,
    # which is 300 bars); a calendar year holds ~252 of them.
    trading_days = max(lookback_days + 30, 300)
    start = date.today() - timedelta(days=trading_days * 365 // 252 + 10)

    def _bars(t: str):
        return yf_gateway.daily_bars(t, start=start, what="historical_volatility")

    out: dict[str, RealizedVolatility] = {}
    # Fetched on the gateway's bounded pool (usually memory hits: the
    # technicals step already downloaded these), then computed in order.
    frames = dict(yf_gateway.map_symbols(_bars, tickers))
    for t in tickers:
        df = frames.get(t)
        if df is None or df.empty:
            logger.info("HV fetch returned no data for %s", t)
            continue
        try:
            closes = [float(c) for c in df["Close"].tolist() if not math.isnan(c)]
        except Exception as e:
            logger.info("HV closes parse failed for %s: %s", t, e)
            continue
        if len(closes) < 30:
            logger.info("HV: too few closes for %s (%d) — skipping", t, len(closes))
            continue
        hv = _annualized_vol_from_closes(closes[-lookback_days - 1 :])
        if hv is None or hv <= 0:
            continue
        out[t] = RealizedVolatility(
            ticker=t,
            hv_annualized=hv,
            sample_size=len(closes),
        )

    logger.info(
        "HV fetch: %d/%d ticker(s) with valid realized-vol estimate. Values: %s",
        len(out),
        len(tickers),
        {t: round(r.hv_annualized * 100, 1) for t, r in out.items()},
    )
    return out
