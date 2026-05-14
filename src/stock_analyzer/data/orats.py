"""ORATS Data v2 — IV rank / IV percentile per ticker.

The free tier (5 req/min) is plenty because we batch every eligible
ticker into one call. Returns None gracefully on any failure — the CC
pipeline degrades to delta-only strike selection (no IVR timing signal),
not catastrophic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import requests

from ..config import Settings
from ..logging import get_logger

logger = get_logger(__name__)

_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class IVRank:
    """IV rank snapshot for one ticker."""
    ticker: str
    iv: float                     # current 30-day ATM IV (decimal)
    iv_rank_1y: float | None      # 0-100 over past 252 trading days
    iv_pct_1y: float | None       # 0-100 percentile, same window
    iv_rank_1m: float | None      # 1-month equivalents (secondary)
    iv_pct_1m: float | None


def _safe_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def fetch_iv_ranks(tickers: list[str]) -> dict[str, IVRank]:
    """Batch ORATS IVR fetch for a list of tickers.

    Returns {ticker: IVRank} for every ticker ORATS had data for.
    Tickers missing from the response (typo, no options listed, etc.)
    are simply absent from the output. On total failure (no API key,
    network error, unexpected payload shape) returns {}.
    """
    if not tickers:
        return {}

    s = Settings()  # type: ignore[call-arg]
    if not s.orats_api_key:
        logger.info(
            "ORATS IVR provider not configured (ORATS_API_KEY unset). "
            "CC pipeline will not include IV-rank timing signal."
        )
        return {}

    # ORATS accepts comma-delimited ticker list — one call covers all.
    params = {
        "token": s.orats_api_key,
        "ticker": ",".join(tickers),
    }
    try:
        resp = requests.get(
            f"{s.orats_base_url}/ivrank",
            params=params,
            timeout=_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        logger.warning(
            "ORATS IVR fetch failed for %d ticker(s): %s. "
            "CC pipeline will proceed without IV-rank signal.",
            len(tickers), e,
        )
        return {}

    # ORATS responses: either a top-level list, or {"data": [...]}.
    rows: list[dict[str, Any]]
    if isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = [r for r in payload["data"] if isinstance(r, dict)]
    else:
        logger.warning(
            "ORATS IVR returned unexpected payload shape (%s); "
            "treating as empty.", type(payload).__name__,
        )
        return {}

    out: dict[str, IVRank] = {}
    for row in rows:
        ticker = row.get("ticker")
        if not isinstance(ticker, str):
            continue
        iv = _safe_float(row.get("iv"))
        if iv is None:
            continue
        out[ticker] = IVRank(
            ticker=ticker,
            iv=iv,
            iv_rank_1y=_safe_float(row.get("ivRank1y")),
            iv_pct_1y=_safe_float(row.get("ivPct1y")),
            iv_rank_1m=_safe_float(row.get("ivRank1m")),
            iv_pct_1m=_safe_float(row.get("ivPct1m")),
        )

    if not out:
        logger.warning(
            "ORATS returned no usable rows for %d ticker(s) requested. "
            "Tickers: %s", len(tickers), tickers,
        )
    else:
        logger.info(
            "ORATS IVR: fetched %d/%d ticker(s). "
            "IVR-1y values: %s",
            len(out), len(tickers),
            {t: round(r.iv_rank_1y, 1) for t, r in out.items()
             if r.iv_rank_1y is not None},
        )
    return out
