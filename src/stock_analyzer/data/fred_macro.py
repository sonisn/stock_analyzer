"""FRED macro data — US economic indicators for regime classification.

Free key from https://fred.stlouisfed.org/docs/api/api_key.html — no quota
in practice. We pull six load-bearing series and synthesize a one-paragraph
regime summary that gets prepended to the Opus ranker's prompt so it can
reason about cyclicals vs defensives, rate regime, etc.
"""
from __future__ import annotations

from typing import Any

import requests

from ..logging import get_logger

logger = get_logger(__name__)

_BASE = "https://api.stlouisfed.org/fred/series/observations"

# series_name (our label) → FRED series ID
SERIES: dict[str, str] = {
    "yield_spread_10y_2y": "T10Y2Y",  # negative = recession warning
    "treasury_10y": "DGS10",
    "vix": "VIXCLS",
    "unemployment": "UNRATE",
    "industrial_production": "INDPRO",  # cycle proxy since FRED retired NAPM
    "fed_funds": "DFF",
}


def _fetch_series(
    series_id: str, api_key: str, limit: int = 24
) -> list[dict[str, Any]]:
    try:
        resp = requests.get(
            _BASE,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("observations", [])
    except Exception as e:
        logger.warning("FRED fetch failed for %s: %s", series_id, e)
        return []


def _latest_value(observations: list[dict[str, Any]]) -> float | None:
    for obs in observations:
        v = obs.get("value", ".")
        if v not in (".", "", None):
            try:
                return float(v)
            except (ValueError, TypeError):
                continue
    return None


def _trend_yoy(observations: list[dict[str, Any]]) -> float | None:
    """Compute YoY % change from the two most-recent same-month observations.
    Assumes obs are sorted descending (most-recent first)."""
    if len(observations) < 13:
        return None
    latest = _latest_value(observations[:1])
    year_ago = _latest_value(observations[12:13])
    if latest is None or year_ago is None or year_ago == 0:
        return None
    return (latest / year_ago - 1) * 100


def fetch_regime_data(api_key: str | None) -> dict[str, Any]:
    if not api_key:
        logger.info("FRED_API_KEY not set; macro regime feed disabled")
        return {}
    data: dict[str, Any] = {"as_of": None}
    indpro_obs: list[dict[str, Any]] = []
    for label, series_id in SERIES.items():
        obs = _fetch_series(series_id, api_key)
        data[label] = _latest_value(obs)
        if series_id == "INDPRO":
            indpro_obs = obs
        if obs and not data["as_of"]:
            data["as_of"] = obs[0].get("date")
    data["industrial_production_yoy"] = _trend_yoy(indpro_obs)
    return data


def regime_summary_text(data: dict[str, Any]) -> str:
    """One-paragraph summary for the ranker prompt. Plain text, no markdown."""
    if not data:
        return "Macro regime: data unavailable (FRED_API_KEY not configured)."

    parts: list[str] = []

    yc = data.get("yield_spread_10y_2y")
    if yc is not None:
        if yc < -0.10:
            parts.append(
                f"10Y-2Y yield curve INVERTED at {yc:+.2f}% — historical recession "
                "lead-indicator; favor balance sheets and defensives"
            )
        elif yc < 0.50:
            parts.append(
                f"yield curve flat at {yc:+.2f}% — late-cycle conditions, "
                "cyclicals at risk"
            )
        else:
            parts.append(
                f"yield curve positive at {yc:+.2f}% — expansion-typical, "
                "cyclicals viable"
            )

    vix = data.get("vix")
    if vix is not None:
        if vix > 30:
            parts.append(f"VIX elevated at {vix:.1f} (risk-off)")
        elif vix > 20:
            parts.append(f"VIX moderate at {vix:.1f}")
        else:
            parts.append(f"VIX low at {vix:.1f} (complacency / risk-on)")

    ur = data.get("unemployment")
    if ur is not None:
        parts.append(f"unemployment {ur:.1f}%")

    ff = data.get("fed_funds")
    t10 = data.get("treasury_10y")
    if ff is not None and t10 is not None:
        parts.append(f"Fed funds {ff:.2f}%, 10Y Treasury {t10:.2f}%")
    elif ff is not None:
        parts.append(f"Fed funds {ff:.2f}%")

    ip_yoy = data.get("industrial_production_yoy")
    if ip_yoy is not None:
        parts.append(f"industrial production YoY {ip_yoy:+.1f}%")

    if not parts:
        return "Macro regime: data unavailable."
    as_of = data.get("as_of") or "recent"
    return f"US macro regime (as of {as_of}): " + "; ".join(parts) + "."
