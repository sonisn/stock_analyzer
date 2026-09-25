"""Add on weakness — holdings worth more money after a drop. No LLM.

For a long-term investor a lower price on an intact business is a better
entry, not a warning. A holding qualifies when it is at least
`ADD_ON_DRAWDOWN_PCT` below its 52-week high AND:
  - its thesis check (for former picks) isn't BROKEN or WATCH,
  - analysts are not cutting its EPS estimates,
  - its sector isn't already over the cap, and
  - it is under `MAX_WEIGHT_PCT` of the portfolio (room to add).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..data import yf_gateway

ADD_ON_DRAWDOWN_PCT = 15.0
MAX_WEIGHT_PCT = 20.0


def price_vs_high(tickers: list[str]) -> dict[str, tuple[float, float]]:
    """{ticker: (price, 52-week high)} from the memoized yfinance info."""
    out: dict[str, tuple[float, float]] = {}
    for t in tickers:
        info = yf_gateway.ticker_call(t, "ticker.info", lambda tk: tk.info or {}, default={}) or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        high = info.get("fiftyTwoWeekHigh")
        if price and high and high > 0:
            out[t] = (float(price), float(high))
    return out


def format_add_on_block(
    reviews: dict[str, Any],
    technicals: dict[str, dict[str, Any]],
    values: dict[str, float],
    *,
    drawdown_pct: float = ADD_ON_DRAWDOWN_PCT,
    max_weight_pct: float = MAX_WEIGHT_PCT,
) -> str:
    """Rebalancer prompt lines for held names that are HOLD with confidence
    >= 7 and 15%+ below their 52-week high, under the weight limit."""
    total = sum(values.values())
    lines = []
    for t, r in sorted(reviews.items()):
        verdict, conf = getattr(r, "verdict", None), getattr(r, "confidence", 0) or 0
        dist = (technicals.get(t) or {}).get("dist_from_52w_high")
        weight = values.get(t, 0.0) / total * 100 if total else 0.0
        dipped = dist is not None and dist * 100 <= -drawdown_pct
        if verdict == "HOLD" and conf >= 7 and dipped and weight < max_weight_pct:
            lines.append(
                f"  {t}: HOLD-{conf}, {dist * 100:+.0f}% vs 52-week high, {weight:.1f}% of holdings"
            )
    return "\n".join(lines)


def add_on_candidates(
    *,
    values: dict[str, float],
    highs: dict[str, tuple[float, float]],
    sector_of: dict[str, str],
    over_cap_sectors: set[str],
    thesis_flagged: set[str],
    estimates_cut: Callable[[list[str]], set[str]],
    drawdown_pct: float = ADD_ON_DRAWDOWN_PCT,
    max_weight_pct: float = MAX_WEIGHT_PCT,
) -> list[dict[str, Any]]:
    """Qualifying holdings, deepest drop first. `estimates_cut(tickers)`
    returns the subset whose EPS estimates are being lowered — called
    only for the few that pass the price test."""
    total = sum(values.values())
    dipped: list[dict[str, Any]] = []
    for t, (price, high) in highs.items():
        off = (price / high - 1) * 100
        weight = values.get(t, 0.0) / total * 100 if total else 0.0
        if (
            off <= -drawdown_pct
            and t not in thesis_flagged
            and sector_of.get(t) not in over_cap_sectors
            and weight < max_weight_pct
        ):
            dipped.append(
                {"ticker": t, "off_high_pct": off, "weight_pct": weight, "sector": sector_of.get(t)}
            )
    if not dipped:
        return []
    cut = estimates_cut([d["ticker"] for d in dipped])
    return sorted((d for d in dipped if d["ticker"] not in cut), key=lambda d: d["off_high_pct"])
