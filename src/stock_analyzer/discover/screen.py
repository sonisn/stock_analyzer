"""Screen module — hard filters + 0-100 quant score.

Pure functions over fundamentals + technicals + universe entries. No I/O,
easy to unit-test. Tune the thresholds here based on what the pipeline
surfaces over a few months.

Hard filters reject anything not in a Stage-2 uptrend with healthy
fundamentals. Score weights mid-long term priorities:
  40 pts fundamentals  (growth, FCF yield, margins, debt)
  35 pts trend         (RS, entry zone, volume, weekly RSI)
  25 pts conviction    (insider/billionaire mention count, source diversity)
"""
from __future__ import annotations

from typing import Any

# Hard filter thresholds — tighten/loosen after running the pipeline a few
# times. These are deliberately conservative for capital-at-stake decisions.
MIN_MARKET_CAP = 5e9
MIN_REVENUE_GROWTH = 0.08
MAX_DEBT_TO_EQUITY = 2.0
MAX_DRAWDOWN_FROM_52W_HIGH = -0.30


def passes_hard_filter(
    fundamentals: dict[str, Any] | None,
    technicals: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    """Return (passes, reasons_failed). Empty reasons list means it passed."""
    reasons: list[str] = []
    if not fundamentals:
        reasons.append("no fundamentals data")
    if not technicals:
        reasons.append("no technicals data")
    if reasons:
        return False, reasons

    assert fundamentals is not None and technicals is not None
    f, t = fundamentals, technicals

    mc = f.get("market_cap")
    if mc is None or mc < MIN_MARKET_CAP:
        reasons.append(f"market_cap={mc} < ${MIN_MARKET_CAP / 1e9:.0f}B")

    rg = f.get("revenue_growth_yoy")
    if rg is None or rg < MIN_REVENUE_GROWTH:
        reasons.append(f"revenue_growth={rg} < {MIN_REVENUE_GROWTH:.0%}")

    ocf = f.get("operating_cash_flow")
    if ocf is None or ocf <= 0:
        reasons.append(f"operating_cash_flow={ocf} not positive")

    de = f.get("debt_to_equity")
    if de is not None and de > MAX_DEBT_TO_EQUITY:
        reasons.append(f"debt_to_equity={de:.2f} > {MAX_DEBT_TO_EQUITY}")

    if not t.get("above_200dma"):
        reasons.append("price not above 200DMA")
    if not t.get("ma_alignment_50_200"):
        reasons.append("50DMA not above 200DMA")

    rs6 = t.get("rs_6mo")
    if rs6 is None or rs6 <= 0:
        reasons.append(f"rs_6mo={rs6} not positive")

    dist = t.get("dist_from_52w_high")
    if dist is None or dist < MAX_DRAWDOWN_FROM_52W_HIGH:
        reasons.append(f"52w drawdown {dist} > {abs(MAX_DRAWDOWN_FROM_52W_HIGH):.0%}")

    return (len(reasons) == 0, reasons)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _score_fundamentals(f: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """0-40 pts. Growth + cash generation + margins + debt health."""
    parts: dict[str, float] = {}

    rg = f.get("revenue_growth_yoy") or 0
    parts["revenue_growth"] = _clamp((rg - 0.08) / (0.25 - 0.08) * 15, 0, 15)

    fcfy = f.get("fcf_yield") or 0
    parts["fcf_yield"] = _clamp(fcfy / 0.06 * 10, 0, 10)

    om = f.get("operating_margin") or 0
    parts["operating_margin"] = _clamp(om / 0.30 * 10, 0, 10)

    de = f.get("debt_to_equity")
    if de is None:
        parts["debt_health"] = 2.5
    elif de >= 2.0:
        parts["debt_health"] = 0
    elif de <= 0.5:
        parts["debt_health"] = 5
    else:
        parts["debt_health"] = 5 * (2.0 - de) / 1.5

    return (sum(parts.values()), parts)


def _score_trend(
    t: dict[str, Any],
    revisions: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    """0-40 pts. RS leadership + entry zone + volume + non-stretched
    momentum + EPS revision flow."""
    parts: dict[str, float] = {}

    rs6 = t.get("rs_6mo") or 0
    parts["rs_6mo"] = _clamp(8 + rs6 * 70, 8, 15) if rs6 > 0 else 0

    dist = t.get("dist_from_52w_high")
    if dist is None:
        parts["entry_zone"] = 5
    else:
        # Triangular peak at -10% drawdown; 0 at +2% (extended) or -30% (broken).
        ideal = -0.10
        spread = 0.20
        parts["entry_zone"] = _clamp(10 * (1 - abs(dist - ideal) / spread), 0, 10)

    vt = t.get("volume_trend_20_60")
    parts["volume_trend"] = 5 if (vt is not None and vt > 0) else 0

    wr = t.get("weekly_rsi")
    if wr is None:
        parts["weekly_rsi"] = 2.5
    elif 40 <= wr <= 65:
        parts["weekly_rsi"] = 5
    elif wr < 80:
        parts["weekly_rsi"] = 2
    else:
        parts["weekly_rsi"] = 0

    # EPS revision flow — one of the strongest forward-thesis signals.
    # Net ups across current quarter + current year over the last 30 days
    # is summarized as direction_30d ('raising' / 'stable' / 'lowering').
    # Missing (no analyst coverage / fetch failed) → neutral 0.
    direction = (revisions or {}).get("direction_30d")
    if direction == "raising":
        parts["eps_revisions"] = 5.0
    elif direction == "lowering":
        parts["eps_revisions"] = -3.0
    else:
        parts["eps_revisions"] = 0.0

    return (sum(parts.values()), parts)


def _score_conviction(u: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """0-25 pts. Universe mention count + how many distinct sources flagged it."""
    parts: dict[str, float] = {}
    parts["mentions"] = _clamp(u.get("conviction", 0) * 1.5, 0, 15)
    n_sources = len(set(u.get("sources", [])))
    parts["source_diversity"] = {0: 0.0, 1: 3.0, 2: 7.0}.get(n_sources, 10.0)
    return (sum(parts.values()), parts)


def score_candidate(
    fundamentals: dict[str, Any],
    technicals: dict[str, Any],
    universe_entry: dict[str, Any],
    revisions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine the three scoring dimensions into a 0-105 total + breakdown.

    `revisions` is the per-ticker EPS-revisions summary (the LLM-stage
    payload's `eps_revisions` field). When present, the trend score
    picks up a +/-5 bonus based on direction_30d. Optional so unit tests
    + legacy callers can still pass three args."""
    fund_total, fund_parts = _score_fundamentals(fundamentals)
    trend_total, trend_parts = _score_trend(technicals, revisions=revisions)
    conv_total, conv_parts = _score_conviction(universe_entry)
    return {
        "score": round(fund_total + trend_total + conv_total, 1),
        "components": {
            "fundamentals": round(fund_total, 1),
            "trend": round(trend_total, 1),
            "conviction": round(conv_total, 1),
        },
        "breakdown": {
            "fundamentals": {k: round(v, 1) for k, v in fund_parts.items()},
            "trend": {k: round(v, 1) for k, v in trend_parts.items()},
            "conviction": {k: round(v, 1) for k, v in conv_parts.items()},
        },
    }
