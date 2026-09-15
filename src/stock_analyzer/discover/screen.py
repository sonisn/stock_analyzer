"""Screen module — hard filters + quant score.

Pure functions over fundamentals + technicals + universe entries. No I/O,
easy to unit-test.

THE FILTER IS A STYLE BET, AND IT IS DELIBERATE. Four of the eight hard
rejections are trend rules (above 200DMA, 50DMA above 200DMA, positive
6-month RS, within 30% of the 52-week high), so the screen only ever
considers momentum/quality names in a Stage-2 uptrend. That excludes every
mean-reversion and deep-value setup by construction. Momentum is a
defensible factor, so this is a choice rather than a bug — but it is a
choice, it skews the book high-beta (which is why the track record reports
beta-adjusted alpha), and survivor counts are logged per run so a collapsed
funnel is visible instead of silently producing "the top 5 of 6".

Score budget — max 100 points, plus whatever the caller adds as a theme
bonus:
  45 pts fundamentals  (growth, FCF yield, margins, debt)
  45 pts trend         (RS, entry zone, volume, weekly RSI, EPS revisions)
  10 pts conviction    (media mention count, source diversity)

CONVICTION IS WEIGHTED LIGHTLY ON PURPOSE. It is a count of how many news
articles mentioned the ticker, i.e. a media-attention measure, and attention
is associated with crowding rather than with forward excess return — the
empirical prior is that its information coefficient is near zero or
negative. It used to carry 25 of ~105 points (a quarter of the score) and
watchlist membership added a flat +5 conviction, which handed every
user-supplied name a ~10-point structural head start and turned the
ranking into partial confirmation of the user's prior. Watchlist membership
is now an INCLUSION rule in `discover/universe.py` (the name always gets
analyzed) and carries no score.

Before re-tuning any weight here, measure it: `uv run validate-screen`
reports the information coefficient of every sub-component below against
realized forward alpha, using the scores already stored per run.
"""

from __future__ import annotations

from typing import Any

# Hard filter thresholds — tighten/loosen after running the pipeline a few
# times. These are deliberately conservative for capital-at-stake decisions.
MIN_MARKET_CAP = 2e9
MIN_REVENUE_GROWTH = 0.08
MAX_DEBT_TO_EQUITY = 2.0
MAX_DRAWDOWN_FROM_52W_HIGH = -0.30


def passes_trend_gate(technicals: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """The four trend rules of the hard filter, from price history alone.

    Split out so the pipeline can apply it BEFORE the expensive fetches.
    Fundamentals cost two Yahoo requests per name and EPS revisions a
    third, but a name that isn't in a Stage-2 uptrend can never pass
    `passes_hard_filter` no matter what those return — so screening a
    500-name frame used to spend roughly 1,500 requests establishing
    facts about names already eliminated by their charts. Gate first,
    then fetch: same survivors, a third of the traffic, far less exposure
    to Yahoo's throttling.
    """
    if not technicals:
        return False, ["no technicals data"]
    t = technicals
    reasons: list[str] = []

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

    # Trend rules live in passes_trend_gate so the pre-fetch gate and the
    # real filter can never drift apart.
    _, trend_reasons = passes_trend_gate(t)
    reasons.extend(trend_reasons)

    return (len(reasons) == 0, reasons)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _score_fundamentals(f: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """0-45 pts. Growth + cash generation + margins + debt health.

    Picked up the points that came off `conviction`: these are measured
    company properties rather than a proxy for press attention.
    """
    parts: dict[str, float] = {}

    rg = f.get("revenue_growth_yoy") or 0
    parts["revenue_growth"] = _clamp((rg - 0.08) / (0.25 - 0.08) * 17, 0, 17)

    fcfy = f.get("fcf_yield") or 0
    parts["fcf_yield"] = _clamp(fcfy / 0.06 * 11, 0, 11)

    om = f.get("operating_margin") or 0
    parts["operating_margin"] = _clamp(om / 0.30 * 11, 0, 11)

    de = f.get("debt_to_equity")
    if de is None:
        parts["debt_health"] = 3.0
    elif de >= 2.0:
        parts["debt_health"] = 0
    elif de <= 0.5:
        parts["debt_health"] = 6
    else:
        parts["debt_health"] = 6 * (2.0 - de) / 1.5

    return (sum(parts.values()), parts)


def _score_trend(
    t: dict[str, Any],
    revisions: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    """0-45 pts (can dip to -3 on a lowering revision). RS leadership +
    entry zone + volume + non-stretched momentum + EPS revision flow."""
    parts: dict[str, float] = {}

    rs6 = t.get("rs_6mo") or 0
    parts["rs_6mo"] = _clamp(9 + rs6 * 78, 9, 17) if rs6 > 0 else 0

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
        parts["eps_revisions"] = 8.0
    elif direction == "lowering":
        parts["eps_revisions"] = -3.0
    else:
        parts["eps_revisions"] = 0.0

    return (sum(parts.values()), parts)


# Source labels that describe WHERE a name came from rather than evidence
# about it. Index membership, the user's own watchlist and an existing
# holding are eligibility facts, so they must not inflate source diversity —
# otherwise every watchlist name scores higher than an identical non-watchlist
# one and the ranking confirms the user's prior instead of testing it.
_NON_EVIDENCE_SOURCES = frozenset({"index", "watchlist", "holding"})


def _score_conviction(u: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """0-10 pts. Media attention, weighted lightly and on purpose.

    Down from 25 of ~105 points. `conviction` counts news-article mentions,
    which measures attention and crowding rather than forward return, and
    only the two genuinely external feeds (insider coverage, hedge-fund
    coverage) count toward source diversity. Run `validate-screen` to see
    this component's measured information coefficient before giving it any
    more weight back.
    """
    parts: dict[str, float] = {}
    parts["mentions"] = _clamp(u.get("conviction", 0) * 0.8, 0, 6)
    evidence_sources = {s for s in u.get("sources", []) if s not in _NON_EVIDENCE_SOURCES}
    parts["source_diversity"] = {0: 0.0, 1: 2.0}.get(len(evidence_sources), 4.0)
    return (sum(parts.values()), parts)


def score_candidate(
    fundamentals: dict[str, Any],
    technicals: dict[str, Any],
    universe_entry: dict[str, Any],
    revisions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine the three scoring dimensions into a 0-100 total + breakdown.

    `revisions` is the per-ticker EPS-revisions summary (the LLM-stage
    payload's `eps_revisions` field). When present, the trend score picks
    up +8 / -3 based on direction_30d — analyst revision flow is one of the
    few forward-looking signals here, so it carries more weight than the
    media-mention count does. Optional so unit tests + legacy callers can
    still pass three args."""
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
