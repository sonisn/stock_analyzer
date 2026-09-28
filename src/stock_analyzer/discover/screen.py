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
  45 pts fundamentals  (growth, FCF yield, margins, debt), plus up to
                        +8 / -3 for contracted-book growth where tagged
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

from collections.abc import Iterable
from statistics import median
from typing import Any

# Hard filter thresholds — tighten/loosen after running the pipeline a few
# times. These are deliberately conservative for capital-at-stake decisions.
MIN_MARKET_CAP = 2e9
MIN_REVENUE_GROWTH = 0.08
MAX_DEBT_TO_EQUITY = 2.0
# The universe's quality rules (data/universe_scan.QUALITY_RULES), checked
# again here so a name that enters another way — an earnings standout, an
# insider cluster, a holding — meets the same bar: positive operating and
# free cash flow and a real return on equity. (Until 2026-09-27 a company
# burning cash could pass on 2+ years of runway and <= 20% dilution; the
# user chose self-funding businesses instead.) The 12-month revenue rule
# has no per-ticker counterpart in Yahoo's quote and stays universe-only.
MIN_RETURN_ON_EQUITY = 0.10
# Estimates, revisions and targets from one or two analysts are an opinion,
# not a consensus — and mid-caps are thinly covered.
MIN_ANALYST_COVERAGE = 3
# Mid-caps swing more, and the trend score rewards big moves: ranked on raw
# points they would crowd the analysis slots on volatility alone. Each
# candidate is ranked within its band instead (rank_within_size_bands).
LARGE_CAP = 10e9
# How many of one sector / one industry may take the paid analysis slots.
# On 2026-09-27 seven of the 25 were energy and four of those refiners
# (VLO, MPC, PSX, DINO) — one bet on refining margins bought four times.
# Pick-time sector caps (DISCOVER_MAX_SECTOR_PCT) only act after that
# analysis is paid for.
MAX_PER_SECTOR_SHORTLIST = 5
MAX_PER_INDUSTRY_SHORTLIST = 2
MAX_DRAWDOWN_FROM_52W_HIGH = -0.30
# The soft gate's only price rule: skip names in collapse (a falling knife).
SOFT_MAX_DRAWDOWN_FROM_52W_HIGH = -0.40
# Where the screen's entry-zone score peaks (10% below the 52-week high).
IDEAL_ENTRY_DRAWDOWN = -0.10

TrendGate = str  # "strict" | "soft" | "off"


def passes_trend_gate(
    technicals: dict[str, Any] | None, mode: TrendGate = "strict"
) -> tuple[bool, list[str]]:
    """The trend rules of the hard filter, from price history alone.

    `mode` (DISCOVER_TREND_GATE):
      - strict: the original four uptrend rules (above the 200-day, 50 over
        200, positive 6-month relative strength, within 30% of the high);
      - soft (default): only "not more than 40% below the 52-week high".
        For 3-5 year holds an uptrend requirement shut out quality names
        in a temporary dip, and on 15 years of S&P 500 prices the strict
        gate's picks did no better over the next year (+3.4% vs +3.6% vs
        SPY, t=0.6). Trend still counts in the score;
      - off: no price rule (data must still exist).

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

    if mode == "off":
        return True, []
    if mode == "soft":
        dist = t.get("dist_from_52w_high")
        if dist is not None and dist < SOFT_MAX_DRAWDOWN_FROM_52W_HIGH:
            reasons.append(f"52w drawdown {dist} > {abs(SOFT_MAX_DRAWDOWN_FROM_52W_HIGH):.0%}")
        return (len(reasons) == 0, reasons)

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
    trend_gate: TrendGate = "strict",
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
    fcf = f.get("free_cash_flow")
    if fcf is None or fcf <= 0:
        reasons.append(f"free_cash_flow={fcf} not positive")
    roe = f.get("return_on_equity")
    if roe is None or roe < MIN_RETURN_ON_EQUITY:
        reasons.append(f"return_on_equity={roe} < {MIN_RETURN_ON_EQUITY:.0%}")

    ac = f.get("analyst_count")
    if ac is None or ac < MIN_ANALYST_COVERAGE:
        reasons.append(f"analyst coverage {ac} < {MIN_ANALYST_COVERAGE}")

    de = f.get("debt_to_equity")
    if de is not None and de > MAX_DEBT_TO_EQUITY:
        reasons.append(f"debt_to_equity={de:.2f} > {MAX_DEBT_TO_EQUITY}")

    # Trend rules live in passes_trend_gate so the pre-fetch gate and the
    # real filter can never drift apart.
    _, trend_reasons = passes_trend_gate(t, trend_gate)
    reasons.extend(trend_reasons)

    return (len(reasons) == 0, reasons)


def prescreen(
    tickers: list[str],
    technicals: dict[str, dict[str, Any]],
    *,
    gate: TrendGate,
    cap: int,
    always: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[str], dict[str, list[str]], int]:
    """(passed, {ticker: reasons} for the rest, how many the cap cut).

    The trend gate, then the cap on what goes on to the expensive fetches,
    with `always` (the user's own names) passing both. The strict gate
    ranks by 6-month relative strength; the soft gate by closeness to the
    ideal entry (10% below the high), so the cap doesn't quietly
    reintroduce a momentum filter. The discover run and the nightly cache
    warm-up (cli/earnings_watch) both call this, so they pick the same names.
    """
    reasons: dict[str, list[str]] = {}
    passed: list[str] = []
    for ticker in tickers:
        ok, why = passes_trend_gate(technicals.get(ticker), gate)
        if ok or ticker in always:
            passed.append(ticker)
        else:
            reasons[ticker] = why
    if len(passed) <= cap:
        return passed, reasons, 0

    def rank_key(t: str) -> float:
        tech = technicals.get(t) or {}
        if gate == "strict":
            return tech.get("rs_6mo") or 0.0
        dist = tech.get("dist_from_52w_high")
        return -abs(dist - IDEAL_ENTRY_DRAWDOWN) if dist is not None else -1.0

    ranked = sorted((t for t in passed if t not in always), key=rank_key, reverse=True)
    keep = set(ranked[: max(0, cap - len(always))]) | set(always)
    capped_out = [t for t in passed if t not in keep]
    for ticker in capped_out:
        reasons[ticker] = ["outside the screen cap for deep analysis"]
    return [t for t in passed if t in keep], reasons, len(capped_out)


def size_band(market_cap: float | None) -> str:
    return "large" if (market_cap or 0) >= LARGE_CAP else "mid"


def rank_within_size_bands(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scored candidates ordered by their percentile WITHIN their size band
    (large >= LARGE_CAP, mid below), raw score breaking ties: the best
    mid-cap stands level with the best large-cap. Each candidate's
    score_breakdown records its band and percentile. Needs `market_cap`
    on the candidate (its fundamentals')."""
    bands: dict[str, list[dict[str, Any]]] = {}
    for c in candidates:
        bands.setdefault(size_band(c.get("market_cap")), []).append(c)
    for members in bands.values():
        scores = sorted(c["score"] or 0 for c in members)
        n = len(scores)
        for c in members:
            v = c["score"] or 0
            below, equal = sum(x < v for x in scores), sum(x == v for x in scores)
            pct = (below + (equal + 1) / 2) / n  # average rank / n, as pandas' pct rank
            c["band_percentile"] = pct
            breakdown = c.get("score_breakdown")
            if isinstance(breakdown, dict):
                breakdown["size_band"] = {
                    "band": size_band(c.get("market_cap")),
                    "percentile": round(pct, 4),
                    "band_size": n,
                }
    return sorted(candidates, key=lambda c: (c["band_percentile"], c["score"] or 0), reverse=True)


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


# Contracted book (SEC remaining performance obligations), year on year.
# Measured 2026-09-20 on point-in-time filings, S&P 500, 2016-2026: 126-day
# excess-return IC +0.12, and it survived beta, sector, 12-1 momentum and
# revenue growth (which keeps nothing once the book is removed). Weighted
# like EPS revisions. Only ~36% of companies tag a book and the study only
# compared companies that do, so a name without one (or without a year-ago
# figure) gets the median points of the names that have one in the same run
# (`typical_book_points`): absence is never evidence, either way.
BOOK_FLAT_BELOW = 0.05
BOOK_FULL_AT = 0.50
BOOK_MAX_POINTS = 8.0
BOOK_SHRINKING_BELOW = -0.10
BOOK_SHRINKING_POINTS = -3.0


def _score_book(book: dict[str, Any] | None) -> float | None:
    """+8 for a book up 50%+ in a year, sliding to 0 at +5%; -3 once it
    has shrunk by 10% or more; None when not tagged or no year-ago figure."""
    yoy = (book or {}).get("yoy_pct")
    if yoy is None:
        return None
    growth = yoy / 100
    if growth <= BOOK_SHRINKING_BELOW:
        return BOOK_SHRINKING_POINTS
    span = BOOK_FULL_AT - BOOK_FLAT_BELOW
    return _clamp((growth - BOOK_FLAT_BELOW) / span * BOOK_MAX_POINTS, 0, BOOK_MAX_POINTS)


# Red flags in the latest SEC filing, as read by `read-filings`
# (data/filing_evidence.red_flags). Not measured like the book: the facts
# only exist since 2026-09-27, so there is no history to test against. The
# weights lean on the accounting literature instead — going-concern
# opinions and material-weakness disclosures precede underperformance —
# and stay small: enough to cost a borderline name its shortlist slot,
# never to outweigh a strong business. The pick scorecard records them
# (track_record) so the next review can check.
FILING_FLAG_POINTS = {"going_concern": -8.0, "material_weakness": -4.0, "restatement": -4.0}
FILING_FLAG_FLOOR = -8.0


def _score_filing_flags(flags: Iterable[str] | None) -> float:
    return max(FILING_FLAG_FLOOR, sum(FILING_FLAG_POINTS.get(f, 0.0) for f in flags or ()))


def fundamental_view(
    fundamentals: dict[str, Any],
    *,
    book: dict[str, Any] | None = None,
    revisions: dict[str, Any] | None = None,
    filing_flags: Iterable[str] | None = None,
) -> float:
    """The screen's own read of the business, without the price trend:
    fundamentals (growth, FCF yield, margins, debt), contracted-book growth
    and EPS revision flow. The dashboard sets it against the IBD-style
    Composite, which is mostly price."""
    total, _ = _score_fundamentals(fundamentals)
    total += _score_book(book) or 0.0
    total += _score_filing_flags(filing_flags)
    direction = (revisions or {}).get("direction_30d")
    total += 8.0 if direction == "raising" else -3.0 if direction == "lowering" else 0.0
    return total


def typical_book_points(books: Iterable[dict[str, Any] | None]) -> float:
    """Median book points among the names that have a scoreable book —
    what a name without one is given. 0 when none do."""
    points = [p for p in map(_score_book, books) if p is not None]
    return round(median(points), 1) if points else 0.0


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
        ideal = IDEAL_ENTRY_DRAWDOWN
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
# An earnings standout IS evidence, but the same evidence the trend score
# already rewards (rising EPS revisions); counting it here too would score
# it twice. An insider cluster is evidence too, but not yet proven enough
# to score (data/insider_buying.py): eligible, graded live, no bonus. An
# IBD-style leader is price strength the trend score already counts.
_NON_EVIDENCE_SOURCES = frozenset(
    {"index", "watchlist", "holding", "earnings_standout", "insider_cluster", "ibd_leader"}
)


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
    book: dict[str, Any] | None = None,
    no_book_points: float = 0.0,
    filing_flags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Combine the three scoring dimensions into a 0-100 total + breakdown.

    `revisions` is the per-ticker EPS-revisions summary (the LLM-stage
    payload's `eps_revisions` field). When present, the trend score picks
    up +8 / -3 based on direction_30d — analyst revision flow is one of the
    few forward-looking signals here, so it carries more weight than the
    media-mention count does. Optional so unit tests + legacy callers can
    still pass three args.

    `book` is the name's contracted-book record (data/backlog.fetch_rpo);
    its year-on-year growth adds up to +8 / -3 to fundamentals, and a name
    without one gets `no_book_points` (the run's `typical_book_points`).

    `filing_flags` are the red-flag categories in the name's latest SEC
    filing (see FILING_FLAG_POINTS); absent means none were read."""
    fund_total, fund_parts = _score_fundamentals(fundamentals)
    book_points = _score_book(book)
    fund_parts["contracted_book"] = no_book_points if book_points is None else book_points
    fund_total += fund_parts["contracted_book"]
    # Always present (0 without flags) so score_attribution can measure it
    # across every candidate, not only the flagged ones.
    fund_parts["filing_red_flags"] = _score_filing_flags(filing_flags)
    fund_total += fund_parts["filing_red_flags"]
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


def diversify_shortlist(
    ranked: list[dict[str, Any]],
    limit: int,
    *,
    max_per_sector: int = MAX_PER_SECTOR_SHORTLIST,
    max_per_industry: int = MAX_PER_INDUSTRY_SHORTLIST,
) -> list[dict[str, Any]]:
    """The first `limit` of `ranked` (best first) with at most
    `max_per_sector` per sector and `max_per_industry` per industry, the
    next-best names taking the freed slots. If the caps leave the list
    short (few sectors pass), the skipped names fill it back up in rank
    order, so the caps never cost a slot. A name without a sector or
    industry isn't capped on it. Skipped names get
    `shortlist_skipped` with the reason."""
    picked: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    sectors: dict[str, int] = {}
    industries: dict[str, int] = {}
    for c in ranked:
        if len(picked) >= limit:
            break
        sector, industry = c.get("sector"), c.get("industry")
        if sector and sectors.get(sector, 0) >= max_per_sector:
            c["shortlist_skipped"] = f"already {max_per_sector} {sector} names on the shortlist"
            skipped.append(c)
            continue
        if industry and industries.get(industry, 0) >= max_per_industry:
            c["shortlist_skipped"] = f"already {max_per_industry} {industry} names on the shortlist"
            skipped.append(c)
            continue
        picked.append(c)
        if sector:
            sectors[sector] = sectors.get(sector, 0) + 1
        if industry:
            industries[industry] = industries.get(industry, 0) + 1
    for c in skipped[: max(0, limit - len(picked))]:
        c.pop("shortlist_skipped", None)
        picked.append(c)
    order = {id(c): i for i, c in enumerate(ranked)}
    return sorted(picked, key=lambda c: order[id(c)])
