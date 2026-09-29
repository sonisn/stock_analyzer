"""IBD-style ratings, chart bases and market direction, from public rules.

Investor's Business Daily (MarketSurge) publishes WHAT its ratings measure
but not their exact weights or data. These are reconstructions from the
rules IBD and William O'Neil describe, computed over every US-listed stock
worth $2B+ (~1,900 names) so the percentiles mean what IBD's mean:

  - RS Rating (1-99): 12-month price performance, the latest quarter
    counted twice (0.4 / 0.2 / 0.2 / 0.2 over the four quarters), ranked.
    "RS line at a high" is the stock-to-SPY ratio at its 52-week high.
  - EPS Rating (1-99): the latest two quarters' EPS growth over the same
    quarter a year earlier and the three-year annual EPS growth rate,
    ranked (SEC-filed EPS, data/quarterly_eps.py).
  - Group rank: industries ranked by their stocks' median 6-month price
    change, 1 = strongest (Yahoo's ~145 industries; IBD uses 197).
  - Acc/Dis (A-E): 13 weeks of volume weighted by where each day closed in
    its range; A = heaviest buying, E = heaviest selling (quintiles).
  - Composite (1-99): EPS and RS counted twice, plus group, Acc/Dis and
    closeness to the 52-week high, ranked. IBD also counts SMR (sales,
    margins, ROE); it is left out here, so this is a close cousin of theirs.
  - Base and buy point: the consolidation since the stock's last high, its
    depth and length, the pivot (the base's high plus 10 cents) and whether
    the price is below it, in the 5% buy zone, or extended. Flat base and
    cup (with or without a handle) are recognised; judgment calls such as
    a base's shape quality are not, so treat a label as a pointer to look.
  - Market direction: distribution days (index down 0.2%+ on higher
    volume, within 25 sessions) on SPY and QQQ, and follow-through days
    (day 4+ of a rally from a low, up 1.2%+ on higher volume).

Display and research only. None of it feeds the discover score until a
point-in-time study shows it adds to what the screen already measures.
No network, no LLM: pure functions over bars and EPS records.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import median
from typing import Any

import numpy as np
import polars as pl

# RS: the four most recent quarters, latest first, in trading days.
RS_QUARTER = 63
RS_WEIGHTS = (0.4, 0.2, 0.2, 0.2)
# Acc/Dis: 13 weeks.
AD_DAYS = 65
AD_GRADES = ("E", "D", "C", "B", "A")
# Groups: 6 months, and at least this many members to be ranked.
GROUP_DAYS = 126
MIN_GROUP_SIZE = 3
# Composite weights.
COMPOSITE_WEIGHTS = {"eps": 2.0, "rs": 2.0, "group": 1.0, "ad": 1.0, "high": 1.0}

# Bases (trading days; a week is 5).
MIN_BASE_DAYS = 25  # 5 weeks: flat base minimum
MIN_CUP_DAYS = 35  # 7 weeks
MAX_BASE_DAYS = 325  # 65 weeks
FLAT_MAX_DEPTH = 0.15
CUP_MAX_DEPTH = 0.35
PRIOR_UPTREND = 0.30  # the run-up into the base
PRIOR_UPTREND_DAYS = 126
HANDLE_MIN_DAYS, HANDLE_MAX_DAYS = 5, 25
HANDLE_MAX_DEPTH = 0.12
BUY_ZONE = 0.05
PIVOT_TICK = 0.10

# Market direction.
DIST_DROP = -0.002
DIST_WINDOW = 25
DIST_EXPIRE_RALLY = 0.05
UNDER_PRESSURE_DIST = 5
CORRECTION_DRAWDOWN = 0.10
FTD_MIN_DAY = 4
FTD_GAIN = 0.012


def _arrays(bars: pl.DataFrame) -> dict[str, np.ndarray]:
    return {
        c: bars[c].cast(pl.Float64).fill_null(strategy="forward").to_numpy()
        for c in ("High", "Low", "Close", "Volume")
    }


def percentile_ranks(values: Mapping[str, float | None]) -> dict[str, int]:
    """1-99 by rank among the non-missing values (ties share a rank)."""
    present = {k: v for k, v in values.items() if v is not None and np.isfinite(v)}
    if not present:
        return {}
    ordered = sorted(present.values())
    n = len(ordered)
    out: dict[str, int] = {}
    for key, value in present.items():
        below = np.searchsorted(ordered, value, side="left")
        equal = np.searchsorted(ordered, value, side="right") - below
        share = (below + (equal - 1) / 2) / max(n - 1, 1)
        out[key] = int(round(1 + share * 98))
    return out


# --- per-stock measures ------------------------------------------------------


def rs_strength(close: np.ndarray) -> float | None:
    """The weighted 12-month performance IBD's RS Rating ranks."""
    if len(close) <= RS_QUARTER * 4:
        return None
    last = close[-1]
    total = 0.0
    for q, weight in enumerate(RS_WEIGHTS, start=1):
        then = close[-1 - RS_QUARTER * q]
        if not then or then <= 0:
            return None
        total += weight * (last / then)
    return total


def rs_line_at_high(close: np.ndarray, spy_close: np.ndarray) -> bool | None:
    """True when stock/SPY is at its 52-week high (aligned last 252 days)."""
    n = min(len(close), len(spy_close), 252)
    if n < 60:
        return None
    line = close[-n:] / spy_close[-n:]
    return bool(line[-1] >= np.nanmax(line) * 0.999)


def accumulation(high: np.ndarray, low: np.ndarray, close: np.ndarray, vol: np.ndarray):
    """13-week money flow: volume weighted by where each day closed in its
    range (+1 at the high, -1 at the low), over total volume."""
    if len(close) < AD_DAYS:
        return None
    h, lo, c, v = high[-AD_DAYS:], low[-AD_DAYS:], close[-AD_DAYS:], vol[-AD_DAYS:]
    span = h - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        where = np.where(span > 0, ((c - lo) - (h - c)) / span, 0.0)
    total = np.nansum(v)
    return float(np.nansum(where * v) / total) if total > 0 else None


def ad_grade(rank: int | None) -> str | None:
    if rank is None:
        return None
    return AD_GRADES[min((rank - 1) * 5 // 99, 4)]


def eps_strength(eps: dict[str, Any] | None) -> float | None:
    """0.4 x latest quarter's growth + 0.2 x the one before + 0.4 x the
    three-year annual rate, growth rates clipped to [-100%, +300%] so one
    turnaround from a near-zero base cannot dominate. Needs the latest
    quarter; the others count when present."""
    if not eps or eps.get("q1_growth") is None:
        return None
    parts = [(0.4, eps.get("q1_growth")), (0.2, eps.get("q2_growth")), (0.4, eps.get("cagr_3y"))]
    got = [(w, float(np.clip(g, -1.0, 3.0))) for w, g in parts if g is not None]
    weight = sum(w for w, _ in got)
    return sum(w * g for w, g in got) / weight


@dataclass
class Base:
    kind: str  # "flat base", "cup with handle", "cup", or "" for none
    weeks: int
    depth: float  # fraction below the base's high
    pivot: float
    vs_pivot: float  # price / pivot - 1
    status: str  # "below pivot", "buy zone", "extended", "breakout"


def find_base(bars: pl.DataFrame) -> Base | None:
    """The consolidation since the last 52-week high, if it is a base."""
    a = _arrays(bars)
    high, low, close, vol = a["High"], a["Low"], a["Close"], a["Volume"]
    n = len(close)
    if n < MIN_BASE_DAYS + PRIOR_UPTREND_DAYS:
        return None
    window = high[-min(n, MAX_BASE_DAYS) :]
    start = n - len(window) + int(np.nanargmax(window))
    top = high[start]
    days = n - 1 - start
    price = close[-1]
    # Broke out of a base in the last few days: judge the base it left.
    recent_breakout = days < 5 and n > MIN_BASE_DAYS + 5
    if recent_breakout:
        prior = high[: n - 5]
        window = prior[-min(len(prior), MAX_BASE_DAYS) :]
        start = len(prior) - len(window) + int(np.nanargmax(window))
        top = high[start]
        days = len(prior) - 1 - start
        body_low = np.nanmin(low[start : n - 5])
    else:
        body_low = np.nanmin(low[start:])
    if days < MIN_BASE_DAYS or start < PRIOR_UPTREND_DAYS:
        return None
    run_up_from = np.nanmin(low[start - PRIOR_UPTREND_DAYS : start])
    if run_up_from <= 0 or top / run_up_from - 1 < PRIOR_UPTREND:
        return None
    depth = 1 - body_low / top
    end = n - 5 if recent_breakout else n
    kind, pivot_high = "", top
    if depth <= FLAT_MAX_DEPTH:
        kind = "flat base"
    elif depth <= CUP_MAX_DEPTH and days >= MIN_CUP_DAYS:
        kind = "cup"
        handle = _handle(high[start:end], low[start:end], top, body_low)
        if handle is not None:
            kind, pivot_high = "cup with handle", handle
    if not kind:
        return None
    pivot = pivot_high + PIVOT_TICK
    vs = price / pivot - 1
    if recent_breakout and vs >= 0:
        avg50 = np.nanmean(vol[-55:-5]) if n >= 55 else np.nanmean(vol[:-5])
        loud = np.nanmax(vol[-5:]) >= 1.4 * avg50 if avg50 else False
        status = (
            "breakout" if loud and vs <= BUY_ZONE else "extended" if vs > BUY_ZONE else "buy zone"
        )
    elif vs < 0:
        status = "below pivot"
    elif vs <= BUY_ZONE:
        status = "buy zone"
    else:
        status = "extended"
    return Base(kind, round(days / 5), round(depth, 3), round(pivot, 2), round(vs, 4), status)


def _handle(high: np.ndarray, low: np.ndarray, top: float, cup_low: float) -> float | None:
    """The handle's high: the right side's peak within the last 25 days,
    at least 5 days back, within 10% of the old high, followed by a pullback
    of at most 12% that stays in the upper half of the cup."""
    n = len(high)
    span = min(HANDLE_MAX_DAYS, n // 3)
    if span < HANDLE_MIN_DAYS:
        return None
    recent = high[-span:]
    peak = n - span + int(np.nanargmax(recent))
    days = n - peak
    h_top = high[peak]
    if days < HANDLE_MIN_DAYS or h_top < top * 0.90:
        return None
    h_low = np.nanmin(low[peak:])
    if 1 - h_low / h_top > HANDLE_MAX_DEPTH or h_low < cup_low + (top - cup_low) / 2:
        return None
    return float(h_top)


# --- the market --------------------------------------------------------------


def distribution_days(bars: pl.DataFrame) -> list[int]:
    """Indices (into `bars`) of live distribution days in the last 25
    sessions: down 0.2%+ on volume above the day before's, not yet cancelled
    by a 5% rally above that day's close."""
    a = _arrays(bars)
    close, vol = a["Close"], a["Volume"]
    n = len(close)
    out = []
    for i in range(max(1, n - DIST_WINDOW), n):
        change = close[i] / close[i - 1] - 1
        rallied_away = np.nanmax(close[i:]) >= close[i] * (1 + DIST_EXPIRE_RALLY)
        if change <= DIST_DROP and vol[i] > vol[i - 1] and not rallied_away:
            out.append(i)
    return out


def follow_through_since_low(bars: pl.DataFrame, low_index: int) -> int | None:
    """Index of the first follow-through day after the low, or None."""
    a = _arrays(bars)
    close, vol, low = a["Close"], a["Volume"], a["Low"]
    n = len(close)
    day = 0
    for i in range(low_index + 1, n):
        if low[i] < low[low_index]:  # undercut: the rally attempt resets
            low_index, day = i, 0
            continue
        day += 1
        change = close[i] / close[i - 1] - 1
        if day >= FTD_MIN_DAY and change >= FTD_GAIN and vol[i] > vol[i - 1]:
            return i
    return None


def market_direction(indexes: dict[str, pl.DataFrame]) -> dict[str, Any]:
    """{"status", "detail", per-index counts} from SPY and QQQ bars."""
    per: dict[str, dict[str, Any]] = {}
    for name, bars in indexes.items():
        if bars is None or bars.height < 60:
            continue
        a = _arrays(bars)
        close, low = a["Close"], a["Low"]
        year = close[-252:]
        drawdown = 1 - close[-1] / np.nanmax(year)
        low_index = len(close) - len(year) + int(np.nanargmin(low[-len(year) :]))
        # The low that matters is the one after the high.
        peak = len(close) - len(year) + int(np.nanargmax(year))
        if low_index < peak:
            low_index = peak + int(np.nanargmin(low[peak:]))
        ftd = follow_through_since_low(bars, low_index)
        per[name] = {
            "distribution_days": len(distribution_days(bars)),
            "drawdown": round(float(drawdown), 4),
            "follow_through": str(bars["date"][ftd]) if ftd is not None else None,
            "low": str(bars["date"][low_index]),
            "above_50dma": bool(close[-1] > np.nanmean(close[-50:])),
        }
    if not per:
        return {"status": "unknown", "detail": "no index data", "indexes": {}}
    worst_dd = max(p["drawdown"] for p in per.values())
    most_dist = max(p["distribution_days"] for p in per.values())
    in_correction = [
        n
        for n, p in per.items()
        if p["drawdown"] >= CORRECTION_DRAWDOWN and not p["follow_through"]
    ]
    if in_correction:
        status = "Market in correction"
        detail = (
            f"{', '.join(in_correction)} {worst_dd:.0%} off the high with no follow-through day yet"
        )
    elif most_dist >= UNDER_PRESSURE_DIST:
        status = "Uptrend under pressure"
        detail = f"{most_dist} distribution days in the last {DIST_WINDOW} sessions"
    else:
        status = "Confirmed uptrend"
        detail = f"{most_dist} distribution day(s) in the last {DIST_WINDOW} sessions"
    return {"status": status, "detail": detail, "indexes": per}


# --- the whole universe --------------------------------------------------------


def rate_universe(
    bars: dict[str, pl.DataFrame],
    *,
    spy: pl.DataFrame | None,
    industries: dict[str, str],
    eps: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per stock with enough history, all ratings filled in."""
    spy_close = _arrays(spy)["Close"] if spy is not None else None
    rows: dict[str, dict[str, Any]] = {}
    for ticker, frame in bars.items():
        if frame is None or frame.height < RS_QUARTER * 4 + 1:
            continue
        a = _arrays(frame)
        close = a["Close"]
        base = find_base(frame)
        rows[ticker] = {
            "ticker": ticker,
            "price": round(float(close[-1]), 2),
            "industry": industries.get(ticker),
            "_rs": rs_strength(close),
            "_ad": accumulation(a["High"], a["Low"], close, a["Volume"]),
            "_eps": eps_strength(eps.get(ticker)),
            "_six_month": float(close[-1] / close[-1 - GROUP_DAYS] - 1),
            # Breadth inputs for the sector check: above the 50-day average
            # today, and ten sessions ago.
            "above_50": bool(close[-1] > np.nanmean(close[-50:])),
            "above_50_before": bool(close[-11] > np.nanmean(close[-60:-10])),
            "off_high": round(float(close[-1] / np.nanmax(a["High"][-252:]) - 1), 4),
            "rs_line_high": rs_line_at_high(close, spy_close) if spy_close is not None else None,
            "eps_q1": (eps.get(ticker) or {}).get("q1_growth"),
            "eps_source": (eps.get(ticker) or {}).get("source", "sec") if ticker in eps else None,
            "base": base.kind if base else None,
            "base_weeks": base.weeks if base else None,
            "base_depth": base.depth if base else None,
            "pivot": base.pivot if base else None,
            "vs_pivot": base.vs_pivot if base else None,
            "base_status": base.status if base else None,
        }
    rs = percentile_ranks({t: r["_rs"] for t, r in rows.items()})
    eps_r = percentile_ranks({t: r["_eps"] for t, r in rows.items()})
    ad = percentile_ranks({t: r["_ad"] for t, r in rows.items()})
    high = percentile_ranks({t: r["off_high"] for t, r in rows.items()})

    by_group: dict[str, list[float]] = {}
    for r in rows.values():
        if r["industry"]:
            by_group.setdefault(r["industry"], []).append(r["_six_month"])
    group_score = {g: median(v) for g, v in by_group.items() if len(v) >= MIN_GROUP_SIZE}
    group_order = sorted(group_score, key=lambda g: -group_score[g])
    group_rank = {g: i + 1 for i, g in enumerate(group_order)}
    group_pct = percentile_ranks(group_score)

    composite_raw: dict[str, float | None] = {}
    for t, r in rows.items():
        g = group_pct.get(r["industry"] or "")
        parts = {
            "eps": eps_r.get(t),
            "rs": rs.get(t),
            "group": g,
            "ad": ad.get(t),
            "high": high.get(t),
        }
        got = {k: v for k, v in parts.items() if v is not None}
        if "rs" not in got:
            composite_raw[t] = None
            continue
        weight = sum(COMPOSITE_WEIGHTS[k] for k in got)
        composite_raw[t] = sum(COMPOSITE_WEIGHTS[k] * v for k, v in got.items()) / weight
    composite = percentile_ranks(composite_raw)

    out = []
    for t, r in rows.items():
        g = r["industry"]
        out.append(
            {
                **{k: v for k, v in r.items() if not k.startswith("_")},
                "composite": composite.get(t),
                "rs_rating": rs.get(t),
                "eps_rating": eps_r.get(t),
                "ad_grade": ad_grade(ad.get(t)),
                "group_rank": group_rank.get(g) if g else None,
                "groups_ranked": len(group_rank),
                "six_month": round(r["_six_month"], 4),
            }
        )
    out.sort(key=lambda r: -(composite.get(r["ticker"]) or 0))
    return out


# --- history and signals -----------------------------------------------------------

HISTORY_MIN_COMPOSITE = 90
SIGNAL_MIN_COMPOSITE = 80
SIGNAL_STATUSES = ("buy zone", "breakout")
SIGNAL_QUIET_DAYS = 30
FULL_SNAPSHOT_EVERY_DAYS = 7


def history_rows(
    rows: list[dict[str, Any]], *, tracked: set[str], full: bool
) -> list[dict[str, Any]]:
    """The rows worth keeping today: every row on a full-snapshot day,
    otherwise Composite 90+, anything at a buy point, and tracked stocks."""
    keep = [
        r
        for r in rows
        if full
        or (r.get("composite") or 0) >= HISTORY_MIN_COMPOSITE
        or r.get("base_status") in SIGNAL_STATUSES
        or r["ticker"] in tracked
    ]
    return [{**r, "full": full} for r in keep]


def new_signals(rows: list[dict[str, Any]], *, recent: set[str]) -> list[dict[str, Any]]:
    """Stocks in a buy zone or breaking out with Composite 80+, except the
    ones already signalled within SIGNAL_QUIET_DAYS (`recent`)."""
    return [
        r
        for r in rows
        if r.get("base_status") in SIGNAL_STATUSES
        and (r.get("composite") or 0) >= SIGNAL_MIN_COMPOSITE
        and r["ticker"] not in recent
    ]


# --- sector direction: is a sector still leading, or turning? --------------------------

# Each sector's SPDR ETF, plus semiconductors as their own line (an industry,
# not a sector, but the one this portfolio is heaviest in).
SECTOR_ETFS = {
    "Technology": "XLK",
    "Semiconductors": "SMH",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}
SEMIS_INDUSTRY = "Semiconductors"
CORRECTION_BREADTH = 0.30
CAUTION_BREADTH = 0.50
BREADTH_FALL = 0.10  # points of breadth lost over ten sessions
# An ETF must be this far under its 200-day to count as a correction; a
# close just below it is a caution, so a sector does not flip on a wiggle.
BELOW_200_MARGIN = 0.02
LEADING_TOP = 3
SECTOR_STATUSES = ("Leading", "Uptrend", "Caution", "Correction")


def _etf_state(bars: pl.DataFrame | None) -> dict[str, Any] | None:
    if bars is None or bars.height < 200:
        return None
    close = _arrays(bars)["Close"]
    ma200 = np.nanmean(close[-200:])
    return {
        "above_50": bool(close[-1] > np.nanmean(close[-50:])),
        "above_200": bool(close[-1] > ma200),
        "well_below_200": bool(close[-1] < ma200 * (1 - BELOW_200_MARGIN)),
        "dist_days": len(distribution_days(bars)),
    }


def sector_direction(
    rows: list[dict[str, Any]],
    *,
    sectors: dict[str, str],
    etfs: dict[str, pl.DataFrame | None],
) -> list[dict[str, Any]]:
    """One row per sector, strongest first: its status (Leading, Uptrend,
    Caution, Correction), the reasons, rank by median 6-month change,
    breadth (share of its stocks above their 50-day average, now and ten
    sessions ago), Composite 90+ count, and its ETF's trend.

      Correction: the ETF is more than 2% below its 200-day average, or
                  under 30% of the sector's stocks are above their 50-day.
      Caution:    the ETF is below its 50-day (or just below its 200-day),
                  or has 5+ distribution days, or breadth is under 50% and
                  fell 10+ points in two weeks.
      Leading:    none of that, and a top-3 sector by 6-month change.
      Uptrend:    none of that.
    """
    members: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        sector = sectors.get(r["ticker"])
        if sector:
            members.setdefault(sector, []).append(r)
        if r.get("industry") == SEMIS_INDUSTRY:
            members.setdefault(SEMIS_INDUSTRY, []).append(r)
    out: list[dict[str, Any]] = []
    for sector, group in members.items():
        if len(group) < MIN_GROUP_SIZE:
            continue
        n = len(group)
        breadth = sum(1 for r in group if r.get("above_50")) / n
        before = sum(1 for r in group if r.get("above_50_before")) / n
        etf = _etf_state(etfs.get(SECTOR_ETFS.get(sector, "")))
        ticker = SECTOR_ETFS.get(sector)
        reasons_red, reasons_amber = [], []
        if etf and etf["well_below_200"]:
            reasons_red.append(f"{ticker} more than 2% below its 200-day average")
        elif etf and not etf["above_200"]:
            reasons_amber.append(f"{ticker} just below its 200-day average")
        if breadth < CORRECTION_BREADTH:
            reasons_red.append(f"only {breadth:.0%} of its stocks above their 50-day")
        if etf and not etf["above_50"]:
            reasons_amber.append(f"{ticker} below its 50-day average")
        if etf and etf["dist_days"] >= UNDER_PRESSURE_DIST:
            reasons_amber.append(f"{ticker} has {etf['dist_days']} distribution days")
        if breadth < CAUTION_BREADTH and breadth - before <= -BREADTH_FALL:
            reasons_amber.append(f"breadth fell from {before:.0%} to {breadth:.0%} in two weeks")
        status = "Correction" if reasons_red else "Caution" if reasons_amber else "Uptrend"
        out.append(
            {
                "sector": sector,
                "status": status,
                "reasons": reasons_red or reasons_amber,
                "stocks": n,
                "breadth": round(breadth, 3),
                "breadth_before": round(before, 3),
                "median_six": round(float(median(r["six_month"] for r in group)), 4),
                "leaders": sum(1 for r in group if (r.get("composite") or 0) >= 90),
                "etf": ticker,
                "etf_above_50": etf["above_50"] if etf else None,
                "etf_above_200": etf["above_200"] if etf else None,
                "etf_dist_days": etf["dist_days"] if etf else None,
            }
        )
    # Semiconductors sit inside Technology; rank the sectors only.
    ranked = sorted(
        (o for o in out if o["sector"] != SEMIS_INDUSTRY), key=lambda o: -float(o["median_six"])
    )
    for i, o in enumerate(ranked, start=1):
        o["rank"] = i
        if o["status"] == "Uptrend" and i <= LEADING_TOP:
            o["status"] = "Leading"
    for o in out:
        if o["sector"] == SEMIS_INDUSTRY:
            tech = next((x for x in ranked if x["sector"] == "Technology"), None)
            o["rank"] = None
            if o["status"] == "Uptrend" and tech and tech["status"] == "Leading":
                o["status"] = "Leading"
    return sorted(out, key=lambda o: (o.get("rank") or 0.5, o["sector"]))
