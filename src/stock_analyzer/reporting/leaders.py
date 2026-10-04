"""Data for the dashboard's Market leaders tab.

The IBD-style ratings (cli/ibd.py) side by side with everything this app
already knows about a stock: the holding's review, recent discover picks
and screen scores, the contracted book, EPS revision flow, hedge-fund 13F
moves, insider buying clusters and earnings standouts. Plus six months of
daily bars per listed stock for the chart, read from the on-disk bar store,
each stock's rating history (ibd_history), and every logged buy-zone signal
graded against SPY (ibd_signals), so progress can be compared over time.

Nothing here makes a network request: every source is the database or a
file cache written by the jobs that fetch.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
from sqlalchemy import select, text

from ..data import bar_store, fetch_cache
from ..data.filing_evidence import red_flags
from ..db.session import exec_sql, get_session
from ..db.tables import IbdMarket, IbdRating
from ..logging import get_logger

logger = get_logger(__name__)

# The leaders list carries this many by Composite; holdings, picks, names
# with one of our signals and names near a buy point are added whatever
# their rank. Each listed stock carries a chart (~6 KB), so all ~1,900
# would make the page ~12 MB.
LEADERS = 200
CHART_DAYS = 126  # six months of daily bars, as MarketSurge's default view
PICK_DAYS = 180


def _rows(db: str) -> tuple[list[Any], list[Any]]:
    # Plain copies: the rows are read after the session closes.
    with get_session(db) as session:
        rows = [SimpleNamespace(**asdict(r)) for r in session.scalars(select(IbdRating))]
        market = [
            SimpleNamespace(**asdict(m))
            for m in session.scalars(select(IbdMarket).order_by(IbdMarket.day.desc()).limit(30))
        ]
    return rows, market


def _safely(what: str, fn, default):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — one signal, not the tab
        logger.warning("Market leaders: %s unavailable (%s)", what, e)
        return default


def _picks(db: str, today: date) -> dict[str, dict[str, Any]]:
    """Latest discover pick per ticker in the last PICK_DAYS."""
    since = (today - timedelta(days=PICK_DAYS)).isoformat()
    out: dict[str, dict[str, Any]] = {}
    with get_session(db) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT p.ticker, p.rank, p.conviction, p.ev_pct, r.run_at, r.id FROM picks p "
                "JOIN runs r ON r.id = p.run_id WHERE r.run_at >= :since ORDER BY r.run_at"
            ),
            params={"since": since},
        ).all()
    for ticker, rank, conviction, ev, run_at, run_id in rows:
        out[ticker] = {
            "rank": rank,
            "conv": conviction,
            "ev": ev,
            "d": str(run_at)[:10],
            "run": run_id,
        }
    return out


def _screen_scores(db: str) -> dict[str, float]:
    """Each ticker's score in the latest run that scored candidates."""
    with get_session(db) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT ticker, score FROM candidates WHERE score IS NOT NULL AND run_id = "
                "(SELECT MAX(run_id) FROM candidates WHERE score IS NOT NULL)"
            ),
        ).all()
    return {t: round(float(s), 1) for t, s in rows}


def _cached(kind: str, field: str) -> dict[str, Any]:
    return {
        t: (e.get("value") or {}).get(field)
        for t, e in fetch_cache.entries(kind).items()
        if (e.get("value") or {}).get(field) is not None
    }


def signals(db: str, tickers: list[str], today: date) -> dict[str, dict[str, Any]]:
    """{ticker: our data} for the tickers that have any."""
    from ..data.hedge_funds_13f import changes, summarize
    from ..data.insider_buying import clusters
    from ..discover.earnings_standouts import recent_standouts

    picks = _safely("picks", lambda: _picks(db, today), {})
    scores = _safely("screen scores", lambda: _screen_scores(db), {})
    book = _safely("contracted book", lambda: _cached("contracted_book", "yoy_pct"), {})
    revisions = _safely("EPS revisions", lambda: _cached("eps_revisions", "direction_30d"), {})
    funds = _safely("13F moves", lambda: changes(db, tickers), {})
    insiders = {
        c["ticker"]: c for c in _safely("insider clusters", lambda: clusters(db, today=today), [])
    }
    standouts = {
        s["ticker"]: s
        for s in _safely("standouts", lambda: recent_standouts(db, days=60, today=today), [])
    }
    out: dict[str, dict[str, Any]] = {}
    for t in tickers:
        s: dict[str, Any] = {}
        if t in picks:
            s["pick"] = picks[t]
        if t in scores:
            s["score"] = scores[t]
        if book.get(t) is not None:
            s["book"] = round(float(book[t]), 1)
        if revisions.get(t):
            s["rev"] = revisions[t]
        if funds.get(t):
            s["funds"] = summarize(funds[t])
            s["funds_net"] = sum(1 if m["action"] in ("new", "added") else -1 for m in funds[t])
        if t in insiders:
            c = insiders[t]
            s["insiders"] = {
                "n": c.get("buyers"),
                "usd": c.get("value_usd"),
                "d": c.get("formed_on"),
            }
        if t in standouts:
            st = standouts[t]
            s["standout"] = {
                "d": st.get("report_date"),
                "react": st.get("reaction_pct"),
                "rev": st.get("revision_pct"),
            }
        if s:
            out[t] = s
    return out


def _series(frame, dates: list[date]) -> dict[str, list]:
    """HLC, volume, 50/200-day averages on the chart's calendar."""
    closes = frame["Close"].cast(float).to_numpy()
    days = frame["date"].to_list()
    at = {d: i for i, d in enumerate(days)}

    def avg(window: int) -> np.ndarray:
        out = np.full(len(closes), np.nan)
        if len(closes) >= window:
            c = np.cumsum(np.insert(closes, 0, 0.0))
            out[window - 1 :] = (c[window:] - c[:-window]) / window
        return out

    ma50, ma200 = avg(50), avg(200)
    vols = frame["Volume"].cast(float).to_numpy()
    vavg = np.full(len(vols), np.nan)
    if len(vols) >= 50:
        v = np.cumsum(np.insert(vols, 0, 0.0))
        vavg[49:] = (v[50:] - v[:-50]) / 50
    cols = {k: frame[k].cast(float).to_numpy() for k in ("High", "Low", "Close")}

    def pick(arr: np.ndarray, d: date, digits: int = 2):
        i = at.get(d)
        if i is None or not np.isfinite(arr[i]):
            return None
        return round(float(arr[i]), digits)

    return {
        "h": [pick(cols["High"], d) for d in dates],
        "l": [pick(cols["Low"], d) for d in dates],
        "c": [pick(cols["Close"], d) for d in dates],
        "v": [None if (x := pick(vols, d, 0)) is None else int(x / 1000) for d in dates],
        "va": [None if (x := pick(vavg, d, 0)) is None else int(x / 1000) for d in dates],
        "m50": [pick(ma50, d) for d in dates],
        "m200": [pick(ma200, d) for d in dates],
    }


def charts(tickers: list[str]) -> tuple[list[str], dict[str, dict[str, list]]]:
    """(calendar, {ticker: series}) for the last CHART_DAYS sessions, from
    the bar store only. The RS line is close / SPY close, rebased to 100."""
    spy = bar_store.load("SPY")
    if spy is None or spy.frame.height < CHART_DAYS:
        return [], {}
    dates = spy.frame["date"].to_list()[-CHART_DAYS:]
    spy_close = dict(zip(spy.frame["date"].to_list(), spy.frame["Close"].to_list(), strict=True))
    out: dict[str, dict[str, list]] = {}
    for t in tickers:
        stored = bar_store.load(t)
        if stored is None or stored.frame.height < 60:
            continue
        s = _series(stored.frame, dates)
        ratio = [
            c / spy_close[d] if c is not None and spy_close.get(d) else None
            for c, d in zip(s["c"], dates, strict=True)
        ]
        first = next((r for r in ratio if r), None)
        s["rs"] = [None if r is None or not first else round(r / first * 100, 2) for r in ratio]
        out[t] = s
    return [d.isoformat() for d in dates], out


HISTORY_DAYS = 180
# Rating history rides along for these only (it is ~2 KB a stock).
HISTORY_TOP = 50
HORIZONS = (21, 63, 126)  # trading days: about 1, 3 and 6 months


def rating_history(db: str, tickers: list[str], today: date) -> dict[str, dict[str, list]]:
    """{ticker: {"d", "c", "rs"}}: Composite and RS over the last half year."""
    if not tickers:
        return {}
    since = (today - timedelta(days=HISTORY_DAYS)).isoformat()
    wanted = set(tickers)
    out: dict[str, dict[str, list]] = {}
    with get_session(db) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT ticker, day, composite, rs_rating FROM ibd_history "
                "WHERE day >= :since ORDER BY day"
            ),
            params={"since": since},
        ).all()
    for ticker, day, comp, rs in rows:
        if ticker in wanted:
            h = out.setdefault(ticker, {"d": [], "c": [], "rs": []})
            h["d"].append(day)
            h["c"].append(comp)
            h["rs"].append(rs)
    return {t: h for t, h in out.items() if len(h["d"]) >= 2}


def _forward(frame, day: str, horizon: int) -> tuple[float | None, int]:
    """(return from the close on `day` to `horizon` sessions later, or to
    the latest close when younger; sessions elapsed)."""
    days = [d.isoformat() for d in frame["date"].to_list()]
    closes = frame["Close"].to_list()
    try:
        i = days.index(day)
    except ValueError:
        return None, 0

    def priced(x) -> bool:  # a missing close is None or NaN (a partial bar)
        return bool(x) and x == x

    # Back off to the last priced session: one missing close cost the whole
    # scorecard its page on 2026-10-01.
    j = min(i + horizon, len(days) - 1)
    while j > i and not priced(closes[j]):
        j -= 1
    if j <= i or not priced(closes[i]):
        return None, 0
    return closes[j] / closes[i] - 1, j - i


def signal_scorecard(db: str) -> dict[str, Any]:
    """Every logged buy-zone signal with its return and SPY's over the same
    sessions at each horizon (partial while younger), and per-horizon
    averages over the signals old enough to have it."""
    with get_session(db) as session:
        signals = [
            dict(
                zip(("t", "d", "st", "p", "piv", "c", "rs", "eps", "base", "bf"), row, strict=True)
            )
            for row in exec_sql(
                session,
                text(
                    "SELECT ticker, day, status, price, pivot, composite, rs_rating, eps_rating, base, "
                    "backfilled FROM ibd_signals ORDER BY day DESC"
                ),
            ).all()
        ]
    if not signals:
        return {"signals": [], "summary": []}
    spy = bar_store.load("SPY")
    frames: dict[str, Any] = {}
    for sgn in signals:
        stored = frames.setdefault(sgn["t"], bar_store.load(sgn["t"]))
        sgn["h"] = {}
        for h in HORIZONS:
            mine, n = _forward(stored.frame, sgn["d"], h) if stored else (None, 0)
            base, _ = _forward(spy.frame, sgn["d"], n) if spy and n else (None, 0)
            # String keys: the page's JSON writer takes no others (JS reads h[21] alike).
            sgn["h"][str(h)] = {
                "ret": None if mine is None else round(mine * 100, 1),
                "spy": None if base is None else round(base * 100, 1),
                "done": n >= h,
            }
    summary = []
    for h in HORIZONS:
        k = str(h)
        done = [
            s["h"][k]
            for s in signals
            if s["h"][k]["done"] and s["h"][k]["ret"] is not None and s["h"][k]["spy"] is not None
        ]
        excess = [x["ret"] - x["spy"] for x in done]
        summary.append(
            {
                "h": h,
                "n": len(done),
                "avg_ret": round(sum(x["ret"] for x in done) / len(done), 1) if done else None,
                "avg_excess": round(sum(excess) / len(excess), 1) if excess else None,
                "beat": round(sum(1 for e in excess if e > 0) / len(excess) * 100)
                if excess
                else None,
            }
        )
    # All of them: the page filters to the selected stock, whose signals may be old.
    return {
        "signals": signals,
        "summary": summary,
        "total": len(signals),
        "backfilled": sum(1 for s in signals if s["bf"]),
    }


# Re-check prompts for holdings (the daily email). Market signals, so for a
# 3-5 year holding they ask for the thesis to be re-read, never a sale.
DROP_POINTS = 25  # Composite fall over about a month
DROP_WINDOW_DAYS = 30
WEAK_RS = 30
BREAK_DAYS = 5  # a 200-day break within the last week
BREAK_VOLUME = 1.4  # x the 50-day average volume


def _two_hundred_day_break(frame) -> dict[str, Any] | None:
    """The day in the last BREAK_DAYS sessions the close fell below the
    200-day average on heavy volume, if the stock is still below it."""
    closes = frame["Close"].cast(float).to_numpy()
    vols = frame["Volume"].cast(float).to_numpy()
    n = len(closes)
    if n < 205:
        return None
    ma = np.convolve(closes, np.ones(200) / 200, mode="valid")  # ma[k] ends at bar k+199
    if closes[-1] >= ma[-1]:
        return None
    for i in range(n - BREAK_DAYS, n):
        k = i - 199
        below, was_above = closes[i] < ma[k], closes[i - 1] >= ma[k - 1]
        avg = vols[max(0, i - 50) : i].mean()
        if below and was_above and avg > 0 and vols[i] >= BREAK_VOLUME * avg:
            return {
                "day": frame["date"][i].isoformat(),
                "volume_x": round(float(vols[i] / avg), 1),
                "ma200": round(float(ma[k]), 2),
            }
    return None


def holding_checks(db: str, held: list[str], *, today: date) -> list[dict[str, Any]]:
    """[{ticker, reasons: [...], alternative}] for holdings whose market
    ratings turned: Composite down DROP_POINTS+ in a month, RS under
    WEAK_RS, or a 200-day break on heavy volume. `alternative` is the
    strongest stock in the same industry group that is not held."""
    rows, _ = _rows(db)
    if not rows:
        return []
    by = {r.ticker: r for r in rows}
    held_set = set(held)
    since = (today - timedelta(days=DROP_WINDOW_DAYS + 7)).isoformat()
    past: dict[str, list[tuple[str, int]]] = {}
    with get_session(db) as session:
        for ticker, day, comp in exec_sql(
            session,
            text(
                "SELECT ticker, day, composite FROM ibd_history WHERE day >= :since "
                "AND composite IS NOT NULL ORDER BY day"
            ),
            params={"since": since},
        ).all():
            if ticker in held_set:
                past.setdefault(ticker, []).append((day, comp))
    best_in_group: dict[str, Any] = {}
    for r in sorted(rows, key=lambda r: -(r.composite or 0)):
        if r.industry and r.ticker not in held_set and r.industry not in best_in_group:
            best_in_group[r.industry] = r
    cutoff = (today - timedelta(days=DROP_WINDOW_DAYS)).isoformat()
    out = []
    for t in held:
        r = by.get(t)
        if r is None or r.composite is None:
            continue
        reasons = []
        then = [c for d, c in past.get(t, []) if d <= cutoff]
        if then and then[-1] - r.composite >= DROP_POINTS:
            reasons.append(f"Composite {then[-1]} → {r.composite} in a month")
        if r.rs_rating is not None and r.rs_rating < WEAK_RS:
            reasons.append(f"RS Rating {r.rs_rating}: lagging most stocks over a year")
        stored = bar_store.load(t)
        brk = _two_hundred_day_break(stored.frame) if stored is not None else None
        if brk:
            reasons.append(
                f"fell below its 200-day average (${brk['ma200']:,.2f}) on {brk['day']} "
                f"on {brk['volume_x']}x normal volume"
            )
        if not reasons:
            continue
        alt = best_in_group.get(r.industry or "")
        out.append(
            {
                "ticker": t,
                "composite": r.composite,
                "rs": r.rs_rating,
                "reasons": reasons,
                "alternative": {
                    "ticker": alt.ticker,
                    "composite": alt.composite,
                    "industry": alt.industry,
                }
                if alt is not None and (alt.composite or 0) > (r.composite or 0)
                else None,
            }
        )
    return out


def held_sector_trends(
    db: str, sector_by_ticker: dict[str, str], values: dict[str, float]
) -> list[dict[str, Any]]:
    """The latest direction of each sector you hold (semiconductors as
    their own line), with your weight in it and your holdings there from
    weakest Composite up, for the daily email."""
    view = sector_view(db, date.today())
    if not view:
        return []
    latest = {x["sector"]: x for x in view["latest"]}
    rows, _ = _rows(db)
    by = {r.ticker: r for r in rows}
    total = sum(v for v in values.values() if v) or 1.0
    groups: dict[str, list[str]] = {}
    for t in values:
        sector = sector_by_ticker.get(t)
        if sector:
            groups.setdefault(sector, []).append(t)
        if by.get(t) is not None and by[t].industry == "Semiconductors":
            groups.setdefault("Semiconductors", []).append(t)
    out = []
    for sector, tickers in groups.items():
        x = latest.get(sector)
        if x is None:
            continue
        weakest = sorted(
            (t for t in tickers if by.get(t) is not None and by[t].composite is not None),
            key=lambda t: by[t].composite,
        )
        out.append(
            {
                "sector": sector,
                "status": x["status"],
                "reasons": x["reasons"],
                "rank": x["rank"],
                "day": view["day"],
                "pct": round(sum(values.get(t) or 0 for t in tickers) / total * 100, 1),
                "holdings": [{"ticker": t, "composite": by[t].composite} for t in weakest],
            }
        )
    order = {"Correction": 0, "Caution": 1, "Leading": 2, "Uptrend": 3}
    return sorted(out, key=lambda o: (order.get(o["status"], 4), -o["pct"]))


def view_map(
    rows: list[Any],
    held: set[str],
    ours: dict[str, dict[str, Any]],
    flags: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Points for the "our view vs the market's" map: every rated stock with
    cached fundamentals (the screen's ~600, holdings included), our
    fundamental view and the Composite, both as 1-99 ranks among them."""
    from ..discover.ibd_ratings import percentile_ranks
    from ..discover.screen import fundamental_view

    fundamentals: dict[str, dict[str, Any]] = {
        t: e["value"] for t, e in fetch_cache.entries("fundamentals").items() if e.get("value")
    }
    books = {t: e.get("value") for t, e in fetch_cache.entries("contracted_book").items()}
    revisions = {t: e.get("value") for t, e in fetch_cache.entries("eps_revisions").items()}
    rated = {r.ticker: r for r in rows if r.composite is not None and fundamentals.get(r.ticker)}
    raw = {
        t: fundamental_view(
            fundamentals[t],
            book=books.get(t),
            revisions=revisions.get(t),
            filing_flags=(flags or {}).get(t),
        )
        for t in rated
    }
    view = percentile_ranks(raw)
    market = percentile_ranks({t: float(r.composite) for t, r in rated.items()})
    return [
        {
            "t": t,
            "x": view[t],
            "y": market[t],
            "comp": rated[t].composite,
            "ind": rated[t].industry,
            "held": t in held,
            "pick": "pick" in (ours.get(t) or {}),
        }
        for t in rated
        if t in view and t in market
    ]


SECTOR_HISTORY_DAYS = 45  # calendar days of sector status for the strip
GROUP_CHART_SESSIONS = 252  # a year of trading days
GROUP_CHART_TOP = 5


def sector_view(db: str, today: date) -> dict[str, Any]:
    """{"latest": [row per sector], "history": {sector: [{d, s}]}} from
    ibd_sectors (cli/ibd.py)."""
    since = (today - timedelta(days=SECTOR_HISTORY_DAYS)).isoformat()
    with get_session(db) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT day, sector, status, reasons, rank, stocks, breadth, breadth_before, "
                "median_six, leaders, etf, etf_above_50, etf_above_200, etf_dist_days "
                "FROM ibd_sectors WHERE day >= :since ORDER BY day"
            ),
            params={"since": since},
        ).all()
    if not rows:
        return {}
    keys = (
        "day",
        "sector",
        "status",
        "reasons",
        "rank",
        "stocks",
        "breadth",
        "breadth_before",
        "median_six",
        "leaders",
        "etf",
        "etf_above_50",
        "etf_above_200",
        "etf_dist_days",
    )
    dicts = [dict(zip(keys, r, strict=True)) for r in rows]
    last = max(d["day"] for d in dicts)
    history: dict[str, list[dict[str, str]]] = {}
    for d in dicts:
        history.setdefault(d["sector"], []).append({"d": d["day"], "s": d["status"]})
    latest = sorted(
        (d for d in dicts if d["day"] == last), key=lambda d: (d["rank"] or 0.5, d["sector"])
    )
    return {"day": last, "latest": latest, "history": history}


def group_chart(rows: list[Any], held: set[str]) -> dict[str, Any]:
    """A year of each industry group's price, as an equal-weight index of its
    rated stocks (daily returns averaged, rebased to 100), for the top
    GROUP_CHART_TOP groups by rank and the groups of your holdings; SPY on
    the same calendar for reference. From the bar store only."""
    spy = bar_store.load("SPY")
    if spy is None or spy.frame.height <= GROUP_CHART_SESSIONS:
        return {}
    calendar = spy.frame["date"].to_list()[-GROUP_CHART_SESSIONS - 1 :]
    at = {d: i for i, d in enumerate(calendar)}
    ranked = sorted({(r.group_rank, r.industry) for r in rows if r.group_rank and r.industry})
    top = [g for _, g in ranked[:GROUP_CHART_TOP]]
    mine = sorted({r.industry for r in rows if r.ticker in held and r.industry})
    rank_of = {g: rk for rk, g in ranked}
    groups = list(dict.fromkeys(top + mine))
    members: dict[str, list[str]] = {}
    for r in rows:
        if r.industry in groups:
            members.setdefault(r.industry, []).append(r.ticker)

    def index(tickers: list[str]) -> list[float | None]:
        total = np.zeros(len(calendar))
        count = np.zeros(len(calendar))
        for t in tickers:
            stored = bar_store.load(t)
            if stored is None:
                continue
            closes = np.full(len(calendar), np.nan)
            for d, c in zip(
                stored.frame["date"].to_list(), stored.frame["Close"].to_list(), strict=True
            ):
                i = at.get(d)
                if i is not None and c:
                    closes[i] = c
            ret = closes[1:] / closes[:-1] - 1
            ok = np.isfinite(ret)
            total[1:][ok] += ret[ok]
            count[1:][ok] += 1
        daily = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        level = 100 * np.cumprod(1 + daily)
        return [round(float(v), 2) for v in level]

    spy_close = spy.frame["Close"].to_list()[-GROUP_CHART_SESSIONS - 1 :]
    return {
        "dates": [d.isoformat() for d in calendar],
        "spy": [round(c / spy_close[0] * 100, 2) for c in spy_close],
        "groups": [
            {
                "name": g,
                "rank": rank_of.get(g),
                "n": len(members.get(g, [])),
                "top": g in top,
                "held": g in mine,
                "series": index(members.get(g, [])),
            }
            for g in groups
            if members.get(g)
        ],
    }


_PROFILE_FIELDS = (
    "name", "summary", "website", "employees", "hq", "market_cap", "trailing_pe", "forward_pe",
    "revenue_growth_yoy", "earnings_growth_yoy", "operating_margin", "profit_margin",
    "analyst_count", "analyst_recommendation", "analyst_target_mean", "analyst_target_upside_pct",
    "next_earnings",
)  # fmt: skip


def company_profiles(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """{ticker: profile} from the fundamentals cache (no request): what the
    company does, its size and valuation, growth, margins and analysts."""
    cached = fetch_cache.entries("fundamentals")
    out = {}
    for t in tickers:
        f = (cached.get(t) or {}).get("value") or {}
        profile = {k: f.get(k) for k in _PROFILE_FIELDS if f.get(k) is not None}
        if profile:
            out[t] = profile
    return out


def stock_news(db: str, tickers: list[str]) -> dict[str, dict[str, Any]]:
    """{ticker: {"fetched", "items"}} from this morning's stock_news."""
    wanted = set(tickers)
    out: dict[str, dict[str, Any]] = {}
    with get_session(db) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT ticker, rank, fetched, title, url, source, published, snippet "
                "FROM stock_news ORDER BY ticker, rank"
            ),
        ).all()
    for ticker, _rank, fetched, title, url, source, published, snippet in rows:
        if ticker in wanted:
            entry = out.setdefault(ticker, {"fetched": fetched, "items": []})
            entry["items"].append(
                {"title": title, "url": url, "source": source, "d": published, "snippet": snippet}
            )
    return out


def collect(db: str, *, held: set[str], today: date) -> dict[str, Any]:
    """Everything the Market leaders tab shows, or {} before the first run."""
    rows, market = _rows(db)
    if not rows:
        return {}
    rows.sort(key=lambda r: -(r.composite or 0))
    ours = signals(db, [r.ticker for r in rows], today)
    near = {"buy zone", "breakout"}

    def listed(i: int, r: Any) -> bool:
        return (
            i < LEADERS
            or r.ticker in held
            or r.ticker in ours
            and any(k in ours[r.ticker] for k in ("pick", "insiders", "standout"))
            or r.base_status in near
            or (
                r.base_status == "below pivot"
                and (r.vs_pivot or -1) >= -0.05
                and (r.composite or 0) >= 70
            )
        )

    keep = [r for i, r in enumerate(rows) if listed(i, r)]
    from ..data.industry_groups import sector_map

    sector_of = _safely("sector map", sector_map, {})
    groups: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.industry and r.group_rank:
            g = groups.setdefault(
                r.industry, {"name": r.industry, "rank": r.group_rank, "n": 0, "leaders": []}
            )
            g["n"] += 1
            if (r.composite or 0) >= 90:
                g["leaders"].append(r.ticker)
    calendar, series = charts([r.ticker for r in keep])
    with_history = [
        r.ticker
        for i, r in enumerate(keep)
        if i < HISTORY_TOP or r.ticker in held or r.ticker in ours or r.base_status in near
    ]
    history = _safely("rating history", lambda: rating_history(db, with_history, today), {})
    scorecard = _safely("signal scorecard", lambda: signal_scorecard(db), {})
    points = _safely(
        "view map",
        lambda: view_map(rows, held, ours, red_flags(db, [r.ticker for r in rows], today=today)),
        [],
    )
    sectors = _safely("sector direction", lambda: sector_view(db, today), {})
    groups_chart = _safely("group chart", lambda: group_chart(rows, held), {})
    listed_tickers = [r.ticker for r in keep]
    profiles = _safely("company profiles", lambda: company_profiles(listed_tickers), {})
    news = _safely("stock news", lambda: stock_news(db, listed_tickers), {})
    latest = market[0] if market else None
    return {
        "as_of": rows[0].as_of,
        "rated": len(rows),
        "leaders": LEADERS,
        "groups_ranked": rows[0].groups_ranked,
        "market": {
            "status": latest.status,
            "detail": latest.detail,
            "indexes": json.loads(latest.indexes or "{}"),
        }
        if latest
        else None,
        "market_history": [{"d": m.day, "s": m.status} for m in reversed(market)],
        "groups": sorted(groups.values(), key=lambda g: g["rank"]),
        "calendar": calendar,
        "charts": series,
        "history": history,
        "scorecard": scorecard,
        "view_map": points,
        "sectors": sectors,
        "group_chart": groups_chart,
        "profiles": profiles,
        "news": news,
        "horizons": list(HORIZONS),
        "rows": [
            {
                "t": r.ticker,
                "held": r.ticker in held,
                "p": r.price,
                "ind": r.industry,
                "sec": sector_of.get(r.ticker),
                "c": r.composite,
                "rs": r.rs_rating,
                "eps": r.eps_rating,
                "ad": r.ad_grade,
                "g": r.group_rank,
                "rsh": r.rs_line_high,
                "off": r.off_high,
                "six": r.six_month,
                "q1": r.eps_q1,
                "es": r.eps_source,
                "base": r.base,
                "wk": r.base_weeks,
                "dep": r.base_depth,
                "piv": r.pivot,
                "vp": r.vs_pivot,
                "st": r.base_status,
                "ours": ours.get(r.ticker) or {},
            }
            for r in keep
        ],
    }


def leadership_block(db: str, tickers: list[str]) -> str:
    """The rebalancer's "Market leadership" input: each sector's direction
    and each given stock's IBD-style ratings. "" before the first run."""
    view = sector_view(db, date.today())
    rows, _ = _rows(db)
    if not view and not rows:
        return ""
    from ..data.industry_groups import sector_map

    sector_of = sector_map()
    status = {x["sector"]: x for x in (view or {}).get("latest", [])}
    lines = [
        f"Market leadership (IBD-style, as of the {(view or {}).get('day') or rows[0].as_of} close):"
    ]
    if status:
        lines.append("Sectors, strongest first:")
        for x in view["latest"]:
            why = (
                f": {x['reasons']}"
                if x["reasons"] and x["status"] in ("Caution", "Correction")
                else ""
            )
            rank = f" #{x['rank']}" if x["rank"] else ""
            lines.append(f"  {x['sector']}{rank}: {x['status'].upper()}{why}")
    by = {r.ticker: r for r in rows}
    groups_ranked = rows[0].groups_ranked if rows else None
    lines.append("Holdings and picks:")
    for t in dict.fromkeys(tickers):
        r = by.get(t)
        if r is None:
            lines.append(f"  {t}: not rated")
            continue
        sector = sector_of.get(t)
        sec = status.get("Semiconductors" if r.industry == "Semiconductors" else sector or "")
        lines.append(
            f"  {t}: Composite {r.composite}, RS {r.rs_rating}, EPS {r.eps_rating or 'n/a'}, "
            f"group {r.industry or '?'} #{r.group_rank or '?'} of {groups_ranked or '?'}"
            + (f"; sector {sec['sector']} {sec['status'].upper()}" if sec else "")
        )
    return "\n".join(lines)


def ratings_for(db: str, tickers: list[str]) -> dict[str, dict[str, Any]]:
    """{ticker: IBD-style ratings and its sector's direction}, for the
    discover Analyst's payload."""
    rows, _ = _rows(db)
    if not rows:
        return {}
    from ..data.industry_groups import sector_map

    sector_of = sector_map()
    status = {
        x["sector"]: x["status"] for x in (sector_view(db, date.today()) or {}).get("latest", [])
    }
    by = {r.ticker: r for r in rows}
    out = {}
    for t in tickers:
        r = by.get(t)
        if r is None:
            continue
        sector = sector_of.get(t)
        out[t] = {
            "composite": r.composite,
            "rs_rating": r.rs_rating,
            "eps_rating": r.eps_rating,
            "group": r.industry,
            "group_rank": r.group_rank,
            "groups_ranked": r.groups_ranked,
            "sector": sector,
            "sector_direction": status.get(
                "Semiconductors" if r.industry == "Semiconductors" else sector or ""
            ),
            "base_status": r.base_status,
        }
    return out
