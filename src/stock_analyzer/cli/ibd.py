"""`ibd-ratings` — IBD-style ratings for every $2B+ US stock, before the open.

Runs at 6 AM New York time (scripts/run_ibd.sh), then rebuilds the
dashboard so its "Market leaders" tab is current by the opening bell:

  1. bars for ~1,900 stocks from the bar store, extended by one batched
     Yahoo download per 100 symbols (~19 requests);
  2. industries from the weekly screener map (data/industry_groups);
  3. quarterly EPS from SEC filings, a week's cache renewed a fifth a day;
  4. ratings, bases and market direction (discover/ibd_ratings), stored
     in `ibd_ratings` (replaced) and `ibd_market` (one row a day);
  5. history for comparing progress: `ibd_history` (daily for the names
     that matter, every stock once a week) and `ibd_signals` (each buy-zone
     entry or breakout, graded later against SPY).

No LLM calls. Display and research only; nothing here feeds the discover
score yet.
"""

from __future__ import annotations

import argparse
import json
import time
from bisect import bisect_right
from datetime import date, timedelta

from dotenv import load_dotenv
from sqlalchemy import delete, func, select

from ..config import Settings
from ..data import fetch_cache, yf_gateway
from ..data.forecast_snapshots import tracked_tickers
from ..data.industry_groups import industry_map, sector_map
from ..data.quarterly_eps import MAX_AGE_DAYS, batch_eps
from ..data.universe_base import all_us_2b
from ..db.session import get_session
from ..db.tables import (
    IbdHistory,
    IbdMarket,
    IbdRating,
    IbdSector,
    IbdSignal,
    StockNews,
    column_names,
)
from ..discover.ibd_ratings import (
    FULL_SNAPSHOT_EVERY_DAYS,
    SECTOR_ETFS,
    SIGNAL_QUIET_DAYS,
    SIGNAL_STATUSES,
    history_rows,
    market_direction,
    new_signals,
    rate_universe,
    sector_direction,
)
from ..logging import get_logger

logger = get_logger(__name__)

# Enough for a year of RS plus a 65-week base and the run-up into it.
HISTORY_DAYS = 730
INDEXES = ("SPY", "QQQ")
# Sector ETFs for the sector-direction check, fetched with the indexes.
ETFS = tuple(sorted(set(SECTOR_ETFS.values())))


def universe(db: str, today: date) -> list[str]:
    """Every tradable US stock worth $2B+ (the bundled scan, not the
    quality-filtered frame: percentiles need the whole market), plus the
    tracked stocks so holdings are always rated."""
    return list(dict.fromkeys([*all_us_2b(), *tracked_tickers(db, today=today)]))


def run(settings: Settings, *, today: date) -> dict:
    db = settings.discover_db_path
    names = universe(db, today)
    start = today - timedelta(days=HISTORY_DAYS)
    bars = yf_gateway.daily_bars_many([*names, *INDEXES, *ETFS], start=start, what="ibd")
    indexes = {i: bars.pop(i, None) for i in INDEXES}
    etfs = {e: bars.pop(e, None) for e in ETFS}
    industries = industry_map()
    eps = batch_eps(names, refresh=fetch_cache.oldest("quarterly_eps", names))
    rows = rate_universe(bars, spy=indexes.get("SPY"), industries=industries, eps=eps)
    before = coverage_before(db)
    short = shortfall(coverage(rows), before)
    if short:
        # A failed load (2026-09-29: out of file handles) rates on bars
        # alone and would store inflated Composites. Try once more, then
        # fail with yesterday's ratings left in place.
        logger.warning(
            "IBD inputs look short (%s) — loading them again in %ds", short, RETRY_WAIT_SECONDS
        )
        time.sleep(RETRY_WAIT_SECONDS)
        industries = industry_map()
        eps = batch_eps(names)
        rows = rate_universe(bars, spy=indexes.get("SPY"), industries=industries, eps=eps)
        short = shortfall(coverage(rows), before)
        if short:
            raise RuntimeError(
                f"IBD inputs still short after a retry ({short}); nothing stored, "
                f"the ratings as of {before['as_of']} stay in place"
            )
        logger.info("IBD inputs recovered on the retry")
    market = market_direction({k: v for k, v in indexes.items() if v is not None})
    sectors = sector_direction(rows, sectors=sector_map(), etfs=etfs)
    spy = indexes.get("SPY")
    as_of = str(spy["date"][-1]) if spy is not None and spy.height else today.isoformat()

    fields = set(column_names(IbdRating))
    tracked = set(tracked_tickers(db, today=today))
    as_of_day = date.fromisoformat(as_of)
    with get_session(db) as session:
        last_full = session.scalars(select(func.max(IbdHistory.day)).where(IbdHistory.full)).one()
        since_signal = (as_of_day - timedelta(days=SIGNAL_QUIET_DAYS)).isoformat()
        recent = set(
            session.scalars(select(IbdSignal.ticker).where(IbdSignal.day >= since_signal)).all()
        )
    full = (
        last_full is None
        or (as_of_day - date.fromisoformat(last_full)).days >= FULL_SNAPSHOT_EVERY_DAYS
    )
    history = history_rows(rows, tracked=tracked, full=full)
    signals = new_signals(rows, recent=recent)
    with get_session(db) as session:
        session.execute(delete(IbdRating))
        for r in rows:
            session.add(IbdRating(as_of=as_of, **{k: v for k, v in r.items() if k in fields}))
        hist_fields = set(column_names(IbdHistory)) - {"day"}
        for r in history:
            session.merge(IbdHistory(day=as_of, **{k: v for k, v in r.items() if k in hist_fields}))
        for r in signals:
            session.merge(
                IbdSignal(
                    ticker=r["ticker"],
                    day=as_of,
                    status=r["base_status"],
                    price=r["price"],
                    pivot=r["pivot"],
                    composite=r["composite"],
                    rs_rating=r["rs_rating"],
                    eps_rating=r["eps_rating"],
                    base=r["base"],
                    industry=r["industry"],
                )
            )
        for sec in sectors:
            session.merge(_sector_row(as_of, sec, backfilled=False))
        session.merge(
            IbdMarket(
                day=as_of,
                status=market["status"],
                detail=market["detail"],
                indexes=json.dumps(market["indexes"]),
            )
        )
        session.commit()
    rated = sum(1 for r in rows if r["composite"] is not None)
    with_eps = sum(1 for r in rows if r["eps_rating"] is not None)
    in_zone = sum(1 for r in rows if r["base_status"] in ("buy zone", "breakout"))
    logger.info(
        "IBD ratings as of %s: %d of %d stocks rated (%d with EPS, %d industries); "
        "%d in a buy zone; market: %s (%s); history: %d rows%s, %d new signal(s)",
        as_of,
        rated,
        len(names),
        with_eps,
        len({r["industry"] for r in rows if r["industry"]}),
        in_zone,
        market["status"],
        market["detail"],
        len(history),
        " (weekly full snapshot)" if full else "",
        len(signals),
    )
    logger.info(
        "Sectors: %s",
        "; ".join(f"{x['sector']} {x['status']}" for x in sectors),
    )
    news_for(db, featured(rows, tracked), today=today)
    return {"as_of": as_of, "rated": rated, "market": market, "sectors": sectors}


# A run whose inputs cover less than this share of the last stored run's is
# a failed load, not a change in the market.
MIN_COVERAGE = {"rated": 0.8, "with_eps": 0.5, "with_industry": 0.5}
RETRY_WAIT_SECONDS = 120


def coverage(rows: list[dict]) -> dict[str, int]:
    """How many rows came out rated, with an EPS rating, with an industry."""
    return {
        "rated": sum(1 for r in rows if r.get("composite") is not None),
        "with_eps": sum(1 for r in rows if r.get("eps_rating") is not None),
        "with_industry": sum(1 for r in rows if r.get("industry")),
    }


def coverage_before(db: str) -> dict:
    """`coverage` of the ratings stored now (the last good run), with their
    as-of day; zeros on the first run."""
    with get_session(db) as session:
        stored = session.execute(
            select(IbdRating.as_of, IbdRating.composite, IbdRating.eps_rating, IbdRating.industry)
        ).all()
    return {
        "as_of": stored[0][0] if stored else None,
        **coverage([{"composite": c, "eps_rating": e, "industry": i} for _, c, e, i in stored]),
    }


def shortfall(now: dict[str, int], before: dict) -> str:
    """What fell below MIN_COVERAGE of the last run, e.g. "with_eps 0 vs
    1448"; "" when nothing did."""
    return ", ".join(
        f"{k} {now[k]} vs {before[k]}"
        for k, share in MIN_COVERAGE.items()
        if before.get(k) and now[k] < share * before[k]
    )


# News is fetched for the stocks the dashboard features, not all 1,900.
NEWS_TOP = 30


def featured(rows: list[dict], tracked: set[str]) -> list[str]:
    """Holdings and tracked picks, the top NEWS_TOP by Composite, and any
    Composite 70+ stock at or within 5% below a buy point."""
    ranked = sorted((r for r in rows if r["composite"] is not None), key=lambda r: -r["composite"])
    near = [
        r["ticker"]
        for r in ranked
        if r["composite"] >= 70
        and (
            r["base_status"] in SIGNAL_STATUSES
            or (r["base_status"] == "below pivot" and (r["vs_pivot"] or -1) >= -0.05)
        )
    ]
    rated = {r["ticker"] for r in rows}
    return list(
        dict.fromkeys(
            [*(t for t in tracked if t in rated), *(r["ticker"] for r in ranked[:NEWS_TOP]), *near]
        )
    )


def news_for(db: str, tickers: list[str], *, today: date) -> int:
    """Refresh the dashboard's stock_news for `tickers`; returns how many
    stocks have news. Never fails the morning job."""
    from ..data.ticker_news import dashboard_news

    names = {
        t: (e.get("value") or {}).get("name")
        for t, e in fetch_cache.entries("fundamentals").items()
    }
    try:
        news = dashboard_news(tickers, names)
    except Exception as e:  # noqa: BLE001 — the ratings are stored already
        logger.warning("Dashboard news failed (%s)", e)
        return 0
    if not news:
        return 0
    with get_session(db) as session:
        session.execute(delete(StockNews))
        for t, items in news.items():
            for i, item in enumerate(items, start=1):
                session.add(StockNews(ticker=t, rank=i, fetched=today.isoformat(), **item))
        session.commit()
    return sum(1 for v in news.values() if v)


def _sector_row(day: str, sec: dict, *, backfilled: bool) -> IbdSector:
    fields = set(column_names(IbdSector)) - {"day", "reasons", "backfilled"}
    return IbdSector(
        day=day,
        reasons="; ".join(sec["reasons"]),
        backfilled=backfilled,
        **{k: v for k, v in sec.items() if k in fields},
    )


# Leaders older than this are not fed to discover: the morning job stopped.
LEADERS_MAX_AGE_DAYS = 5


def top_leaders(db: str, n: int, *, today: date) -> tuple[str, ...]:
    """The top `n` by Composite plus every Composite 90+ stock in a buy zone
    or breaking out, from this morning's ratings; () when they are stale."""
    if n <= 0:
        return ()
    with get_session(db) as session:
        rows = session.execute(
            select(IbdRating.ticker, IbdRating.composite, IbdRating.base_status, IbdRating.as_of)
        ).all()
    if not rows or not rows[0][3]:
        return ()
    as_of = date.fromisoformat(rows[0][3])
    if (today - as_of).days > LEADERS_MAX_AGE_DAYS:
        logger.warning("IBD ratings are from %s — not feeding leaders to discover", as_of)
        return ()
    ranked = sorted(
        ((t, comp, status) for t, comp, status, _ in rows if comp is not None),
        key=lambda r: -r[1],
    )
    top = [t for t, _, _ in ranked[:n]]
    zone = [t for t, comp, status in ranked if comp >= 90 and status in SIGNAL_STATUSES]
    return tuple(dict.fromkeys(top + zone))


# A year of backfill needs a year of RS history before its first day, plus
# the 65-week base and its run-up: about three years of bars.
BACKFILL_HISTORY_DAYS = 1150


def _eps_as_of(facts: dict[str, list], dates: list[date]):
    """{day: {ticker: growth}} for each day, recomputed for a company only
    when it has filed something new since the previous day."""
    from ..data.quarterly_eps import growth_as_of

    filed = {
        t: sorted(date.fromisoformat(str(f["filed"])[:10]) for f in fs if f.get("filed"))
        for t, fs in facts.items()
    }
    last: dict[str, tuple[int, dict | None]] = {}
    for day in dates:
        out: dict[str, dict] = {}
        for t, fs in facts.items():
            known = bisect_right(filed[t], day)
            seen = last.get(t)
            if seen is None or seen[0] != known:
                seen = (known, growth_as_of(fs, day) if known else None)
                last[t] = seen
            g = seen[1]
            # The same answer goes stale with time, as it does live.
            if g and (day - date.fromisoformat(g["latest_quarter"])).days <= MAX_AGE_DAYS:
                out[t] = g
        yield day, out


def backfill(settings: Settings, *, sessions: int, today: date) -> dict:
    """Rebuild `sessions` trading days of ibd_history, ibd_signals and
    ibd_market from stored bars and SEC filings, as each day would have
    rated them: bars up to that close, EPS filed by that day. Live rows are
    never overwritten. Today's universe and industries are used throughout,
    so the rebuilt record leans toward survivors (names that fell below $2B
    or delisted are missing): read it as indicative."""
    from ..data.quarterly_eps import fetch_facts

    db = settings.discover_db_path
    names = universe(db, today)
    start = today - timedelta(days=BACKFILL_HISTORY_DAYS)
    bars = yf_gateway.daily_bars_many([*names, *INDEXES, *ETFS], start=start, what="ibd.backfill")
    indexes = {i: bars.pop(i, None) for i in INDEXES}
    etfs = {e: bars.pop(e, None) for e in ETFS}
    sectors_of = sector_map()
    spy = indexes.get("SPY")
    if spy is None:
        raise RuntimeError("no SPY bars to backfill against")
    industries = industry_map()
    facts = {t: f for t, f in yf_gateway.map_symbols(fetch_facts, names, workers=3) if f}
    logger.info("Backfill: %d stocks with bars, %d with SEC EPS facts", len(bars), len(facts))

    with get_session(db) as session:
        live_days = set(session.scalars(select(IbdHistory.day).where(~IbdHistory.backfilled)).all())
    calendar = spy["date"].to_list()
    days = [d for d in calendar[-sessions - 1 : -1] if d.isoformat() not in live_days]
    tracked = set(tracked_tickers(db, today=today))
    dates = {t: f["date"].to_list() for t, f in bars.items()}
    index_dates = {k: v["date"].to_list() for k, v in indexes.items() if v is not None}
    etf_dates = {k: v["date"].to_list() for k, v in etfs.items() if v is not None}

    def upto(frame, frame_dates, day):
        return frame.head(bisect_right(frame_dates, day))

    last_signal: dict[str, date] = {}
    last_full: date | None = None
    stored_days = signals_logged = rows_written = 0
    for day, eps in _eps_as_of(facts, days):
        rows = rate_universe(
            {t: upto(f, dates[t], day) for t, f in bars.items()},
            spy=upto(spy, index_dates["SPY"], day),
            industries=industries,
            eps=eps,
        )
        market = market_direction(
            {k: upto(v, index_dates[k], day) for k, v in indexes.items() if v is not None}
        )
        sectors = sector_direction(
            rows,
            sectors=sectors_of,
            etfs={k: upto(v, etf_dates[k], day) for k, v in etfs.items() if v is not None},
        )
        full = last_full is None or (day - last_full).days >= FULL_SNAPSHOT_EVERY_DAYS
        if full:
            last_full = day
        recent = {t for t, d in last_signal.items() if (day - d).days < SIGNAL_QUIET_DAYS}
        signals = new_signals(rows, recent=recent)
        history = history_rows(rows, tracked=tracked, full=full)
        iso = day.isoformat()
        hist_fields = set(column_names(IbdHistory)) - {"day", "backfilled"}
        with get_session(db) as session:
            for r in history:
                session.merge(
                    IbdHistory(
                        day=iso,
                        backfilled=True,
                        **{k: v for k, v in r.items() if k in hist_fields},
                    )
                )
            for r in signals:
                last_signal[r["ticker"]] = day
                session.merge(
                    IbdSignal(
                        ticker=r["ticker"],
                        day=iso,
                        status=r["base_status"],
                        price=r["price"],
                        pivot=r["pivot"],
                        composite=r["composite"],
                        rs_rating=r["rs_rating"],
                        eps_rating=r["eps_rating"],
                        base=r["base"],
                        industry=r["industry"],
                        backfilled=True,
                    )
                )
            for sec in sectors:
                seen = session.get(IbdSector, (iso, sec["sector"]))
                if seen is None or seen.backfilled:  # a live row is never replaced
                    session.merge(_sector_row(iso, sec, backfilled=True))
            if session.get(IbdMarket, iso) is None:
                session.add(
                    IbdMarket(
                        day=iso,
                        status=market["status"],
                        detail=market["detail"],
                        indexes=json.dumps(market["indexes"]),
                    )
                )
            session.commit()
        stored_days += 1
        signals_logged += len(signals)
        rows_written += len(history)
        if stored_days % 25 == 0:
            logger.info("Backfill: %d of %d days (%s)", stored_days, len(days), iso)
    logger.info(
        "Backfill done: %d days, %d history rows, %d signals (live days kept: %d)",
        stored_days,
        rows_written,
        signals_logged,
        len(live_days),
    )
    return {"days": stored_days, "rows": rows_written, "signals": signals_logged}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ibd-ratings", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--backfill",
        type=int,
        metavar="SESSIONS",
        help="rebuild this many past trading days of history from stored bars instead",
    )
    args = parser.parse_args(argv)
    load_dotenv()
    yf_gateway.reload_from_env()
    settings = Settings.from_env()
    if args.backfill:
        backfill(settings, sessions=args.backfill, today=date.today())
    else:
        run(settings, today=date.today())
    yf_gateway.log_stats("ibd-ratings")


if __name__ == "__main__":
    main()
