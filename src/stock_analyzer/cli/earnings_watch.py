"""`earnings-watch` — the nightly earnings-standout check, forecast snapshot
and insider-buying check.

Records the day's clear beats across the US market, measures how the
market took them, and confirms the ones analysts then revised up
(discover/earnings_standouts.py). Standouts show up in the next daily
email and in the next discover run's universe. Then stores today's
analyst forecasts for every tracked stock (data/forecast_snapshots.py),
the open-market insider purchases at every S&P 500 and tracked company
(data/insider_buying.py), and any new quarterly 13F from the tracked hedge
funds (data/hedge_funds_13f.py). Last, it rescans the >= $2B market with
the quality rules (data/universe_scan.py, 2 requests) and pre-fetches fundamentals and EPS
revisions for the names the next discover run will screen (same trend gate
and cap) that are new, have just reported, or are the oldest fifth in
data/fetch_cache.py, so the morning runs make almost no such requests. No LLM
calls, no email. A part that
fails doesn't stop the others, and the job exits non-zero so the cron
wrapper sends an alert.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

from dotenv import load_dotenv
from sqlalchemy import text

from ..config import Settings
from ..data import bar_store, fetch_cache, finnhub, hedge_funds_13f, insider_buying, yf_gateway
from ..data.analyst_actions import fetch_analyst_actions
from ..data.backlog import batch_rpo
from ..data.earnings_history import fetch_track_record
from ..data.eps_revisions import batch_eps_revisions, fetch_estimate_change
from ..data.forecast_snapshots import record_snapshots, tracked_tickers
from ..data.fundamentals import batch_fundamentals
from ..data.sec_edgar import load_ticker_cik_map
from ..data.technical_indicators import batch_technicals
from ..data.universe_base import load_base_universe, refresh_us_2b, sp500
from ..db.session import exec_sql, get_session
from ..discover.earnings_standouts import recent_standouts, watch
from ..discover.screen import prescreen
from ..logging import get_logger

logger = get_logger(__name__)


def _past_picks(db_path: str) -> set[str]:
    with get_session(db_path) as session:
        return {t for (t,) in exec_sql(session, text("SELECT DISTINCT ticker FROM picks")).all()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list", action="store_true", help="print standouts from the last 30 days and exit"
    )
    args = parser.parse_args()
    load_dotenv()
    yf_gateway.reload_from_env()
    finnhub.reload_from_env()
    settings = Settings.from_env()
    db = settings.discover_db_path

    if args.list:
        for s in recent_standouts(db, days=30):
            print(s)
        return
    failed = []
    try:
        _watch(db)
    except Exception:
        logger.exception("Earnings watch failed")
        failed.append("earnings watch")
    try:
        today = date.today()
        record_snapshots(db, today=today, tickers=tracked_tickers(db, today=today))
    except Exception:
        logger.exception("Forecast snapshot failed")
        failed.append("forecast snapshot")
    try:
        today = date.today()
        # The S&P 500 plus tracked stocks (~10 minutes at Finnhub's free
        # rate); the full >= $2B universe would be ~40.
        watched = list(dict.fromkeys([*sp500(), *tracked_tickers(db, today=today)]))
        insider_buying.watch(db, today=today, tickers=watched)
    except Exception:
        logger.exception("Insider-buying check failed")
        failed.append("insider buying")
    try:
        hedge_funds_13f.sync(db)
    except Exception:
        logger.exception("13F hedge-fund sync failed")
        failed.append("hedge-fund 13F")
    try:
        refresh_us_2b()  # the quality rules move with each earnings season
    except Exception:
        logger.exception("Universe rescan failed — keeping the last one")
        failed.append("universe rescan")
    try:
        warm_screen_cache(db, settings)
    except Exception:
        logger.exception("Screen cache warm-up failed")
        failed.append("screen cache warm-up")
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


def _watch(db: str) -> None:
    watch(
        db,
        today=date.today(),
        last_final=bar_store.last_final_close(datetime.now().astimezone()).date(),
        calendar=finnhub.fetch_earnings_calendar,
        closes=lambda symbols, start: yf_gateway.daily_closes(symbols, start, "earnings_watch"),
        estimate_change=fetch_estimate_change,
        track_record=fetch_track_record,
        analyst_actions=fetch_analyst_actions,
        picks=_past_picks(db),
        listed=set(load_ticker_cik_map()),
    )


def warm_screen_cache(db: str, settings: Settings) -> int:
    """Keep the fetch cache current for the names the next discover run
    will screen. Returns how many names.

    The frame is the base universe plus the tracked stocks (recent picks,
    views, standouts — the holdings among them), and the tracked ones pass
    the gate as the user's names do in discover. Bars are on disk, so the
    technicals cost next to nothing; a name that differs by morning is
    simply fetched then."""
    tracked = tracked_tickers(db, today=date.today())
    tickers = list(dict.fromkeys([*load_base_universe(), *tracked]))
    technicals = batch_technicals(tickers)
    names, _, _ = prescreen(
        tickers,
        technicals,
        gate=settings.discover_trend_gate,
        cap=settings.discover_max_screen_candidates,
        always=set(tracked),
    )
    # Names never fetched, or whose company has reported since, are
    # fetched anyway; on top of that the oldest fifth is renewed, so each
    # answer is at most a week old and no night refetches everything.
    batch_fundamentals(names, refresh=fetch_cache.oldest("fundamentals", names))
    batch_eps_revisions(names, refresh=fetch_cache.oldest("eps_revisions", names))
    # Contracted books are scored too; free SEC requests, same rotation.
    batch_rpo(names, refresh=fetch_cache.oldest("contracted_book", names))
    logger.info("Screen cache warmed: %d names", len(names))
    return len(names)


if __name__ == "__main__":
    main()
