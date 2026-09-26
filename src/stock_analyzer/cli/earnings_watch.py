"""`earnings-watch` — the nightly earnings-standout check and forecast snapshot.

Records the day's clear beats across the US market, measures how the
market took them, and confirms the ones analysts then revised up
(discover/earnings_standouts.py). Standouts show up in the next daily
email and in the next discover run's universe. Then stores today's
analyst forecasts for every tracked stock (data/forecast_snapshots.py).
No LLM calls, no email. Either half failing still runs the other, and
the job exits non-zero so the cron wrapper sends an alert.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

from dotenv import load_dotenv
from sqlalchemy import text

from ..config import Settings
from ..data import bar_store, finnhub, yf_gateway
from ..data.analyst_actions import fetch_analyst_actions
from ..data.earnings_history import fetch_track_record
from ..data.eps_revisions import fetch_estimate_change
from ..data.forecast_snapshots import record_snapshots, tracked_tickers
from ..data.sec_edgar import load_ticker_cik_map
from ..db.session import exec_sql, get_session
from ..discover.earnings_standouts import recent_standouts, watch
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


if __name__ == "__main__":
    main()
