"""`quarterly-review` — first trading day of each quarter: how last
quarter's advice worked out, plus today's portfolio health, emailed.
No LLM calls; brokerage positions and yfinance prices only.

Cron (scripts/run_quarterly_review.sh) fires on the first seven days of
Jan/Apr/Jul/Oct; the command itself exits unless today is the quarter's
first trading day (weekends, New Year's, Good Friday and July 4th
skipped). `--force` runs it any day; `--print` prints the HTML instead of
emailing.
"""

from __future__ import annotations

from datetime import date

from dotenv import load_dotenv

from ..config import Settings
from ..data import finnhub, yf_gateway
from ..logging import get_logger

logger = get_logger(__name__)


def build_review(settings: Settings, today: date) -> tuple[str, str]:
    """(subject, HTML) for the quarter before `today`."""
    from ..data.brokerage import fetch_portfolio_holdings
    from ..reporting.health import aggregate_positions, render_decisions_html, render_health_html
    from ..reporting.quarterly import (
        collect_suggestions,
        grade_suggestions,
        previous_quarter,
        render_quarterly_html,
        summarize,
    )
    from .portfolio import portfolio_health

    label, start, end = previous_quarter(today)
    holdings = fetch_portfolio_holdings()
    units_now = {t: p["units"] for t, p in aggregate_positions(holdings).items()}
    graded = grade_suggestions(
        collect_suggestions(settings.discover_db_path, start, end),
        today=today,
        units_now=units_now,
    )
    summary = summarize(graded)
    health = portfolio_health(settings, holdings)
    health_html = (
        "<h2>Where the portfolio stands today</h2>"
        + render_decisions_html(health)
        + render_health_html(health)
        if health is not None
        else ""
    )
    body = render_quarterly_html(
        label=label, start=start, end=end, graded=graded, summary=summary, health_html=health_html
    )
    scored = [g for g in graded if g["edge_pct"] is not None]
    good = sum(1 for g in scored if g["edge_pct"] >= 0)
    tally = f"{good} of {len(scored)} calls worked out" if scored else "no graded calls"
    return f"Quarterly review — {label}: {tally}", body


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="quarterly-review", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="run even if today isn't the quarter's first trading day",
    )
    parser.add_argument(
        "--print", dest="print_only", action="store_true", help="print the HTML instead of emailing"
    )
    args = parser.parse_args(argv)

    load_dotenv()
    yf_gateway.reload_from_env()
    finnhub.reload_from_env()
    settings = Settings.from_env()

    from ..reporting.quarterly import first_trading_day, is_first_trading_day_of_quarter, quarter_of

    today = date.today()
    if not args.force and not is_first_trading_day_of_quarter(today):
        logger.info(
            "Not the quarter's first trading day (that is %s) — nothing to do",
            first_trading_day(today.year, quarter_of(today)),
        )
        return

    subject, body = build_review(settings, today)
    if args.print_only or not settings.email_to:
        print(body)
        return
    from ..reporting.smtp import SmtpServer

    SmtpServer().send_email(settings.email_to, subject, body, content_type="html")
    logger.info("Quarterly review emailed: %s", subject)


if __name__ == "__main__":
    main()
