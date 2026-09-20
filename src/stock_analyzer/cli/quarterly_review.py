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


def performance_section(settings: Settings, *, start: date, today: date) -> str:
    """ "Your portfolio vs SPY" for last quarter, year to date and since the
    first snapshot. Never blocks the review."""
    from ..data.transactions import fetch_cash_activity, implied_plan_flows
    from ..db.repository import fetch_snapshots, snapshot_accounts
    from ..db.session import get_session
    from ..reporting.performance import (
        account_change_flows,
        performance_vs_spy,
        render_performance_html,
    )

    try:
        with get_session(settings.discover_db_path) as session:
            stored = fetch_snapshots(session)
            snaps = [(date.fromisoformat(s.day), s.total) for s in stored]
            breakdown = [(date.fromisoformat(s.day), snapshot_accounts(s)) for s in stored]
        if not snaps:
            return render_performance_html([], first_day=None)
        first = snaps[0][0]
        lookback = (today - first).days + 7
        activity = fetch_cash_activity(days_back=lookback, db_path=settings.discover_db_path)
        flows = [(f["date"], f["amount"]) for f in activity["flows"]]
        # Money that arrives without a deposit row: a payroll-funded plan's
        # purchases, and an account joining (or leaving) the feed.
        flows += [
            (f["date"], f["amount"])
            for f in implied_plan_flows(days_back=lookback, db_path=settings.discover_db_path)
        ]
        flows += account_change_flows(breakdown)
        rows = performance_vs_spy(
            snaps,
            flows,
            windows={
                "Last quarter": start,
                "Year to date": date(today.year, 1, 1),
                "Since tracking began": first,
            },
        )
        return render_performance_html(rows, first_day=first)
    except Exception as e:  # noqa: BLE001
        logger.warning("Portfolio-vs-SPY section failed (%s)", e)
        return ""


def options_section(settings: Settings, *, start: date, end: date, label: str) -> str:
    """Premium written last quarter, and any shares called away. Never
    blocks the review."""
    from ..data.options_income import fetch_option_income
    from ..reporting.quarterly import render_options_income_html

    try:
        income = fetch_option_income(start=start, end=end, db_path=settings.discover_db_path)
        return render_options_income_html(income, label=label)
    except Exception as e:  # noqa: BLE001
        logger.warning("Options-income section failed (%s)", e)
        return ""


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
        label=label,
        start=start,
        end=end,
        graded=graded,
        summary=summary,
        health_html=health_html,
        performance_html=performance_section(settings, start=start, today=today),
        options_html=options_section(settings, start=start, end=end, label=label),
    )
    scored = [g for g in graded if g["edge_pct"] is not None]
    good = sum(1 for g in scored if g["edge_pct"] >= 0)
    tally = f"{good} of {len(scored)} calls worked out" if scored else "no graded calls"
    return f"Quarterly review — {label}: {tally}", body


def main(argv: list[str] | None = None) -> None:
    from ..market_time import use_market_timezone

    use_market_timezone()
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
