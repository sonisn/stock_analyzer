"""`plan-check` — are the holdings in the right accounts, and is the
portfolio on course for its goal? Emailed (or printed with `--print`).
No LLM calls: brokerage positions, the stored activity ledger and daily
bars from the bar store.

The same two sections also go into the quarterly review. Set the goal with
GOAL_TARGET_USD and GOAL_DATE (or GOAL_HORIZON_YEARS); see config.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median
from typing import Any

from dotenv import load_dotenv

from ..config import Settings
from ..data import finnhub, yf_gateway
from ..logging import get_logger

logger = get_logger(__name__)

# Monthly history the projection resamples from; the bar store keeps it.
HISTORY_YEARS = 15
# Deposits in, withdrawals out, payroll plan purchases. Transfers are left
# out: between two connected accounts they cancel, and one from outside
# (an ACAT) is a one-off, not money that keeps arriving.
_CONTRIBUTION_TYPES = {"CONTRIBUTION", "DEPOSIT", "WITHDRAWAL"}


@dataclass
class PlanCheck:
    asset_html: str
    goal_html: str
    headlines: list[str]


def _months_until(today: date, goal: date) -> int:
    return max((goal.year - today.year) * 12 + goal.month - today.month, 0)


def _account_kinds(holdings: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    from ..data.brokerage import classify_account_kind, fetch_account_meta

    meta = fetch_account_meta()
    return {
        account: (meta.get(account) or {}).get("kind") or classify_account_kind(None, account)
        for account in holdings
    }


def typical_monthly(flows: list[tuple[date, float]], today: date, *, months: int = 12) -> float:
    """Median of the monthly totals over the complete months since the
    first flow (at most `months`). A median, because the money that keeps
    arriving is the point: one $50,000 deposit or an IRA rollover in the
    window would otherwise read as $4,000 more every month for five years."""
    if not flows:
        return 0.0
    this_month = date(today.year, today.month, 1)
    first = max(min(d for d, _ in flows), _add_months(this_month, -months))
    totals: dict[tuple[int, int], float] = {}
    month = date(first.year, first.month, 1)
    while month < this_month:
        totals[(month.year, month.month)] = 0.0
        month = _add_months(month, 1)
    for day, amount in flows:
        key = (day.year, day.month)
        if key in totals:
            totals[key] += amount
    if not totals:
        return 0.0
    return max(float(median(totals.values())), 0.0)


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    return date(d.year + y, m + 1, 1)


def monthly_contribution(settings: Settings, today: date) -> tuple[float, str]:
    """(dollars a month, where the number came from)."""
    if settings.goal_monthly_contribution is not None:
        return settings.goal_monthly_contribution, "GOAL_MONTHLY_CONTRIBUTION"
    from ..data.transactions import fetch_cash_activity, implied_plan_flows

    db = settings.discover_db_path
    try:
        flows = [
            (f["date"], f["amount"])
            for f in fetch_cash_activity(days_back=400, db_path=db)["flows"]
            if f["type"] in _CONTRIBUTION_TYPES
        ]
        flows += [(f["date"], f["amount"]) for f in implied_plan_flows(days_back=400, db_path=db)]
    except Exception as e:  # noqa: BLE001 — a default beats no projection
        logger.warning("Could not read last year's contributions (%s) — assuming none", e)
        return 0.0, "contributions unreadable; assumed none"
    return (
        typical_monthly(flows, today),
        "last year's median month of deposits and plan purchases, so one-off lumps "
        "don't count; GOAL_MONTHLY_CONTRIBUTION overrides",
    )


def _start_value(settings: Settings, holdings_value: float) -> float:
    """The daily email's latest reconciled total, else today's holdings."""
    from ..db.repository import fetch_snapshots
    from ..db.session import get_session

    try:
        with get_session(settings.discover_db_path) as session:
            snaps = fetch_snapshots(session)
            total = float(snaps[-1].total) if snaps else None
        if total:
            return total
    except Exception as e:  # noqa: BLE001
        logger.warning("No stored portfolio total (%s) — using holdings value", e)
    return holdings_value


def build_plan_check(
    settings: Settings, holdings: dict[str, list[dict[str, Any]]], *, today: date
) -> PlanCheck:
    from ..data.brokerage import listed_tickers
    from ..data.options_income import summarize_contracts
    from ..data.reference import profiles
    from ..data.transactions import fetch_activities_by_account
    from ..discover import asset_location as al
    from ..discover import goal_projection as gp
    from ..discover.tax_lot_helper import long_term_rate, short_term_rate
    from ..reporting.health import aggregate_positions
    from ..reporting.plan_check import (
        asset_location_headline,
        goal_headline,
        render_asset_location_html,
        render_goal_html,
    )

    tickers, _ = listed_tickers(holdings)
    bars = yf_gateway.daily_bars_many(
        [*tickers, "SPY"],
        start=today - timedelta(days=round(HISTORY_YEARS * 365.25)),
        what="plan_check",
    )

    # --- asset location ---
    report = None
    try:
        info = profiles(tickers, settings.discover_db_path)
        facts = {
            t: al.TickerTaxFacts(
                dividend_yield=al.trailing_yield(bars.get(t), today=today),
                ordinary_dividends=al.ordinary_dividends(
                    (info.get(t) or {}).get("sector"), (info.get(t) or {}).get("industry")
                ),
            )
            for t in tickers
        }
        contracts = summarize_contracts(
            fetch_activities_by_account(db_path=settings.discover_db_path)
        )
        premium = al.option_premium_by_account(contracts, since=today - timedelta(days=365))
        report = al.analyze(
            {a: [h for h in rows if h.get("ticker") in tickers] for a, rows in holdings.items()},
            _account_kinds(holdings),
            facts,
            premium,
            long_term_rate=long_term_rate(),
            short_term_rate=short_term_rate(),
            min_drag_usd=settings.asset_location_min_drag_usd,
            max_breakeven_years=settings.asset_location_max_breakeven_years,
            today=today,
        )
    except Exception as e:  # noqa: BLE001 — one section never blocks the other
        logger.warning("Asset location failed (%s)", e)

    # --- goal projection ---
    projection = None
    contribution, source = 0.0, ""
    left_out: list[str] = []
    goal_date = settings.goal_date
    try:
        positions = aggregate_positions(holdings)
        # A symbol with no price history at all (a money-market fund like
        # SPAXX) is cash, not a stock that moves with the market.
        cash_like = {t for t in tickers if t not in bars or gp.is_cash_like(bars[t])}
        weights = {
            t: p["value"]
            for t, p in positions.items()
            if t in tickers and t not in cash_like and p["value"] > 0
        }
        left_out = sorted(cash_like)
        months = (
            _months_until(today, goal_date)
            if goal_date
            else round(settings.goal_horizon_years * 12)
        )
        contribution, source = monthly_contribution(settings, today)
        projection = gp.project(
            weights=weights,
            returns=gp.monthly_returns(bars),
            start_value=_start_value(settings, sum(weights.values())),
            months=months,
            monthly_contribution=contribution,
            expected_return=settings.goal_expected_return,
            target=settings.goal_target_usd,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Goal projection failed (%s)", e)

    return PlanCheck(
        asset_html=render_asset_location_html(report),
        goal_html=render_goal_html(
            projection, goal_date=goal_date, contribution_note=source, left_out=left_out
        ),
        headlines=[goal_headline(projection, goal_date), asset_location_headline(report)],
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="plan-check", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--print", dest="print_only", action="store_true", help="print the HTML instead of emailing"
    )
    args = parser.parse_args(argv)

    load_dotenv()
    yf_gateway.reload_from_env()
    finnhub.reload_from_env()
    settings = Settings.from_env()

    from ..data.brokerage import fetch_portfolio_holdings
    from ..reporting.html import _wrap_html

    today = date.today()
    check = build_plan_check(settings, fetch_portfolio_holdings(), today=today)
    yf_gateway.log_stats("plan-check")
    body = _wrap_html("Plan check", check.goal_html + check.asset_html)
    subject = "Plan check: " + " · ".join(check.headlines)
    if args.print_only or not settings.email_to:
        print(body)
        return
    from ..reporting.smtp import SmtpServer

    SmtpServer().send_email(settings.email_to, subject, body, content_type="html")
    logger.info("Plan check emailed: %s", subject)


if __name__ == "__main__":
    main()
