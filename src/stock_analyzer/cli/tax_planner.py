"""`tax-planner` — December: realized gains so far, losses to harvest,
gains to leave alone until they turn long-term. Emailed; no LLM calls.

Cron (scripts/run_tax_planner.sh) fires on December 1-7; the command acts
only on December's first trading day. `--force` runs it any day;
`--print` prints the HTML instead of emailing.
"""

from __future__ import annotations

from datetime import date, timedelta

from dotenv import load_dotenv

from ..config import Settings
from ..data import finnhub, yf_gateway
from ..logging import get_logger

logger = get_logger(__name__)


def first_trading_day_of_december(year: int) -> date:
    from ..reporting.quarterly import market_closed

    d = date(year, 12, 1)
    while market_closed(d):
        d += timedelta(days=1)
    return d


def build_plan(settings: Settings, today: date) -> tuple[str, str]:
    from ..data.brokerage import (
        fetch_account_meta,
        fetch_covered_call_obligations,
        fetch_portfolio_holdings,
    )
    from ..data.transactions import (
        fetch_activities_by_account,
        fetch_transaction_history,
        to_tax_payloads,
    )
    from ..discover.reinvest import sector_peers
    from ..discover.tax_harvest import find_harvest_candidates, harvest_report_data
    from ..discover.tax_planner import (
        gains_turning_long_term,
        last_trading_day_of_year,
        plan_summary,
        realized_this_year,
    )
    from ..reporting.tax_plan import render_tax_plan_html
    from .rebalance import _build_position_splits

    holdings = fetch_portfolio_holdings()
    meta = fetch_account_meta()
    taxable = sorted(name for name, m in meta.items() if m.get("tax_status") == "taxable")
    activities = fetch_activities_by_account(db_path=settings.discover_db_path)
    realized = {
        acct: realized_this_year(activities.get(acct, []), year=today.year) for acct in taxable
    }
    splits = _build_position_splits(holdings, meta)
    prices = {h["ticker"]: h.get("price") for items in holdings.values() for h in items}
    tax_lots = to_tax_payloads(fetch_transaction_history(db_path=settings.discover_db_path))
    harvest = harvest_report_data(
        find_harvest_candidates(
            splits,
            prices,
            tax_lots,
            sector_peers(settings.discover_db_path, list(splits), held=set(splits)),
            min_loss_usd=settings.harvest_min_loss_usd,
            min_loss_pct=settings.harvest_min_loss_pct,
            covered_calls=fetch_covered_call_obligations(),
        )
    )
    soon = gains_turning_long_term(tax_lots, prices, set(taxable), today=today)
    summary = plan_summary(realized, harvest)
    body = render_tax_plan_html(
        year=today.year,
        taxable_accounts=taxable,
        realized=realized,
        summary=summary,
        harvest=harvest,
        soon=soon,
        last_day=last_trading_day_of_year(today.year),
    )
    subject = (
        f"Tax planner {today.year}: net realized ~${summary['net_gain']:,.0f}, "
        f"${summary['harvestable_loss']:,.0f} of losses to harvest"
    )
    return subject, body


def main(argv: list[str] | None = None) -> None:
    import argparse

    from ..market_time import use_market_timezone

    use_market_timezone()
    parser = argparse.ArgumentParser(prog="tax-planner", description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="run on any day")
    parser.add_argument(
        "--print", dest="print_only", action="store_true", help="print the HTML instead of emailing"
    )
    args = parser.parse_args(argv)

    load_dotenv()
    yf_gateway.reload_from_env()
    finnhub.reload_from_env()
    settings = Settings.from_env()

    today = date.today()
    if not args.force and today != first_trading_day_of_december(today.year):
        logger.info(
            "Not December's first trading day (%s) — nothing to do",
            first_trading_day_of_december(today.year),
        )
        return
    subject, body = build_plan(settings, today)
    if args.print_only or not settings.email_to:
        print(body)
        return
    from ..reporting.smtp import SmtpServer

    SmtpServer().send_email(settings.email_to, subject, body, content_type="html")
    logger.info("Tax planner emailed: %s", subject)


if __name__ == "__main__":
    main()
