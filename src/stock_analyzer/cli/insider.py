"""Insider + political + hedge fund trade analysis pipeline."""

from __future__ import annotations

from datetime import date

from dotenv import load_dotenv

from ..agents.insider import InsiderAgent
from ..config import Settings
from ..data import finnhub, yf_gateway
from ..data.form4 import fetch_form4_trades, form4_section
from ..data.hedge_funds import fetch_hedge_fund_trades
from ..data.insider import fetch_insider_trades
from ..data.political import fetch_political_trades
from ..logging import get_logger
from ..reporting.html import format_insider_html
from ..reporting.smtp import SmtpServer

logger = get_logger(__name__)

# Form 4s are due two business days after the trade; a week back from a
# Monday run always covers the previous one.
_FORM4_MIN_DAYS = 7


def _my_tickers(settings: Settings) -> set[str]:
    """Held and watched names. A brokerage outage costs the holdings, not
    the watchlist."""
    tickers = set(settings.discover_watchlist)
    try:
        from ..data.brokerage import fetch_portfolio_holdings

        for items in fetch_portfolio_holdings().values():
            tickers.update(h["ticker"] for h in items if h.get("ticker"))
    except Exception as e:  # noqa: BLE001
        logger.warning("Holdings unavailable for the Form 4 lookup (%s)", e)
    return tickers


def run_analysis(settings: Settings) -> str | None:
    """The report text, or None when every source came back empty."""
    days = settings.insider_lookback_days
    political = fetch_political_trades(days=days)
    insider = fetch_insider_trades(days=days)
    hedge_funds = fetch_hedge_fund_trades(days=days)
    form4_days = max(days, _FORM4_MIN_DAYS)
    form4 = fetch_form4_trades(_my_tickers(settings), days=form4_days)

    parts: list[str] = []
    if political or insider or hedge_funds:
        agent = InsiderAgent(
            settings.insider_provider or settings.llm_provider,
            settings.insider_model or settings.llm_model,
        )
        parts.append(agent.run(political, insider, hedge_funds))
    else:
        # The news fetchers swallow their own failures, so an exhausted
        # search quota used to arrive as a quiet "no data" email.
        logger.error(
            "No congressional, insider or billionaire news coverage came back — "
            "check the TAVILY_API_KEY quota"
        )
        if form4 is None:
            return None
        parts.append(
            f"=== INSIDER TRADING ON YOUR HOLDINGS — {date.today().strftime('%b %d, %Y')} ===\n\n"
            "Data Gaps:\n"
            "- No congressional, insider or billionaire news coverage came back: every "
            "search failed or found nothing (check the Tavily quota)."
        )
    if form4 is not None:
        parts.append(form4_section(form4, days=form4_days))
    return "\n\n".join(parts)


def main() -> None:
    load_dotenv()
    # Pacing knobs live in the environment, and these modules are
    # imported before `.env` is loaded — re-read them now.
    yf_gateway.reload_from_env()
    finnhub.reload_from_env()
    settings = Settings.from_env()

    result = run_analysis(settings)
    if result is None:
        logger.error("Every insider source failed — no email sent")
        raise SystemExit(1)
    if not settings.email_to:
        logger.error("EMAIL_TO not set; printing report instead of emailing")
        print(result)
        return

    subject = f"Insider, Political & Billionaire Trades - {date.today().strftime('%b-%d')}"
    SmtpServer().send_email(
        settings.email_to,
        subject,
        format_insider_html(result, title=subject),
        content_type="html",
    )


if __name__ == "__main__":
    main()
