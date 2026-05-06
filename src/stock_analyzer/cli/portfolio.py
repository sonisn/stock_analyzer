"""Portfolio analysis pipeline: SnapTrade holdings → analysis → email."""
from __future__ import annotations

from datetime import date

from dotenv import load_dotenv

from ..agents.portfolio import PortfolioAgent
from ..config import Settings
from ..data.brokerage import fetch_portfolio_holdings
from ..logging import get_logger
from ..reporting.html import format_html
from ..reporting.smtp import SmtpServer

# Caching is disabled. To re-enable for iterating on email format without
# re-running data fetch + LLM:
#   from ..cache import FileCache
#   CACHE = FileCache("last_analysis.txt")
# Then in run_analysis(): check CACHE.read() first, write CACHE.write(result)
# at the end. Settings already has `use_cached_analysis` as the toggle.

logger = get_logger(__name__)


def _build_agent() -> PortfolioAgent:
    return PortfolioAgent(
        "Portfolio Manager",
        "claude",
        "claude-sonnet-4-6",
        sentiment_provider="claude",
        sentiment_model="claude-haiku-4-5",
        ticker_provider="claude",
        ticker_model="claude-haiku-4-5",
        rerank_provider="claude",
        rerank_model="claude-haiku-4-5",
    )


def run_analysis(settings: Settings) -> str:
    holdings = fetch_portfolio_holdings()
    tickers = sorted(
        {h["ticker"] for items in holdings.values() for h in items if h.get("ticker")}
    )
    if not tickers:
        raise RuntimeError(
            "No tickers returned from SnapTrade — check connected accounts."
        )
    logger.info("Analyzing %d tickers: %s", len(tickers), ", ".join(tickers))

    agent = _build_agent()
    return agent.run_analysis(tickers, holdings=holdings)


def main() -> None:
    load_dotenv()
    settings = Settings.from_env()

    result = run_analysis(settings)
    if not settings.email_to:
        logger.error("EMAIL_TO not set; printing report instead of emailing")
        print(result)
        return

    subject = f"Portfolio Analysis - {date.today().strftime('%b-%d')}"
    SmtpServer().send_email(
        settings.email_to,
        subject,
        format_html(result, title=subject),
        content_type="html",
    )


if __name__ == "__main__":
    main()
