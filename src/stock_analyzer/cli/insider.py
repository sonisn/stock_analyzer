"""Insider + political trade analysis pipeline."""
from __future__ import annotations

from datetime import date

from dotenv import load_dotenv

from ..agents.insider import InsiderAgent
from ..config import Settings
from ..data.insider import fetch_insider_trades
from ..data.political import fetch_political_trades
from ..logging import get_logger
from ..reporting.html import format_insider_html
from ..reporting.smtp import SmtpServer

# Caching is disabled. To re-enable for iterating on email format without
# re-running data fetch + LLM:
#   from ..cache import FileCache
#   CACHE = FileCache("last_insider_analysis.txt")
# Then check CACHE.read() before fetching, and CACHE.write(result) afterwards.

logger = get_logger(__name__)


def run_analysis(settings: Settings) -> str:
    political = fetch_political_trades(days=settings.insider_lookback_days)
    insider = fetch_insider_trades(days=settings.insider_lookback_days)

    agent = InsiderAgent("claude", "claude-haiku-4-5")
    return agent.run(political, insider)


def main() -> None:
    load_dotenv()
    settings = Settings.from_env()

    result = run_analysis(settings)
    if not settings.email_to:
        logger.error("EMAIL_TO not set; printing report instead of emailing")
        print(result)
        return

    subject = f"Insider & Political Trades - {date.today().strftime('%b-%d')}"
    SmtpServer().send_email(
        settings.email_to,
        subject,
        format_insider_html(result, title=subject),
        content_type="html",
    )


if __name__ == "__main__":
    main()
