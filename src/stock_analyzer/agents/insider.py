"""Insider + political trade synthesis agent."""
from __future__ import annotations

from datetime import date

from ..llm import AgnoAgent, Provider
from ..logging import get_logger

logger = get_logger(__name__)

INSIDER_INSTRUCTIONS = f"""\
You are a financial intelligence analyst. The user provides two lists:
1. Recent congressional trade coverage (politicians on a high-profile watchlist)
2. Recent insider trade coverage (corporate executives)

DO NOT make tool calls. Use ONLY the data provided. Never invent values.
The data is article-level — extract concrete trades (politician/insider, ticker,
buy/sell, approx value, date) from the snippets when stated.

Output format (plain text only, no markdown headings, no bold):

=== INSIDER & POLITICAL TRADING — {date.today().strftime('%b %d, %Y')} ===

Notable Congressional Trades:
List up to 5 most material trades. Each as one line:
- <Politician> (<Party>): <BUY/SELL> <TICKER> (<value range or size>) — <date if known>
If a snippet only describes a theme (no concrete trade), summarize the theme in one line.

Recent Insider Activity:
List up to 5 notable insider trades. Each as one line:
- <Executive name + role> at <COMPANY/TICKER>: <BUY/SELL> (<size or value>) — <date if known>

Tickers to Watch:
Identify 3-5 tickers that appear most active across both sources, with one-line rationale each:
- <TICKER>: <why it stands out>

CRITICAL:
- Only use facts present in the snippets. If unsure, omit.
- Be terse. No filler. No closing remarks.
- Begin reply with the "===" header line.\
"""


class InsiderAgent:
    def __init__(self, provider: Provider, model: str):
        self.agent = AgnoAgent(
            "Insider Analyst",
            provider,
            model,
            instructions=INSIDER_INSTRUCTIONS,
        )

    def run(
        self,
        political_items: list[dict],
        insider_items: list[dict],
    ) -> str:
        if not political_items and not insider_items:
            return "No recent insider or political trade data available."

        political_block = (
            "\n".join(
                f"- [{i}] ({', '.join(p.get('politicians', []))}) "
                f"{p['title']} — {p.get('snippet', '')[:300]} ({p['link']})"
                for i, p in enumerate(political_items)
            )
            or "(none)"
        )

        insider_block = (
            "\n".join(
                f"- [{i}] {it['title']} — {it.get('snippet', '')[:300]} ({it['link']})"
                for i, it in enumerate(insider_items)
            )
            or "(none)"
        )

        prompt = (
            "Congressional trade coverage:\n"
            f"{political_block}\n\n"
            "Insider trade coverage:\n"
            f"{insider_block}"
        )
        logger.info(
            "Synthesizing insider report (%d political, %d insider items)",
            len(political_items),
            len(insider_items),
        )
        return self.agent.run(prompt).content
