"""Insider + political + hedge fund trade synthesis agent."""

from __future__ import annotations

import re
from datetime import date

from ..logging import get_logger
from ..openrouter import helper_agent
from ..providers import HelperProvider, Provider

logger = get_logger(__name__)

INSIDER_INSTRUCTIONS = """\
You are a financial intelligence analyst. The user provides three lists:
1. Recent congressional trade coverage (politicians on a high-profile watchlist)
2. Recent insider trade coverage (corporate executives, Form 4 filings)
3. Recent billionaire-investor coverage — only managers with documented
   significant-profit track records (Buffett, Icahn, Ackman, Tepper, Burry,
   Druckenmiller, Klarman, Loeb, Singer, Einhorn, Griffin, Cohen, Englander)

DO NOT make tool calls. Use ONLY the data provided. Never invent values.
The data is article-level — extract concrete trades (politician / insider /
fund, ticker, buy/sell or new-position/exit, approx value or size, date)
from the snippets when stated.

Output format (plain text only, no markdown headings, no bold):

=== INSIDER, POLITICAL & BILLIONAIRE TRADING — {today} ===

Notable Congressional Trades:
List up to 5 most material trades. Each as one line:
- <Politician> (<Party>): <BUY/SELL> <TICKER> (<value range or size>) — <date if known>
If a snippet only describes a theme (no concrete trade), summarize the theme in one line.

Recent Insider Activity:
List up to 5 notable insider trades. Each as one line:
- <Executive name + role> at <COMPANY/TICKER>: <BUY/SELL> (<size or value>) — <date if known>

Top Billionaire Investor Moves:
List up to 5 notable moves, ONLY from the watchlist managers above. Skip items that
mention non-watchlist funds or generic "hedge fund" coverage with no named billionaire.
Each as one line:
- <Manager (Fund)>: <NEW POSITION/ADD/TRIM/EXIT/BUY/SELL> <TICKER> (<size, % of portfolio, or value>) — <date or filing period if known>
If a snippet only describes a theme or thesis (no concrete trade), summarize the theme in one line.

Tickers to Watch:
Identify 3-5 tickers that appear most active across the three sources, with one-line rationale each.
Flag tickers where multiple source types converge (e.g. politician + insider + fund) — those are the highest-signal entries:
- <TICKER>: <why it stands out (note convergence across sources)>

CRITICAL:
- Only use facts present in the snippets. If unsure, omit.
- Be terse. No filler. No closing remarks.
- Begin reply with the "===" header line.\
"""


# A ticker the report names: "(DKS)" or a "- DKS:" line.
_TICKER_RE = re.compile(r"\(([A-Z]{1,5}(?:\.[A-Z])?)\)|^\s*-\s*([A-Z]{1,5}(?:\.[A-Z])?):", re.M)
_TITLE_SKIP = {"THE", "A", "AN"}
# Parenthesised words that are roles or acronyms, not tickers ("(CFO)").
_NOT_TICKERS = {"CEO", "CFO", "COO", "CTO", "EVP", "SVP", "VP", "IPO", "ETF", "SEC", "AI", "US"}


def ungrounded_tickers(prompt: str, report: str) -> list[str]:
    """Tickers the report names that the source items don't support: not
    in the items as a symbol, and the company's name (its first word, as
    the SEC lists it) isn't there either. An open model that invents a
    ticker — or reads input a host altered — fails this, and the report is
    written by the fallback model instead."""
    from ..data.sec_edgar import load_ticker_titles

    def plain(text: str) -> str:  # "Dick’s" / "DICK'S" -> "dicks"
        return re.sub(r"[^a-z0-9]+", " ", re.sub(r"['’]", "", text.lower()))

    titles = load_ticker_titles()
    source = f" {plain(prompt)} "
    bad = []
    for m in _TICKER_RE.finditer(report):
        ticker = m.group(1) or m.group(2)
        if ticker in _NOT_TICKERS:
            continue
        if re.search(rf"\b{re.escape(ticker)}\b", prompt):
            continue
        words = [w for w in plain(titles.get(ticker, "")).split() if w.upper() not in _TITLE_SKIP]
        if words and f" {words[0]} " in source:
            continue
        if ticker not in bad:
            bad.append(ticker)
    return [f"{t} is not in the source items" for t in bad]


class InsiderAgent:
    def __init__(
        self,
        provider: HelperProvider,
        model: str,
        *,
        fallback: tuple[Provider, str] | None = None,
    ):
        self.agent = helper_agent(
            "Insider Analyst",
            provider,
            model,
            # Dated when the agent is built, not when this module is imported:
            # the import runs before the CLI switches to market time.
            INSIDER_INSTRUCTIONS.format(today=date.today().strftime("%b %d, %Y")),
            fallback=fallback,
            validate=ungrounded_tickers,
        )

    def run(
        self,
        political_items: list[dict],
        insider_items: list[dict],
        hedge_fund_items: list[dict],
    ) -> str:
        if not political_items and not insider_items and not hedge_fund_items:
            return "No recent insider, political, or hedge fund trade data available."

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

        hedge_fund_block = (
            "\n".join(
                f"- [{i}] ({', '.join(h.get('funds', []))}) "
                f"{h['title']} — {h.get('snippet', '')[:300]} ({h['link']})"
                for i, h in enumerate(hedge_fund_items)
            )
            or "(none)"
        )

        prompt = (
            "Congressional trade coverage:\n"
            f"{political_block}\n\n"
            "Insider trade coverage:\n"
            f"{insider_block}\n\n"
            "Hedge fund trade coverage:\n"
            f"{hedge_fund_block}"
        )
        logger.info(
            "Synthesizing insider report (%d political, %d insider, %d hedge fund items)",
            len(political_items),
            len(insider_items),
            len(hedge_fund_items),
        )
        return self.agent.run(prompt).content
