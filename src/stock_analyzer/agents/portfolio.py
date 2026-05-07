"""Portfolio analysis agent: market sentiment + per-ticker synthesis."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from ..data.market_news import fetch_market_sentiment_news
from ..data.ticker import fetch_ticker_data
from ..llm import AgnoAgent, Provider
from ..logging import get_logger
from .news_reranker import NewsReranker

logger = get_logger(__name__)

# Per-ticker work runs in parallel: each ticker does ~6 yfinance HTTP calls
# plus 2 LLM calls. Cap concurrency to bound load on yfinance and the LLM
# provider while still cutting wall-clock time roughly proportionally to
# this value for portfolios above _TICKER_MAX_WORKERS holdings.
_TICKER_MAX_WORKERS = 5

SENTIMENT_INSTRUCTIONS = f"""\
You are a macro/market analyst. The user provides recent US-market news headlines
and snippets (as of {date.today()}). DO NOT make any tool calls. Use ONLY the
data provided; never invent news.

Synthesize what you see into ONE plain-text block covering:
- Major geopolitical or policy news affecting US equities
- Notable sector moves (tech, energy, financials, etc.)
- US economic data released this week or scheduled for today
  (CPI, jobs, GDP, FOMC, retail sales, PMI, etc.)

Output exactly this form, nothing else:

Social/Economic Sentiment:
<3-5 tight sentences combining the above>

CRITICAL:
- Plain text only. No markdown headings, no bold, no bullets.
- Begin reply with the literal text "Social/Economic Sentiment:".
- No preamble, no "Here is", no closing remarks.\
"""

TICKER_INSTRUCTIONS = """\
You are an equity analyst. The user provides ONE ticker's pre-fetched data as JSON.
DO NOT make any tool calls. Use ONLY the data in the JSON; never invent values.
For any field that is null/missing in the JSON, omit that line entirely.

Begin your reply with a line of 40 dashes, then a blank line, then the block.

Block format:

----------------------------------------

[TICKER] - [Company Name]
Your Position: <position.units> shares @ avg <position.avg_buy_price> | Unrealized: <position.unrealized_pl> (<position.pl_pct>)
Price:       <price> (<pct_today> today)
Market Cap:  <market_cap>
52W Range:   <range_52w>
P/E:         <pe>
Div Yield:   <dividend_yield>
Top News:    List items from `news` in the order given (already ranked by materiality), as: "- <title> (<link>)"
Analysts:    Buy/Hold/Sell counts from `analysts`, mean target <analyst_target>
Outlook:     2-3 sentences synthesizing fundamentals + trend + news. This is your only narrative section.
Earnings:    Last quarter EPS/Revenue (estimate vs actual) from earnings.history. Next earnings date with estimate from earnings.estimates.
Trend:       7days: <trend_7days>; 1mo: <trend_1mo>; 3mo: <trend_3mo>; 6mo: <trend_6mo>; 1yr: <trend_1yr>

CRITICAL:
- Copy values verbatim from the JSON. The only place you reason is "Outlook".
- Plain text, no markdown headings or bold.
- Begin reply with the dashes line — no introductions, no closing remarks.\
"""


class PortfolioAgent:
    def __init__(
        self,
        name: str,
        provider: Provider,
        model: str,
        *,
        ticker_provider: Provider | None = None,
        ticker_model: str | None = None,
        sentiment_provider: Provider | None = None,
        sentiment_model: str | None = None,
        rerank_provider: Provider | None = None,
        rerank_model: str | None = None,
    ):
        self.name = name
        self.model = model

        self.sentiment_agent = AgnoAgent(
            f"{name} (sentiment)",
            sentiment_provider or provider,
            sentiment_model or model,
            instructions=SENTIMENT_INSTRUCTIONS,
        )

        self.ticker_agent = AgnoAgent(
            f"{name} (ticker)",
            ticker_provider or provider,
            ticker_model or model,
            instructions=TICKER_INSTRUCTIONS,
        )

        self.news_reranker = NewsReranker(
            f"{name} (news rerank)",
            rerank_provider or ticker_provider or provider,
            rerank_model or ticker_model or model,
        )

        self._positions_by_ticker: dict[str, dict] = {}

    def run_analysis(
        self,
        stocks: list[str],
        *,
        holdings: dict[str, list[dict]] | None = None,
    ) -> str:
        self._positions_by_ticker = self._aggregate_positions(holdings or {})
        logger.info(
            "Running analysis for %d tickers (max_workers=%d)",
            len(stocks),
            _TICKER_MAX_WORKERS,
        )

        # Run sentiment in parallel with ticker fan-out: it's a separate
        # data + LLM round-trip that doesn't depend on ticker data, so the
        # whole pipeline can finish in roughly max(sentiment, slowest-batch)
        # rather than the sum.
        with ThreadPoolExecutor(
            max_workers=_TICKER_MAX_WORKERS + 1
        ) as ex:
            sentiment_future = ex.submit(self._run_sentiment)
            ticker_results = list(ex.map(self._run_ticker, stocks))
            sentiment = sentiment_future.result()

        return "\n\n".join([sentiment, *ticker_results])

    @staticmethod
    def _aggregate_positions(
        holdings: dict[str, list[dict]],
    ) -> dict[str, dict]:
        agg: dict[str, dict] = {}
        for items in holdings.values():
            for h in items:
                ticker = h.get("ticker")
                units = h.get("units") or 0
                avg = h.get("average_purchase_price") or 0
                if not ticker or not units:
                    continue
                cur = agg.setdefault(ticker, {"units": 0.0, "cost_basis": 0.0})
                cur["units"] += float(units)
                cur["cost_basis"] += float(units) * float(avg)

        out: dict[str, dict] = {}
        for ticker, v in agg.items():
            if v["units"]:
                out[ticker] = {
                    "units": v["units"],
                    "avg_buy_price": v["cost_basis"] / v["units"],
                }
        return out

    def _run_sentiment(self) -> str:
        items = fetch_market_sentiment_news()
        if not items:
            return "Social/Economic Sentiment: market news data unavailable."
        listing = "\n".join(
            f"- {it['title']}: {it.get('snippet', '')}" for it in items
        )
        prompt = f"Today's US market news:\n\n{listing}"
        logger.info("Synthesizing sentiment from %d items", len(items))
        return self.sentiment_agent.run(prompt).content

    def _run_ticker(self, ticker: str) -> str:
        data = fetch_ticker_data(ticker)
        if data.get("news"):
            data["news"] = self.news_reranker.rerank(
                data["news"], data["symbol"], data.get("name")
            )
        position = self._build_position_block(ticker, data.get("price"))
        if position:
            data["position"] = position
        prompt = (
            f"Ticker data:\n```json\n{json.dumps(data, default=str, indent=2)}\n```"
        )
        logger.info("Synthesizing block for %s", ticker)
        return self.ticker_agent.run(prompt).content

    def _build_position_block(
        self, ticker: str, current_price_str: str | None
    ) -> dict | None:
        pos = self._positions_by_ticker.get(ticker)
        if not pos:
            return None

        units = pos["units"]
        avg = pos["avg_buy_price"]
        block: dict = {
            "units": (
                f"{int(units)}" if units == int(units) else f"{units:.4f}"
            ),
            "avg_buy_price": f"${avg:,.2f}",
        }
        try:
            current = float((current_price_str or "").replace("$", "").replace(",", ""))
            pl_per_share = current - avg
            block["unrealized_pl"] = f"${pl_per_share * units:+,.2f}"
            block["pl_pct"] = f"{(pl_per_share / avg * 100):+.2f}%" if avg else None
        except (ValueError, AttributeError, TypeError):
            pass
        return block
