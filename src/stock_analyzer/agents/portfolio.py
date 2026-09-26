"""Portfolio analysis agent: market sentiment + per-ticker synthesis."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

from ..data.market_news import fetch_market_sentiment_news
from ..data.ticker import fetch_ticker_data
from ..llm import AgnoAgent, Provider
from ..logging import get_logger
from ..serialization import dumps_prompt
from .news_reranker import NewsReranker
from .stock_views import is_equity

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

VIEW_INSTRUCTIONS = """\
You are an equity analyst writing for a LONG-TERM investor: every holding is
a 3-5 year investment. The user provides ONE ticker's pre-fetched data as JSON.
DO NOT make tool calls. Use ONLY the data in the JSON; never invent values.

Write the stock's long-term view: 2-3 sentences on the 3-5 year case —
business quality and growth runway, valuation against long-run earnings,
and whether the recent news or earnings change that case. Daily price
moves and short-term trend are noise unless they signal the business
changing. Never suggest trading around earnings, momentum or short-term
targets.

Reply with the 2-3 sentences only: plain text, no label, no heading, no
markdown, no preamble.\
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
        db_path: str | None = None,
        view_max_age_days: int = 7,
        view_move_pct: float = 8.0,
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
            instructions=VIEW_INSTRUCTIONS,
        )

        self.news_reranker = NewsReranker(
            f"{name} (news rerank)",
            rerank_provider or ticker_provider or provider,
            rerank_model or ticker_model or model,
        )

        self._positions_by_ticker: dict[str, dict] = {}
        # Stored views (agents/stock_views.py); None = always write fresh.
        self.db_path = db_path
        self.view_max_age_days = view_max_age_days
        self.view_move_pct = view_move_pct
        # Each ticker's fetched data, for checks that run after the analysis
        # (e.g. the post-earnings check) without fetching it again.
        self.ticker_data: dict[str, dict] = {}
        # Filled per phase in run_analysis.
        self.stored_views: dict[str, Any] = {}
        self.ranked_news: dict[str, list[dict]] = {}
        self.facts: dict[str, dict[str, str]] = {}
        self.views_written = 0
        self.views_reused = 0

    def run_analysis(
        self,
        stocks: list[str],
        *,
        holdings: dict[str, list[dict]] | None = None,
    ) -> str:
        self._positions_by_ticker = self._aggregate_positions(holdings or {})
        # Per-run state: each run starts from a clean slate, so a second
        # run in the same process can't show the first one's rankings.
        self.ticker_data, self.stored_views = {}, {}
        self.ranked_news, self.facts = {}, {}
        self.views_written = self.views_reused = 0
        logger.info(
            "Running analysis for %d tickers (max_workers=%d)",
            len(stocks),
            _TICKER_MAX_WORKERS,
        )

        # Run sentiment in parallel with the ticker work: it's a separate
        # data + LLM round-trip that doesn't depend on ticker data, so the
        # whole pipeline can finish in roughly max(sentiment, the rest)
        # rather than the sum.
        # Each piece is guarded: one stock's data or model failure must not
        # cost the whole email (it did on 2026-07-22, a single timeout).
        with ThreadPoolExecutor(max_workers=_TICKER_MAX_WORKERS + 1) as ex:
            sentiment_future = ex.submit(self._safe_sentiment)
            # 1. Data for every holding, no model calls.
            list(ex.map(self._safe_fetch, stocks))
            # 2. One rerank call for the whole portfolio, on the headlines
            #    that haven't been in an earlier email.
            self._rank_all_news(stocks)
            # 3. For holdings with nothing to read, what the company
            #    actually did — filings, revisions, Form 4s.
            self._collect_facts(stocks)
            # 4. Long-term views (mostly reused) and the blocks themselves.
            ticker_results = list(ex.map(self._safe_ticker, stocks))
            sentiment = sentiment_future.result()

        return "\n\n".join([sentiment, *ticker_results])

    def _safe_fetch(self, ticker: str) -> None:
        """Phase 1: fetch and stash one holding's data and stored view."""
        from .stock_views import load_view

        try:
            data = fetch_ticker_data(ticker)
            position = self._build_position_block(ticker, data.get("price"))
            if position:
                data["position"] = position
            self.ticker_data[ticker] = data
            self.stored_views[ticker] = load_view(self.db_path, ticker)
        except Exception as e:  # noqa: BLE001
            logger.warning("Data fetch for %s failed (%s) — placeholder in the email", ticker, e)

    def _rank_all_news(self, stocks: list[str]) -> None:
        """Phase 2: one batched rerank for every holding at once."""
        from .stock_views import shown_links

        candidates, names = {}, {}
        for ticker in stocks:
            data = self.ticker_data.get(ticker)
            if not data:
                continue
            if not is_equity(data):
                continue  # a money-market fund has no company news
            seen = shown_links(self.stored_views.get(ticker))
            # Yesterday's headlines are not news; a story doing the rounds
            # for a week used to fill the section every morning.
            candidates[ticker] = [n for n in data.get("news") or [] if n.get("link") not in seen]
            names[ticker] = data.get("name")
        try:
            self.ranked_news = self.news_reranker.rerank_batch(candidates, names)
        except Exception as e:  # noqa: BLE001
            logger.warning("News ranking failed (%s) — blocks go out without news", e)
            self.ranked_news = {}

    def _collect_facts(self, stocks: list[str]) -> None:
        """Phase 3: estimate revisions for every holding, and for the ones
        with nothing to read, what the company actually did."""
        from ..discover.stock_facts import THIN_NEWS, fetch_company_facts

        equities = [t for t in stocks if is_equity(self.ticker_data.get(t))]
        if not equities:
            return
        thin = [t for t in equities if len(self.ranked_news.get(t) or []) < THIN_NEWS]
        if thin:
            logger.info(
                "No company-specific news for %s — fetching filings and insider activity",
                ", ".join(thin),
            )
        try:
            self.facts = fetch_company_facts(equities, deep=thin)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not fetch company facts (%s)", e)
            self.facts = {}

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

    def _safe_sentiment(self) -> str:
        try:
            return self._run_sentiment() or "Social/Economic Sentiment: unavailable today."
        except Exception as e:  # noqa: BLE001
            logger.warning("Market sentiment failed (%s) — email goes without it", e)
            return "Social/Economic Sentiment: unavailable today."

    def _safe_ticker(self, ticker: str) -> str:
        try:
            return self._run_ticker(ticker) or self._unavailable_block(ticker, "empty reply")
        except Exception as e:  # noqa: BLE001
            logger.warning("Analysis for %s failed (%s) — placeholder in the email", ticker, e)
            return self._unavailable_block(ticker, type(e).__name__)

    @staticmethod
    def _unavailable_block(ticker: str, why: str) -> str:
        return (
            f"{'-' * 40}\n\n{ticker} - analysis unavailable today\n"
            f"Long-term view: The analysis for {ticker} could not be produced today "
            f"({why}); it will be back tomorrow. Holdings and health checks above "
            f"still include it."
        )

    def _run_sentiment(self) -> str:
        items = fetch_market_sentiment_news()
        if not items:
            return "Social/Economic Sentiment: market news data unavailable."
        listing = "\n".join(f"- {it['title']}: {it.get('snippet', '')}" for it in items)
        prompt = f"Today's US market news:\n\n{listing}"
        logger.info("Synthesizing sentiment from %d items", len(items))
        return self.sentiment_agent.run(prompt).content

    def _run_ticker(self, ticker: str) -> str:
        """Phase 4: this holding's block, from data phases 1-3 gathered."""
        from .stock_views import (
            format_ticker_block,
            last_earnings_event,
            record_shown_news,
            refresh_reason,
            save_view,
        )

        today = date.today()
        data = self.ticker_data.get(ticker)
        if not data:
            raise RuntimeError("no data fetched")
        news = self.ranked_news.get(ticker) or []
        data["news"] = news
        facts = self.facts.get(ticker) or {}

        stored = self.stored_views.get(ticker)
        reason = refresh_reason(
            stored,
            price=data.get("price_value"),
            reported_on=last_earnings_event(data, today),
            today=today,
            max_age_days=self.view_max_age_days,
            move_pct=self.view_move_pct,
        )
        if reason is None and stored is not None:
            self.views_reused += 1
            written = date.fromisoformat(stored.written_on)
            block = format_ticker_block(
                data,
                view=stored.view,
                view_note=f" (view from {written:%b %d})",
                news=news,
                facts=facts,
                today=today,
            )
        else:
            prompt = f"Ticker data:\n```json\n{dumps_prompt(data)}\n```"
            logger.info("Writing long-term view for %s (%s)", ticker, reason)
            view = (self.ticker_agent.run(prompt).content or "").strip()
            if not view:
                raise RuntimeError("empty long-term view")
            save_view(self.db_path, ticker, view=view, price=data.get("price_value"), today=today)
            self.views_written += 1
            block = format_ticker_block(data, view=view, news=news, facts=facts, today=today)

        record_shown_news(
            self.db_path, ticker, [n["link"] for n in news if n.get("link")], today=today
        )
        return block

    def _build_position_block(self, ticker: str, current_price_str: str | None) -> dict | None:
        pos = self._positions_by_ticker.get(ticker)
        if not pos:
            return None

        units = pos["units"]
        avg = pos["avg_buy_price"]
        block: dict = {
            "units": (f"{int(units)}" if units == int(units) else f"{units:.4f}"),
            "avg_buy_price": f"${avg:,.2f}",
        }
        try:
            current = float((current_price_str or "").replace("$", "").replace(",", ""))
            pl_per_share = current - avg
            block["unrealized_pl"] = f"${pl_per_share * units:+,.2f}"
            block["pl_pct"] = f"{(pl_per_share / avg * 100):+.2f}%" if avg else None
        except ValueError, AttributeError, TypeError:
            pass
        return block
