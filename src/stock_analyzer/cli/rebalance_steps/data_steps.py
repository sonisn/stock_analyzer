"""Rebalance steps that gather holdings data: positions, prices, transactions,
and the holdings-specific overrides of the discover enrichment steps."""

from __future__ import annotations

from agno.workflow.types import StepInput, StepOutput

from ...data.brokerage import (
    fetch_account_cash,
    fetch_account_meta,
    fetch_account_sync_status,
    fetch_covered_call_obligations,
    fetch_portfolio_holdings,
    listed_tickers,
    stale_account_notes,
)
from ...data.finnhub import batch_finnhub_signals
from ...data.fundamentals import batch_fundamentals
from ...data.insider_selling import insider_selling_mentions
from ...data.sec_edgar import batch_quarterly_mda, batch_risk_factors
from ...data.share_trades import batch_share_trade_data
from ...data.technical_indicators import batch_technicals
from ...data.transactions import fetch_transaction_history, to_tax_payloads
from ...data.transcripts import batch_transcript_snippets
from ...discover.peers import batch_peer_comparison
from ...logging import get_logger
from ..pipeline_base import PipelineBase
from .helpers import (
    _aggregate_positions,
    _build_position_splits,
)

logger = get_logger("stock_analyzer.cli.rebalance")


class RebalanceDataSteps(PipelineBase):
    # --- new step executors -------------------------------------------------

    def step_holdings_fetch(self, step_input: StepInput) -> StepOutput:
        try:
            holdings = fetch_portfolio_holdings()
        except Exception as e:
            raise RuntimeError(f"Could not fetch SnapTrade holdings: {e}")  # noqa: B904
        positions = _aggregate_positions(holdings)
        if not positions:
            raise RuntimeError(
                "No SnapTrade positions found — rebalance requires existing holdings"
            )
        # Fetch account metadata (type + tax_status) and build per-account
        # position splits so the LLM stages can reason about which slice
        # of each holding is tax-advantaged.
        account_meta = fetch_account_meta()
        position_splits = _build_position_splits(holdings, account_meta)
        account_cash = fetch_account_cash()
        cash = sum(account_cash.values()) if account_cash else None
        # Cash and share counts from a broker that has stopped syncing are
        # as old as the connection: the plan's buys and put collateral
        # would be sized against numbers from that day.
        self.state["stale_accounts"] = stale_account_notes(fetch_account_sync_status())
        self.state["account_cash"] = account_cash
        # Which shares are already promised to a written call. Needed by
        # the harvester, the rebalancer's prompt and the sale validator —
        # fetched once here rather than three times.
        try:
            self.state["covered_call_obligations"] = fetch_covered_call_obligations()
        except Exception as e:  # noqa: BLE001 — a sale plan is still better than none
            logger.warning("Covered-call obligations unavailable (%s)", e)
            self.state["covered_call_obligations"] = {}
        self.state["holdings_positions"] = positions
        self.state["account_meta"] = account_meta
        self.state["position_splits"] = position_splits
        self.state["cash_balance"] = cash
        # Every holding stays in `positions` for valuation and tax, but
        # only the market-listed ones get fundamentals, chains and reviews.
        analyzable, unlisted = listed_tickers(holdings)
        if unlisted:
            logger.info("No market data for %s — not reviewed", ", ".join(unlisted))
        self.state["holdings_tickers"] = [t for t in positions if t in set(analyzable)]
        ta_count = sum(1 for v in position_splits.values() if v.get("has_tax_advantaged"))
        cash_str = f"${cash:,.0f}" if cash is not None else "unknown"
        return StepOutput(
            content=(
                f"Holdings: {len(positions)} positions ({ta_count} with "
                f"tax-advantaged exposure); cash {cash_str}"
            )
        )

    def step_holdings_data(self, step_input: StepInput) -> StepOutput:
        tickers = self.state["holdings_tickers"]
        self.state["holdings_fundamentals"] = batch_fundamentals(tickers)
        self.state["holdings_technicals"] = batch_technicals(tickers)
        self.state["holdings_risk_factors"] = batch_risk_factors(tickers)
        # Forward-narrative + peers + transcript for holdings — same data the
        # discover pipeline fetches for survivors. Reviewer uses these to
        # judge whether the current holding still ranks against peers.
        self.state["holdings_quarterly_mda"] = batch_quarterly_mda(tickers)
        target_meta = {
            t: {
                "name": (self.state["holdings_fundamentals"].get(t) or {}).get("name"),
                "sector": (self.state["holdings_fundamentals"].get(t) or {}).get("sector"),
            }
            for t in tickers
        }
        self.state["holdings_peers"] = batch_peer_comparison(
            tickers,
            target_meta,
            fallback=(
                self.settings.discover_fallback_provider,
                self.settings.resolve_fallback_model(),
            ),
        )
        self.state["holdings_transcripts"] = batch_transcript_snippets(tickers)
        return StepOutput(
            content=(
                f"Holdings enrichment: fundamentals={len(self.state['holdings_fundamentals'])}, "
                f"10-Q MD&A={len(self.state['holdings_quarterly_mda'])}, "
                f"peers={len(self.state['holdings_peers'])}, "
                f"transcripts={len(self.state['holdings_transcripts'])}"
            )
        )

    def step_transaction_history(self, step_input: StepInput) -> StepOutput:
        """Pull 3yr of SnapTrade activities and build per-ticker tax lot summaries.
        Runs independently of survivors — relies only on SnapTrade auth."""
        summaries = fetch_transaction_history(db_path=self.settings.discover_db_path)
        self.state["tax_lots"] = to_tax_payloads(summaries)
        n_lots = sum(s.get("lot_count", 0) for s in self.state["tax_lots"].values())
        return StepOutput(
            content=(
                f"Tax lots: {len(self.state['tax_lots'])} tickers, "
                f"{n_lots} total lots over 3yr lookback"
            )
        )

    def step_news(self, step_input: StepInput) -> StepOutput:
        """Override: include holdings tickers so reviewer sees recent catalysts."""
        from ..discover_steps.helpers import _batch_news

        tickers = set(self.state.get("survivor_tickers") or [])
        if self.state.get("holdings_tickers"):
            tickers |= set(self.state["holdings_tickers"])
        if not tickers:
            self.state["news"] = {}
            self.state["recent_news"] = {}
            return StepOutput(content="news: no tickers; skipping")
        self.state["news"] = _batch_news(list(tickers))
        self.state["recent_news"] = self._fetch_recent_news(sorted(tickers))
        return StepOutput(content=f"News fetched for {len(tickers)} tickers")

    def step_insider_selling(self, step_input: StepInput) -> StepOutput:
        """Override: include holdings tickers so reviewer sees selling on them too."""
        tickers = set(self.state.get("survivor_tickers") or [])
        if self.state.get("holdings_tickers"):
            tickers |= set(self.state["holdings_tickers"])
        if not tickers:
            self.state["insider_selling"] = {}
            return StepOutput(content="insider_selling: no tickers; skipping")
        self.state["insider_selling"] = insider_selling_mentions(tickers, days=14)
        return StepOutput(content=f"Insider selling: {len(self.state['insider_selling'])} flagged")

    def step_share_trades(self, step_input: StepInput) -> StepOutput:
        """Override: fetch insider/institutional data for both survivors AND holdings."""
        tickers = set(self.state.get("survivor_tickers") or [])
        if self.state.get("holdings_tickers"):
            tickers |= set(self.state["holdings_tickers"])
        if not tickers:
            self.state["share_trades"] = {}
            return StepOutput(content="share_trades: no tickers; skipping")
        self.state["share_trades"] = batch_share_trade_data(list(tickers))
        return StepOutput(
            content=f"Share trades fetched for {len(self.state['share_trades'])}/{len(tickers)}"
        )

    def step_finnhub_signals(self, step_input: StepInput) -> StepOutput:
        """Earnings surprise + recommendation trend + price targets +
        Form-4 insider activity for survivors AND current holdings."""
        tickers = set(self.state.get("survivor_tickers") or [])
        if self.state.get("holdings_tickers"):
            tickers |= set(self.state["holdings_tickers"])
        if not tickers:
            self.state["finnhub_signals"] = {}
            return StepOutput(content="finnhub_signals: no tickers; skipping")
        self.state["finnhub_signals"] = batch_finnhub_signals(list(tickers))
        n = sum(1 for v in self.state["finnhub_signals"].values() if v)
        return StepOutput(content=f"Finnhub signals: {n}/{len(tickers)} tickers covered")

    def step_contracted_book(self, step_input: StepInput) -> StepOutput:
        """SEC-filed order books for the holdings, for the sell decisions.

        Free (SEC XBRL), deterministic, and the one forward-looking number
        in the prompt that is signed rather than forecast. Never blocks the
        run: a name with no book is normal, and so is the API being down.
        """
        from ...data.backlog import backlog_block, batch_rpo

        tickers = self.state.get("holdings_tickers") or []
        if not tickers:
            return StepOutput(content="contracted_book: no analyzable holdings")
        try:
            books = batch_rpo(list(tickers))
        except Exception as e:  # noqa: BLE001
            logger.warning("Contracted-book fetch failed (%s) — continuing without it", e)
            books = {}
        self.state["contracted_book"] = books
        self.state["backlog_block"] = backlog_block(books)
        if books:
            logger.info(
                "Contracted book: %d of %d holding(s) tag one (%s)",
                len(books),
                len(tickers),
                ", ".join(
                    f"{t} {books[t]['yoy_pct']:+.0f}%"
                    for t in sorted(books)
                    if books[t].get("yoy_pct") is not None
                )
                or "no YoY comparison yet",
            )
        return StepOutput(content=f"contracted_book: {len(books)}/{len(tickers)} tagged")
