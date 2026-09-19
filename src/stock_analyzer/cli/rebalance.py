"""Portfolio rebalance pipeline.

Run: python -m stock_analyzer.cli.rebalance

Extends the discover pipeline with three new steps:
  - holdings_fetch    SnapTrade positions + cash balance
  - holdings_data     fundamentals/technicals/risk factors for held tickers
  - review_holdings   Sonnet per holding → HOLD / TRIM / SELL verdict
  - rebalance         Opus combines verdicts + discover picks + cash into action list

User-configured behavior (locked in via conversation):
  - Sizing: self-fund from SELL/TRIM proceeds AND add available cash
  - Horizon: every holding and pick is a long-term (3-5 year) investment;
    sells need a broken thesis, a clearly better long-term use of the
    money, concentration, or a tax loss — never short-term price action

Output: email with HTML body (charts for new picks inline) + PDF attachment.
Run history shares the discover.db SQLite file.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

from agno.db.sqlite import SqliteDb
from agno.workflow import Parallel, Step, Workflow
from agno.workflow.types import StepInput, StepOutput
from dotenv import load_dotenv

from ..config import Settings
from ..data import finnhub, yf_gateway
from ..data.brokerage import fetch_account_meta, fetch_portfolio_holdings, fetch_total_cash
from ..data.finnhub import batch_finnhub_signals
from ..data.fundamentals import batch_fundamentals
from ..data.insider_selling import insider_selling_mentions
from ..data.sec_edgar import batch_quarterly_mda, batch_risk_factors
from ..data.share_trades import batch_share_trade_data
from ..data.technical_indicators import batch_technicals
from ..data.transactions import fetch_transaction_history, to_tax_payloads
from ..data.transcripts import batch_transcript_snippets
from ..db.repository import fetch_recent_holdings_history, fetch_recent_picks
from ..db.session import get_session
from ..discover.catalysts import repair_catalysts
from ..discover.peers import batch_peer_comparison
from ..discover.premortem import PreMortemAgent
from ..discover.rebalance_cc import (
    apply_cc_plan_validation,
    cc_empty_state,
    log_rebalancer_input_estimate,
    run_cc_data_pipeline,
)
from ..discover.rebalance_csp import (
    apply_csp_plan_validation,
    csp_empty_state,
    csp_report_data,
    run_csp_data_pipeline,
)
from ..discover.rebalance_holdings import (
    build_holding_review_payloads,
    flag_drawdown_reviews,
)
from ..discover.rebalance_persist import (
    deliver_rebalance_email,
    fetch_pick_charts,
    gross_premium_from_plan,
    log_full_analysis,
    persist_rebalance_run,
    print_rebalance_terminal,
)
from ..discover.rebalancer import Rebalancer
from ..discover.report import (
    build_rebalance_sections,
    render_html_email,
    render_pdf,
)
from ..discover.reviewer import REVIEWER_INSTRUCTIONS, Reviewer, review_batch
from ..discover.tax_harvest import (
    find_harvest_candidates,
    flag_plan_conflicts,
    format_harvest_block,
    harvest_report_data,
)
from ..logging import get_logger
from ..preflight import PreflightError, preflight
from ..serialization import dumps_pretty
from ..usage import BUDGET, TRACKER, BudgetExceededError, estimate_cost, log_usage_summary
from .discover import (
    _QUARTERLY_MDA_CHARS,
    _RISK_FACTORS_CHARS,
    _TRANSCRIPT_CHARS,
    DiscoverPipeline,
    without_step_retries,
)

logger = get_logger(__name__)

_build_rebalance_sections = build_rebalance_sections


def _build_history_block(db_path: str, *, n_runs: int = 3) -> str:
    """Reach into the discover DB and produce a compact `Previous decisions`
    block for the rebalancer prompt. Per-holding format:

        AAPL: HOLD-8 (2026-05-05) -> HOLD-7 (2026-05-10) -> today

    Oldest first so the LLM sees chronological drift left-to-right. Returns
    an empty string if no history exists (first run, fresh DB, or every
    prior run was a discover-only run).
    """
    try:
        with get_session(db_path) as session:
            history = fetch_recent_holdings_history(session, n_runs=n_runs)
    except Exception as e:
        logger.warning("history fetch failed (%s) — proceeding without it", e)
        return ""
    if not history:
        return ""
    lines: list[str] = []
    for ticker in sorted(history.keys()):
        entries = history[ticker]
        parts: list[str] = []
        for e in entries:
            verdict = e.get("verdict") or "?"
            conf = e.get("confidence")
            run_at = (e.get("run_at") or "")[:10]
            label = f"{verdict}-{conf}" if conf is not None else verdict
            parts.append(f"{label} ({run_at})")
        parts.append("today")
        lines.append(f"{ticker}: {' -> '.join(parts)}")
    return "\n".join(lines)


def _aggregate_positions(
    holdings: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Collapse holdings-across-accounts into one position per ticker."""
    agg: dict[str, dict[str, float]] = {}
    for items in holdings.values():
        for h in items:
            ticker = h.get("ticker")
            units = h.get("units") or 0
            avg = h.get("average_purchase_price") or 0
            if not ticker or not units:
                continue
            cur = agg.setdefault(ticker, {"units": 0.0, "cost": 0.0})
            cur["units"] += float(units)
            cur["cost"] += float(units) * float(avg)
    out: dict[str, dict[str, float]] = {}
    for ticker, v in agg.items():
        if v["units"]:
            out[ticker] = {
                "units": v["units"],
                "avg_buy_price": v["cost"] / v["units"],
                "cost_basis": v["cost"],
            }
    return out


def build_email_subject(*, action_count: int, gross_premium_usd: float) -> str:
    """Subject line for the rebalance email. Annotates premium total
    only when WRITE_CALL / SELL_PUT actions produced a non-trivial credit."""
    today = date.today()
    base = f"Portfolio Rebalance — {today.strftime('%b-%d')}"
    if gross_premium_usd >= 1.0:
        return f"{base} ({action_count} actions + ${gross_premium_usd:,.0f} premium)"
    return base


def _build_position_splits(
    holdings: dict[str, list[dict[str, Any]]],
    account_meta: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Preserve per-account splits AND aggregate totals per ticker.

    Returns dict keyed by ticker with both views — the aggregated total
    (so existing code that reads holdings_positions still works) PLUS a
    list of per-account splits so the reviewer + rebalancer can reason
    about which slice of a position is in a tax-advantaged account
    (zero-tax trim) vs taxable (real tax cost):

      {
        "AAPL": {
          "total_units": 100, "total_cost": 12000, "avg_buy_price": 120,
          "splits": [
            {"account": "Fidelity Brokerage", "tax_status": "taxable",
             "units": 60, "avg_buy_price": 116.67, "cost_basis": 7000},
            {"account": "Fidelity IRA", "tax_status": "tax_advantaged",
             "units": 40, "avg_buy_price": 125.00, "cost_basis": 5000},
          ],
          "has_tax_advantaged": True,
          "has_taxable": True,
          "tax_advantaged_units": 40, "taxable_units": 60,
        },
      }
    """
    raw: dict[str, list[dict[str, Any]]] = {}
    for account_name, items in holdings.items():
        meta = account_meta.get(account_name) or {}
        tax_status = meta.get("tax_status") or "taxable"
        for h in items:
            ticker = h.get("ticker")
            units = h.get("units") or 0
            avg = h.get("average_purchase_price") or 0
            if not ticker or not units:
                continue
            raw.setdefault(ticker, []).append(
                {
                    "account": account_name,
                    "tax_status": tax_status,
                    "units": float(units),
                    "avg_buy_price": float(avg),
                    "cost_basis": float(units) * float(avg),
                }
            )

    out: dict[str, dict[str, Any]] = {}
    for ticker, splits in raw.items():
        total_units = sum(s["units"] for s in splits)
        total_cost = sum(s["cost_basis"] for s in splits)
        if not total_units:
            continue
        ta_units = sum(s["units"] for s in splits if s["tax_status"] == "tax_advantaged")
        tx_units = sum(s["units"] for s in splits if s["tax_status"] == "taxable")
        out[ticker] = {
            "total_units": total_units,
            "total_cost": total_cost,
            "avg_buy_price": total_cost / total_units if total_units else 0,
            "splits": splits,
            "has_tax_advantaged": ta_units > 0,
            "has_taxable": tx_units > 0,
            "tax_advantaged_units": ta_units,
            "taxable_units": tx_units,
        }
    return out


class RebalancePipeline(DiscoverPipeline):
    """Discovery + per-holding review + Opus rebalance plan delivery."""

    # Rebalance runs also review every holding, so the discover-side
    # Analyst fan-out gets a smaller slice of the budget.
    ANALYST_BUDGET_SHARE = 0.3

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
        cash = fetch_total_cash()
        self.state["holdings_positions"] = positions
        self.state["account_meta"] = account_meta
        self.state["position_splits"] = position_splits
        self.state["cash_balance"] = cash
        self.state["holdings_tickers"] = list(positions.keys())
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
        summaries = fetch_transaction_history(years_back=3)
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
        from .discover import _batch_news

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

    def step_review_holdings(self, step_input: StepInput) -> StepOutput:
        payloads = build_holding_review_payloads(
            positions=self.state["holdings_positions"],
            fund=self.state["holdings_fundamentals"],
            tech=self.state["holdings_technicals"],
            rfs=self.state["holdings_risk_factors"],
            insider_selling=self.state.get("insider_selling", {}),
            finnhub_signals=self.state.get("finnhub_signals", {}),
            eps_revisions=self.state.get("eps_revisions", {}),
            position_splits=self.state.get("position_splits", {}),
            account_meta=self.state.get("account_meta", {}),
            tax_lots_raw=self.state.get("tax_lots", {}),
            share_trades=self.state.get("share_trades", {}),
            holdings_quarterly_mda=self.state.get("holdings_quarterly_mda", {}),
            holdings_peers=self.state.get("holdings_peers", {}),
            holdings_transcripts=self.state.get("holdings_transcripts", {}),
            news=self.state.get("news") or {},
            recent_news=self.state.get("recent_news") or {},
            thesis_checks=self.state.get("thesis_checks") or [],
            risk_factors_chars=_RISK_FACTORS_CHARS,
            quarterly_mda_chars=_QUARTERLY_MDA_CHARS,
            transcript_chars=_TRANSCRIPT_CHARS,
        )
        reviewer = Reviewer(
            "claude",
            self._reviewer_model(payloads),
            fallback=(
                self.settings.discover_fallback_provider,
                self.settings.resolve_fallback_model(),
            ),
        )
        reviews, _ = repair_catalysts(
            review_batch(reviewer, payloads), self.state.get("recent_news") or {}
        )
        drawdown_notes = flag_drawdown_reviews(
            reviews,
            self.state["holdings_positions"],
            self.state["holdings_technicals"],
        )
        if drawdown_notes:
            self.state["stop_loss_warnings"] = drawdown_notes
        self.state["holdings_reviews"] = reviews
        return StepOutput(content=f"Reviewed {len(self.state['holdings_reviews'])} holdings")

    def _reviewer_model(self, payloads: dict[str, dict[str, Any]]) -> str:
        """Sonnet, unless reviewing every holding on it would not fit the cost
        cap's remaining room (after the final-stage reserve) — then Haiku."""
        sonnet = self.settings.discover_sonnet_model
        haiku = self.settings.discover_haiku_model
        available = BUDGET.available_for()
        if available is None or not haiku:
            return sonnet
        est = sum(
            estimate_cost(sonnet, len(dumps_pretty(p)) + len(REVIEWER_INSTRUCTIONS), 2000) or 0.0
            for p in payloads.values()
        )
        if est <= available:
            return sonnet
        BUDGET.note(
            f"reviewed {len(payloads)} holdings on {haiku} instead of {sonnet} "
            f"(estimated ${est:.2f} vs ${available:.2f} available)"
        )
        return haiku

    def step_cc_data(self, step_input: StepInput) -> StepOutput:
        """Build the COVERED-CALL CONTEXT block consumed by the rebalancer."""
        if not self.settings.cc_enabled:
            self.state.update(cc_empty_state())
            return StepOutput(content="cc_data: disabled via CC_ENABLED=0")

        self.state.update(cc_empty_state())
        try:
            result = run_cc_data_pipeline(self.state, self.settings)
            self.state["cc_context_block"] = result.context_block
            self.state["cc_eligibility"] = result.eligibility
            self.state["cc_round_lot_coverage"] = result.coverage
            self.state["cc_stub_pool_total_usd"] = result.stub_pool
            self.state["cc_chains"] = result.chains
            self.state["cc_iv_hv_regimes"] = result.iv_hv_regimes
            return StepOutput(content=result.content)
        except Exception as e:
            logger.error(
                "step_cc_data crashed (%s) — rebalance will run WITHOUT "
                "CC context. Investigate the traceback below.",
                e,
                exc_info=True,
            )
            return StepOutput(
                content=f"cc_data: failed ({type(e).__name__}); CC disabled for this run"
            )

    def step_csp_data(self, step_input: StepInput) -> StepOutput:
        """Build the CASH-SECURED PUT CONTEXT block consumed by the rebalancer."""
        self.state.update(csp_empty_state())
        if not self.settings.csp_enabled:
            return StepOutput(content="csp_data: disabled via CSP_ENABLED=0")
        try:
            with get_session(self.settings.discover_db_path) as session:
                recent = fetch_recent_picks(session, n_runs=self.settings.csp_pick_lookback_runs)
            result = run_csp_data_pipeline(self.state, self.settings, recent)
        except Exception as e:
            logger.error(
                "step_csp_data crashed (%s) — rebalance will run WITHOUT put ideas.",
                e,
                exc_info=True,
            )
            return StepOutput(
                content=f"csp_data: failed ({type(e).__name__}); puts disabled for this run"
            )
        self.state["csp_context_block"] = result.context_block
        self.state["csp_eligibility"] = result.eligibility
        self.state["csp_chains"] = result.chains
        self.state["csp_cash_budget"] = result.cash_budget
        return StepOutput(content=result.content)

    def step_tax_harvest(self, step_input: StepInput) -> StepOutput:
        """Deterministic tax-loss harvesting candidates (no LLM), fed to the
        Rebalancer and shown in the report."""
        try:
            prices = {
                t: (v or {}).get("price")
                for t, v in (self.state.get("holdings_technicals") or {}).items()
            }
            candidates = find_harvest_candidates(
                self.state.get("position_splits") or {},
                prices,
                self.state.get("tax_lots") or {},
                self.state.get("holdings_peers") or {},
                min_loss_usd=self.settings.harvest_min_loss_usd,
                min_loss_pct=self.settings.harvest_min_loss_pct,
            )
        except Exception as e:
            logger.warning("tax-loss harvest scan failed (%s) — skipping", e)
            candidates = []
        self.state["harvest_candidates_obj"] = candidates
        self.state["harvest_block"] = format_harvest_block(candidates)
        total = sum(-c.loss_usd for c in candidates)
        return StepOutput(
            content=f"Tax-loss harvest: {len(candidates)} candidates, ${total:,.0f} of losses"
        )

    def step_rebalance(self, step_input: StepInput) -> StepOutput:
        history_block = _build_history_block(self.settings.discover_db_path)
        if history_block:
            logger.info(
                "Cross-run context: %d holdings have prior decisions",
                history_block.count("\n"),
            )
        ranker_text = self.state.get("ranker_text") or ""
        if not ranker_text:
            logger.warning(
                "Rebalance: no ranker_text in state (Ranker step likely "
                "failed). Producing holdings-only plan from reviews."
            )
        rebalancer = Rebalancer(
            "claude",
            self.settings.discover_opus_model,
            cc_target_delta_min=self.settings.cc_target_delta_min,
            cc_target_delta_max=self.settings.cc_target_delta_max,
            cc_dte_min=self.settings.cc_dte_min,
            cc_dte_max=self.settings.cc_dte_max,
            cc_min_premium_usd=self.settings.cc_min_premium_usd,
            cc_slippage_buffer=self.settings.cc_slippage_buffer,
            cc_min_stub_usd=self.settings.cc_min_stub_usd,
            cc_stub_optimization=self.settings.cc_stub_optimization,
            csp_target_delta_min=self.settings.csp_target_delta_min,
            csp_target_delta_max=self.settings.csp_target_delta_max,
            csp_dte_min=self.settings.csp_dte_min,
            csp_dte_max=self.settings.csp_dte_max,
            csp_max_pct_per_put=self.settings.csp_max_pct_per_put,
            csp_max_pct_total=self.settings.csp_max_pct_total,
        )
        log_rebalancer_input_estimate(
            self.state,
            ranker_text=ranker_text,
            history_block=history_block,
        )
        plan = rebalancer.decide(
            self.state.get("holdings_reviews", {}),
            ranker_text,
            self.state.get("cash_balance"),
            self.state.get("macro_summary", ""),
            aggressiveness=self.settings.discover_rebalance_aggressiveness,
            history_block=history_block,
            market_themes_block=self.state.get("market_themes_block", ""),
            cc_context_block=self.state.get("cc_context_block", ""),
            harvest_block=self.state.get("harvest_block", ""),
            csp_context_block=self.state.get("csp_context_block", ""),
        )
        try:
            plan, cc_warnings = apply_cc_plan_validation(
                plan,
                chains=self.state.get("cc_chains") or {},
                eligibility=self.state.get("cc_eligibility") or {},
                cc_context_block=self.state.get("cc_context_block") or "",
            )
            if cc_warnings:
                self.state["cc_warnings"] = cc_warnings
        except Exception as e:
            logger.error(
                "CC validation crashed (%s) — using unvalidated plan. "
                "WRITE_CALL orphans / oversized contracts may slip through.",
                e,
                exc_info=True,
            )
            self.state["cc_warnings"] = [f"validation crashed: {e}"]
        try:
            plan, csp_warnings = apply_csp_plan_validation(
                plan,
                chains=self.state.get("csp_chains") or {},
                eligibility=self.state.get("csp_eligibility") or {},
                cash_budget=self.state.get("csp_cash_budget") or 0.0,
                settings=self.settings,
            )
        except Exception as e:
            # Unvalidated puts could over-commit cash — drop them all.
            logger.error("CSP validation crashed (%s) — dropping all puts.", e, exc_info=True)
            plan = plan.model_copy(
                update={
                    "actions": [a for a in plan.actions if a.action != "SELL_PUT"],
                    "csp_writes": [],
                }
            )
            csp_warnings = [f"put validation crashed ({e}); all puts dropped"]
        if csp_warnings:
            self.state["csp_warnings"] = csp_warnings
        self.state["rebalance_plan"] = plan
        self.state["rebalance_text"] = plan.full_text
        self.state["harvest_candidates"] = harvest_report_data(
            flag_plan_conflicts(self.state.get("harvest_candidates_obj") or [], plan)
        )
        return StepOutput(
            content=(
                f"Rebalance plan generated "
                f"(status={plan.status}, "
                f"aggressiveness={plan.aggressiveness_applied}, "
                f"actions={len(plan.actions)})"
            )
        )

    def step_premortem(self, step_input: StepInput) -> StepOutput:
        """Adversarial hindsight on the rebalance plan: imagine reading the
        news 6 months from now where this plan went wrong, and write the
        post-mortem from that future. Skips on NO_ACTION (nothing to
        pre-mortem)."""
        plan = self.state.get("rebalance_plan")
        if plan is None or getattr(plan, "status", None) != "ACTION":
            self.state["premortem"] = None
            return StepOutput(content="premortem: skipped (NO_ACTION plan)")
        # Format the holdings_reviews into a single text blob for the agent.
        from ..models.llm import HoldingReview

        reviews_text = "\n\n".join(
            f"=== {ticker} ===\n{r.full_text if isinstance(r, HoldingReview) else r}"
            for ticker, r in self.state.get("holdings_reviews", {}).items()
        )
        agent = PreMortemAgent("claude", self.settings.discover_opus_model)
        try:
            premortem = agent.run(
                rebalance_plan_text=plan.full_text,
                ranker_text=self.state.get("ranker_text", ""),
                holdings_reviews_text=reviews_text,
            )
        except BudgetExceededError:
            self.state["premortem"] = None
            return StepOutput(content="premortem: skipped (cost cap)")
        self.state["premortem"] = premortem
        if premortem is None:
            return StepOutput(content="premortem: agent returned no content")
        return StepOutput(
            content=(
                f"Pre-mortem: verdict={premortem.overall_verdict}, "
                f"{len(premortem.failures)} failure mode(s)"
            )
        )

    def _record_plan_suggestions(self, run_id: int) -> None:
        """Keep the plan's actions for the quarterly review."""
        from ..db.repository import record_suggestions

        plan = self.state.get("rebalance_plan")
        if plan is None or not plan.actions:
            return
        positions = self.state.get("holdings_positions") or {}
        prices = {
            **{t: (v or {}).get("price") for t, v in (self.state.get("technicals") or {}).items()},
            **{
                t: (v or {}).get("price")
                for t, v in (self.state.get("holdings_technicals") or {}).items()
            },
        }
        today = date.today().isoformat()
        rows = [
            {
                "suggested_on": today,
                "source": "rebalance",
                "action": a.action,
                "ticker": a.ticker,
                "detail": a.sizing,
                "price": prices.get(a.ticker),
                "units_held": (positions.get(a.ticker) or {}).get("units", 0.0),
                "run_id": run_id,
            }
            for a in plan.actions
        ]
        try:
            with get_session(self.settings.discover_db_path) as session:
                record_suggestions(session, rows)
        except Exception as e:
            logger.warning("Could not record the plan's suggestions (%s)", e)

    def _reinvest_for_unfunded_sales(self) -> dict[str, Any] | None:
        """Ideas for proceeds the plan leaves without a destination (runs
        after persistence, so this run's picks are in the pool)."""
        from ..discover.reinvest import load_pick_pool, reinvest_ideas, unfunded_sales

        sold = unfunded_sales(self.state.get("rebalance_plan"))
        if not sold:
            return None
        broken = {
            c["ticker"] for c in self.state.get("thesis_checks") or [] if c["status"] == "BROKEN"
        }
        try:
            ideas = reinvest_ideas(
                load_pick_pool(self.settings.discover_db_path),
                held=set(self.state.get("holdings_positions") or {}),
                exclude=broken,
                n=3,
            )
        except Exception as e:
            logger.warning("reinvestment ideas failed (%s) — report goes without them", e)
            return None
        return {"sold": sold, "ideas": ideas}

    def step_persist_and_email_rebalance(self, step_input: StepInput) -> StepOutput:
        candidates = self.state.get("candidates") or []
        survivors = self.state.get("survivors") or []
        picks = self.state.get("picks") or []
        analyses = self.state.get("analyses") or {}
        ranker_text = self.state.get("ranker_text") or ""
        redteam_text = self.state.get("redteam_text") or ""
        sizer_text = self.state.get("sizer_text") or ""

        with get_session(self.settings.discover_db_path) as session:
            run_id = persist_rebalance_run(
                session,
                state=self.state,
                settings=self.settings,
                candidates=candidates,
                survivors=survivors,
                picks=picks,
                analyses=analyses,
                ranker_text=ranker_text,
                redteam_text=redteam_text,
                sizer_text=sizer_text,
            )

        self._record_plan_suggestions(run_id)
        self._record_pick_suggestions(run_id)
        charts, chart_cids = fetch_pick_charts(picks)
        reinvest = self._reinvest_for_unfunded_sales()
        sections = build_rebalance_sections(
            rebalance_text=self.state.get("rebalance_text", "") or "",
            holdings_reviews=self.state.get("holdings_reviews", {}),
            ranker_text=ranker_text,
            redteam_text=redteam_text,
            sizer_text=sizer_text,
            candidates=candidates,
            cash_balance=self.state.get("cash_balance"),
            macro_summary=self.state.get("macro_summary", ""),
            sector_rotation=self.state.get("sector_rotation"),
            holdings_positions=self.state.get("holdings_positions", {}),
            holdings_technicals=self.state.get("holdings_technicals", {}),
            holdings_fundamentals=self.state.get("holdings_fundamentals", {}),
            track_record_block=self.state.get("track_record_block", ""),
            track_record=self.state.get("track_record"),
            thesis_checks=self.state.get("thesis_checks"),
            harvest_candidates=self.state.get("harvest_candidates"),
            rebalance_plan=self.state.get("rebalance_plan"),
            market_themes=self.state.get("market_themes"),
            premortem=self.state.get("premortem"),
            holdings_news=self.state.get("news"),
            cc_eligibility=self.state.get("cc_eligibility") or {},
            cc_round_lot_coverage=self.state.get("cc_round_lot_coverage") or {},
            cc_stub_pool_total_usd=self.state.get("cc_stub_pool_total_usd") or 0.0,
            cc_warnings=self.state.get("cc_warnings") or [],
            cc_slippage_buffer=self.settings.cc_slippage_buffer,
            csp_summary=csp_report_data(
                self.state.get("rebalance_plan"),
                cash_budget=self.state.get("csp_cash_budget") or 0.0,
            ),
            csp_warnings=self.state.get("csp_warnings") or [],
            reinvest=reinvest,
            stop_loss_warnings=self.state.get("stop_loss_warnings") or [],
            usage=TRACKER.report_data(),
        )
        html_body = render_html_email(sections, chart_cids)
        pdf_bytes = render_pdf(sections, charts)

        today = date.today()
        plan = self.state.get("rebalance_plan")
        action_count, gross_premium = gross_premium_from_plan(plan)
        if gross_premium > 0:
            logger.info(
                "CC summary at email time: %d WRITE_CALL(s), $%s gross premium "
                "across %d total action(s)",
                sum(1 for a in plan.actions if a.action == "WRITE_CALL"),
                f"{gross_premium:,.0f}",
                action_count,
            )

        subject = build_email_subject(
            action_count=action_count,
            gross_premium_usd=gross_premium,
        )
        pdf_filename = f"rebalance-{today.isoformat()}.pdf"
        delivered, delivery_error, local_pdf_path = deliver_rebalance_email(
            self.settings,
            subject=subject,
            html_body=html_body,
            charts=charts,
            chart_cids=chart_cids,
            pdf_bytes=pdf_bytes,
            pdf_filename=pdf_filename,
        )
        log_full_analysis(
            delivered=delivered,
            delivery_error=delivery_error,
            local_pdf_path=str(local_pdf_path),
            rebalance_text=self.state.get("rebalance_text", "") or "",
            ranker_text=ranker_text,
            redteam_text=redteam_text,
            sizer_text=sizer_text,
            holdings_reviews=self.state.get("holdings_reviews", {}),
        )
        print_rebalance_terminal(
            plan=plan,
            cc_block=self.state.get("cc_context_block") or "",
            ranker_text=ranker_text,
            sizer_text=sizer_text,
            rebalance_text=self.state.get("rebalance_text", "") or "",
            local_pdf_path=local_pdf_path,
        )

        self.state["run_id"] = run_id
        self.state["pdf_bytes"] = pdf_bytes
        self.state["local_pdf_path"] = str(local_pdf_path)
        status = "emailed" if delivered else "persisted (no email)"
        return StepOutput(
            content=(
                f"Rebalance run #{run_id} {status}; PDF {len(pdf_bytes)} bytes "
                f"(saved to {local_pdf_path})"
            )
        )

    # --- workflow assembly -------------------------------------------------

    def build_workflow(self) -> Workflow:
        db_path = Path(os.path.expanduser(self.settings.discover_db_path))
        db_path.parent.mkdir(parents=True, exist_ok=True)

        return without_step_retries(
            Workflow(
                name="Portfolio Rebalance",
                description=(
                    "Discover new picks + review current holdings + emit aggressive rebalance plan"
                ),
                db=SqliteDb(db_file=str(db_path), session_table="workflow_session"),
                steps=[
                    Step(name="universe", executor=self.step_universe),
                    Parallel(
                        Step(name="fundamentals", executor=self.step_fundamentals),
                        Step(name="technicals", executor=self.step_technicals),
                        Step(name="sector_rotation", executor=self.step_sector_rotation),
                        Step(name="macro_regime", executor=self.step_macro_regime),
                        Step(name="track_record", executor=self.step_track_record),
                        Step(name="eps_revisions", executor=self.step_eps_revisions),
                        Step(name="holdings_fetch", executor=self.step_holdings_fetch),
                        Step(
                            name="transaction_history",
                            executor=self.step_transaction_history,
                        ),
                        name="market_data",
                    ),
                    # Market themes after market_data (depends on sector_rotation +
                    # macro_regime from inside the parallel block).
                    Step(name="market_themes", executor=self.step_market_themes),
                    Step(name="screen", executor=self.step_screen),
                    Step(name="thesis_check", executor=self.step_thesis_check),
                    Parallel(
                        Step(name="risk_factors", executor=self.step_risk_factors),
                        Step(name="quarterly_mda", executor=self.step_quarterly_mda),
                        Step(name="news", executor=self.step_news),
                        Step(name="earnings", executor=self.step_earnings),
                        Step(name="insider_selling", executor=self.step_insider_selling),
                        Step(name="share_trades", executor=self.step_share_trades),
                        Step(name="peer_comparison", executor=self.step_peer_comparison),
                        Step(name="earnings_transcripts", executor=self.step_earnings_transcripts),
                        Step(name="finnhub_signals", executor=self.step_finnhub_signals),
                        Step(name="holdings_data", executor=self.step_holdings_data),
                        name="enrichment",
                    ),
                    Step(name="analyst", executor=self.step_analyst),
                    Step(name="holdings", executor=self.step_holdings),
                    Step(name="ranker", executor=self.step_ranker),
                    Step(name="redteam", executor=self.step_redteam),
                    Step(name="sizer", executor=self.step_sizer),
                    Step(name="review_holdings", executor=self.step_review_holdings),
                    Step(name="cc_data", executor=self.step_cc_data),
                    Step(name="csp_data", executor=self.step_csp_data),
                    Step(name="tax_harvest", executor=self.step_tax_harvest),
                    Step(name="rebalance", executor=self.step_rebalance),
                    Step(name="premortem", executor=self.step_premortem),
                    Step(
                        name="persist_and_email_rebalance",
                        executor=self.step_persist_and_email_rebalance,
                    ),
                    Step(name="history_upkeep", executor=self.step_history_upkeep),
                ],
            )
        )


def run() -> None:
    from ..market_time import use_market_timezone

    use_market_timezone()
    load_dotenv()
    # Pacing knobs live in the environment, and these modules are
    # imported before `.env` is loaded — re-read them now.
    yf_gateway.reload_from_env()
    finnhub.reload_from_env()
    settings = Settings.from_env()
    try:
        preflight(
            settings,
            needs_llm=True,
            needs_brokerage=True,
            needs_finnhub=bool(settings.finnhub_api_key),
            needs_email=bool(settings.email_to),
        )
    except PreflightError as e:
        logger.error("%s", e)
        raise SystemExit(2) from e
    pipeline = RebalancePipeline(settings)
    workflow = pipeline.build_workflow()
    logger.info("=== Portfolio rebalance pipeline starting ===")
    try:
        workflow.print_response(input="rebalance", stream=True)
    finally:
        log_usage_summary()
    if pipeline.state.get("run_id"):
        print(f"\nRun #{pipeline.state['run_id']} stored in {settings.discover_db_path}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
