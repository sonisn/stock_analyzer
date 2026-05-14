"""Portfolio rebalance pipeline.

Run: python -m stock_analyzer.cli.rebalance

Extends the discover pipeline with three new steps:
  - holdings_fetch    SnapTrade positions + cash balance
  - holdings_data     fundamentals/technicals/risk factors for held tickers
  - review_holdings   Sonnet per holding → HOLD / TRIM / SELL verdict
  - rebalance         Opus combines verdicts + discover picks + cash into action list

User-configured behavior (locked in via conversation):
  - Sizing: self-fund from SELL/TRIM proceeds AND add available cash
  - Aggressiveness: AGGRESSIVE churn — actively recommend SELLs where a
    materially better alternative exists, not only when thesis is broken

Output: email with HTML body (charts for new picks inline) + PDF attachment.
Run history shares the discover.db SQLite file.
"""
from __future__ import annotations

import contextlib
import os
from datetime import date
from pathlib import Path
from typing import Any

from agno.db.sqlite import SqliteDb
from agno.workflow import Parallel, Step, Workflow
from agno.workflow.types import StepInput, StepOutput
from dotenv import load_dotenv

from ..config import Settings
from ..data.brokerage import fetch_account_meta, fetch_portfolio_holdings, fetch_total_cash
from ..data.chart_img import fetch_charts
from ..data.finnhub import batch_finnhub_signals
from ..data.fundamentals import batch_fundamentals
from ..data.insider_selling import insider_selling_mentions
from ..data.sec_edgar import batch_quarterly_mda, batch_risk_factors
from ..data.share_trades import batch_share_trade_data
from ..data.technical_indicators import batch_technicals
from ..data.transactions import fetch_transaction_history, to_tax_payloads
from ..data.transcripts import batch_transcript_snippets
from ..discover.peers import batch_peer_comparison
from ..discover.persistence import (
    connect,
    fetch_recent_holdings_history,
    insert_candidate,
    insert_holdings_review,
    insert_pick,
    insert_run,
    insert_run_outputs,
    insert_scorecard,
)
from ..discover.premortem import PreMortem, PreMortemAgent
from ..discover.rebalancer import Rebalancer
from ..discover.report import (
    Section,
    build_sections,
    parse_confidence,
    parse_rebalance_status,
    parse_verdict,
    print_terminal_summary,
    render_html_email,
    render_pdf,
)
from ..discover.reviewer import Reviewer, review_batch
from ..discover.tax_lot_helper import enrich_tax_lots_with_impact
from ..logging import current_log_file, get_logger
from ..preflight import PreflightError, preflight
from ..reporting.smtp import SmtpServer
from .discover import (
    _QUARTERLY_MDA_CHARS,
    _RISK_FACTORS_CHARS,
    _TRANSCRIPT_CHARS,
    DiscoverPipeline,
    _trim,
)

logger = get_logger(__name__)

# Cap CC eligible holdings sent to the rebalancer to bound prompt size.
# 25 covers virtually every realistic portfolio while keeping the
# CC context block <~40KB (per-ticker ~1.5KB × 25 + round-lot table).
_CC_MAX_ELIGIBLE_FOR_PROMPT = 25


def _save_local_pdf(pdf_bytes: bytes, filename: str) -> Path:
    """Persist the PDF to ~/.stock_analyzer/reports/ so a missed email
    never costs the user the report. Override via REPORTS_DIR env."""
    reports_dir = Path(
        os.path.expanduser(os.getenv("REPORTS_DIR", "~/.stock_analyzer/reports"))
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / filename
    path.write_bytes(pdf_bytes)
    return path


def _build_history_block(db_path: str, *, n_runs: int = 3) -> str:
    """Reach into the discover DB and produce a compact `Previous decisions`
    block for the rebalancer prompt. Per-holding format:

        AAPL: HOLD-8 (2026-05-05) -> HOLD-7 (2026-05-10) -> today

    Oldest first so the LLM sees chronological drift left-to-right. Returns
    an empty string if no history exists (first run, fresh DB, or every
    prior run was a discover-only run).
    """
    try:
        with connect(db_path) as conn:
            history = fetch_recent_holdings_history(conn, n_runs=n_runs)
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


def _log_full_analysis(
    *,
    delivered: bool,
    delivery_error: str | None,
    local_pdf_path: str,
    rebalance_text: str,
    ranker_text: str,
    redteam_text: str,
    sizer_text: str,
    holdings_reviews: dict[str, Any],
) -> None:
    """Dump every analyst-produced section to the logger so the user can
    recover the full report from the log file when email fails."""
    from ..discover.schemas import HoldingReview
    bar = "=" * 70
    if not delivered:
        logger.error(
            "%s\nEMAIL NOT DELIVERED — full analysis follows in this log.\n"
            "Reason: %s\nPDF: %s\n%s",
            bar,
            delivery_error or "EMAIL_TO not configured",
            local_pdf_path,
            bar,
        )
    logger.info("%s\nREBALANCE PLAN\n%s\n%s", bar, bar, rebalance_text)
    logger.info("%s\nRANKER — discover picks\n%s\n%s", bar, bar, ranker_text)
    logger.info("%s\nRED TEAM — bear cases\n%s\n%s", bar, bar, redteam_text)
    logger.info("%s\nSIZER — allocation\n%s\n%s", bar, bar, sizer_text)
    for ticker in sorted(holdings_reviews):
        review = holdings_reviews.get(ticker)
        text = (
            review.full_text if isinstance(review, HoldingReview)
            else (review or "(review unavailable)")
        )
        logger.info("%s\nHOLDING REVIEW — %s\n%s\n%s", bar, ticker, bar, text)


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
    only when WRITE_CALL actions produced a non-trivial credit."""
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
            raw.setdefault(ticker, []).append({
                "account": account_name,
                "tax_status": tax_status,
                "units": float(units),
                "avg_buy_price": float(avg),
                "cost_basis": float(units) * float(avg),
            })

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
        ta_count = sum(
            1 for v in position_splits.values() if v.get("has_tax_advantaged")
        )
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
        self.state["holdings_peers"] = batch_peer_comparison(tickers, target_meta)
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
            return StepOutput(content="news: no tickers; skipping")
        self.state["news"] = _batch_news(list(tickers))
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
        return StepOutput(
            content=f"Insider selling: {len(self.state['insider_selling'])} flagged"
        )

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
        return StepOutput(
            content=f"Finnhub signals: {n}/{len(tickers)} tickers covered"
        )

    def step_review_holdings(self, step_input: StepInput) -> StepOutput:
        positions = self.state["holdings_positions"]
        fund = self.state["holdings_fundamentals"]
        tech = self.state["holdings_technicals"]
        rfs = self.state["holdings_risk_factors"]
        selling = self.state.get("insider_selling", {})
        finnhub_signals = self.state.get("finnhub_signals", {})
        eps_revisions = self.state.get("eps_revisions", {})
        position_splits = self.state.get("position_splits", {})
        account_meta = self.state.get("account_meta", {})
        tax_lots_raw = self.state.get("tax_lots", {})

        payloads: dict[str, dict[str, Any]] = {}
        for ticker, pos in positions.items():
            t = tech.get(ticker) or {}
            current = t.get("price")
            avg = pos["avg_buy_price"]
            units = pos["units"]
            pnl = None
            pnl_pct = None
            if current and avg:
                pnl = (current - avg) * units
                pnl_pct = (current - avg) / avg * 100
            fh = finnhub_signals.get(ticker) or {}
            insider_activity: Any = fh.get("insider_activity") or {
                "mention_count": selling.get(ticker, 0)
            }
            splits_info = position_splits.get(ticker) or {}
            payloads[ticker] = {
                "position": {
                    "units": units,
                    "avg_buy_price": avg,
                    "current_price": current,
                    "cost_basis": pos["cost_basis"],
                    "unrealized_pnl": pnl,
                    "unrealized_pnl_pct": pnl_pct,
                    # NEW: per-account splits so the reviewer knows
                    # whether trims trigger tax. tax_advantaged_units
                    # are FREE to trim (no realized gains, no tax).
                    "account_splits": splits_info.get("splits") or [],
                    "tax_advantaged_units": splits_info.get(
                        "tax_advantaged_units", 0
                    ),
                    "taxable_units": splits_info.get("taxable_units", 0),
                    "has_tax_advantaged": splits_info.get(
                        "has_tax_advantaged", False
                    ),
                    "has_taxable": splits_info.get("has_taxable", True),
                },
                "fundamentals": fund.get(ticker) or {},
                "technicals": t,
                "insider_activity": insider_activity,
                "earnings_surprise_history": fh.get("earnings_surprise") or [],
                "recommendation_trend": fh.get("recommendation_trend") or [],
                "analyst_price_targets": fh.get("price_targets") or {},
                "eps_revisions": eps_revisions.get(ticker) or {},
                "share_trades": self.state.get("share_trades", {}).get(ticker),
                "risk_factors_10k": _trim(
                    (rfs.get(ticker) or {}).get("risk_factors"),
                    _RISK_FACTORS_CHARS,
                ),
                "quarterly_mda": _trim(
                    (self.state.get("holdings_quarterly_mda", {}).get(ticker) or {}).get("mda"),
                    _QUARTERLY_MDA_CHARS,
                ),
                "peers": self.state.get("holdings_peers", {}).get(ticker),
                "earnings_transcript": _trim(
                    (self.state.get("holdings_transcripts", {}).get(ticker) or {}).get("snippet"),
                    _TRANSCRIPT_CHARS,
                ),
                "news": (self.state.get("news") or {}).get(ticker, []),
                "tax_lots": enrich_tax_lots_with_impact(
                    tax_lots_raw.get(ticker) or {},
                    current or 0.0,
                    account_meta,
                ),
            }

        reviewer = Reviewer("claude", self.settings.discover_sonnet_model)
        self.state["holdings_reviews"] = review_batch(reviewer, payloads)
        return StepOutput(
            content=f"Reviewed {len(self.state['holdings_reviews'])} holdings"
        )

    def step_cc_data(self, step_input: StepInput) -> StepOutput:
        """Build the COVERED-CALL CONTEXT block consumed by the rebalancer.

        Pulls option chains, parses open short-call positions, computes
        eligibility + round-lot coverage + earnings-filtered chains,
        and stashes the assembled prompt block in
        `self.state['cc_context_block']`.

        Gracefully degrades: if CC_ENABLED is false or no holdings are
        eligible, `cc_context_block` is "" and the rebalancer prompt
        simply omits the CC section.
        """
        if not self.settings.cc_enabled:
            self.state["cc_context_block"] = ""
            self.state["cc_eligibility"] = {}
            self.state["cc_round_lot_coverage"] = {}
            self.state["cc_stub_pool_total_usd"] = 0.0
            return StepOutput(content="cc_data: disabled via CC_ENABLED=0")

        # Safe defaults — populated below on success.
        self.state["cc_context_block"] = ""
        self.state["cc_eligibility"] = {}
        self.state["cc_round_lot_coverage"] = {}
        self.state["cc_stub_pool_total_usd"] = 0.0

        try:
            from ..data.brokerage import fetch_open_option_positions
            from ..data.options_chain import fetch_chains
            from ..discover.cc_eligibility import (
                apply_earnings_filter,
                build_cc_context_block,
                eligible_holdings,
                round_lot_coverage,
            )

            logger.info(
                "CC pipeline starting: CC_ENABLED=%s, delta_band=[%.2f, %.2f], "
                "DTE_band=[%d, %d], min_premium=$%.0f, slippage_buffer=%.0f%%",
                self.settings.cc_enabled,
                self.settings.cc_target_delta_min, self.settings.cc_target_delta_max,
                self.settings.cc_dte_min, self.settings.cc_dte_max,
                self.settings.cc_min_premium_usd,
                self.settings.cc_slippage_buffer * 100,
            )

            positions = self.state.get("holdings_positions") or {}
            denylist = self.settings.cc_denylist

            try:
                open_short_calls = fetch_open_option_positions()
            except Exception as e:
                logger.warning("open option position fetch failed: %s", e)
                open_short_calls = {}

            if open_short_calls:
                logger.info(
                    "CC: %d ticker(s) already collateralizing short calls: %s",
                    len(open_short_calls), dict(open_short_calls),
                )
            else:
                logger.info("CC: no existing short-call coverage detected")

            eligible = eligible_holdings(
                positions, open_short_calls=open_short_calls, denylist=denylist,
            )

            # ORATS IV-rank (timing signal). One batched call per run.
            from ..data.orats import fetch_iv_ranks
            iv_ranks = fetch_iv_ranks(list(eligible))
            self.state["cc_iv_ranks"] = iv_ranks

            # Bound eligible holdings to cap prompt size.
            if len(eligible) > _CC_MAX_ELIGIBLE_FOR_PROMPT:
                # Rank by available dollar exposure (proxy for premium potential).
                # Larger positions get prompt priority — they unlock more contracts
                # and more premium per contract.
                def _exposure(t: str) -> float:
                    rec = eligible[t]
                    spot = (
                        self.state.get("holdings_technicals", {}).get(t) or {}
                    ).get("price") or 0.0
                    return float(rec.available_shares) * float(spot)
                kept = sorted(eligible, key=_exposure, reverse=True)[:_CC_MAX_ELIGIBLE_FOR_PROMPT]
                dropped = sorted(set(eligible) - set(kept))
                logger.warning(
                    "CC: %d eligible holdings exceed prompt cap (%d); "
                    "keeping top %d by exposure, dropping %s",
                    len(eligible), _CC_MAX_ELIGIBLE_FOR_PROMPT,
                    len(kept), dropped,
                )
                eligible = {t: eligible[t] for t in kept}

            logger.info(
                "CC eligibility: %d/%d positions eligible (≥100 shares, not denylisted, "
                "post-short-call coverage). Eligible: %s",
                len(eligible), len(positions), sorted(eligible.keys()),
            )
            if not eligible:
                logger.warning(
                    "CC: NO eligible holdings — rebalancer will produce NO WRITE_CALL "
                    "recommendations. Reasons: positions < 100 shares OR all in denylist "
                    "OR fully collateralized by existing short calls."
                )

            spots = {
                t: (self.state.get("holdings_technicals", {}).get(t) or {}).get("price") or 0.0
                for t in positions
            }
            coverage = round_lot_coverage(positions, spots=spots)
            stub_pool = sum(
                rec.stub_dollar_value for rec in coverage.values() if rec.stub_shares
            )

            stub_eligible = sum(1 for rec in coverage.values() if rec.stub_dollar_value >= self.settings.cc_min_stub_usd)
            logger.info(
                "CC round-lot coverage: %d holding(s) have stubs, $%s total stub pool; "
                "%d stub(s) exceed CC_MIN_STUB_USD=$%s threshold",
                sum(1 for r in coverage.values() if r.stub_shares > 0),
                f"{stub_pool:,.0f}",
                stub_eligible,
                f"{self.settings.cc_min_stub_usd:,.0f}",
            )

            chains = fetch_chains(
                list(eligible),
                dte_min=self.settings.cc_dte_min,
                dte_max=self.settings.cc_dte_max,
            )

            chain_sources = {}
            for c in chains.values():
                chain_sources[c.source] = chain_sources.get(c.source, 0) + 1
            logger.info(
                "CC chain fetch: %d eligible ticker(s); sources: %s",
                len(chains), dict(chain_sources),
            )
            if chains and all(c.source == "missing" for c in chains.values()):
                logger.error(
                    "CC: ALL chain fetches failed (SnapTrade + yfinance both miss). "
                    "Opus will see UNAVAILABLE for every ticker and won't emit "
                    "WRITE_CALL. Check yfinance connectivity + SnapTrade tier."
                )

            finnhub_signals = self.state.get("finnhub_signals") or {}
            earnings_map: dict[str, date] = {}
            for ticker in eligible:
                sig = finnhub_signals.get(ticker) or {}
                raw = sig.get("next_earnings_date") or sig.get("earnings_date")
                if isinstance(raw, str):
                    with contextlib.suppress(ValueError):
                        earnings_map[ticker] = date.fromisoformat(raw[:10])
                elif isinstance(raw, date):
                    earnings_map[ticker] = raw

            logger.info(
                "CC earnings dates: %d/%d eligible tickers have known earnings date(s)",
                len(earnings_map), len(eligible),
            )

            filtered_chains: dict[str, object] = {}
            for ticker, chain in chains.items():
                filtered, _ = apply_earnings_filter(
                    chain, earnings_date=earnings_map.get(ticker),
                )
                filtered_chains[ticker] = filtered

            # Stash chains so step_rebalance can backfill OptionWrite
            # entries from WRITE_CALL sizing strings if Opus only fills
            # the actions list (observed in production).
            self.state["cc_chains"] = filtered_chains

            block = build_cc_context_block(
                eligible=eligible, chains=filtered_chains,
                coverage=coverage, reviews=self.state.get("holdings_reviews", {}),
                earnings=earnings_map, stub_pool_total_usd=stub_pool,
                iv_ranks=iv_ranks,
            )
            self.state["cc_context_block"] = block
            self.state["cc_eligibility"] = eligible
            self.state["cc_round_lot_coverage"] = coverage
            self.state["cc_stub_pool_total_usd"] = stub_pool

            logger.info(
                "CC context block built: %d chars (will be fed to rebalancer Opus)",
                len(block),
            )
            return StepOutput(content=(
                f"cc_data: {len(eligible)} eligible holding(s); "
                f"chain sources {sorted(chain_sources.keys())}; "
                f"stub pool ${stub_pool:,.0f}; "
                f"context block {len(block)} chars"
            ))
        except Exception as e:
            logger.error(
                "step_cc_data crashed (%s) — rebalance will run WITHOUT "
                "CC context. Investigate the traceback below.",
                e, exc_info=True,
            )
            return StepOutput(content=f"cc_data: failed ({type(e).__name__}); CC disabled for this run")

    def step_rebalance(self, step_input: StepInput) -> StepOutput:
        history_block = _build_history_block(self.settings.discover_db_path)
        if history_block:
            logger.info(
                "Cross-run context: %d holdings have prior decisions",
                history_block.count("\n"),
            )
        # Defensive: when discover stages (ranker/sizer) fail their retry
        # budget, the rebalancer can still produce a holdings-only plan
        # off the per-holding reviews. Pass empty discover context rather
        # than KeyError-ing the whole run.
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
        )

        # Pre-flight: estimate the rebalancer's prompt size so any
        # context-window pressure is visible before we burn the call.
        reviews_block_chars = sum(
            len(getattr(r, "full_text", str(r)) or "")
            for r in (self.state.get("holdings_reviews") or {}).values()
        )
        cc_block_chars = len(self.state.get("cc_context_block") or "")
        ranker_chars = len(ranker_text)
        history_chars = len(history_block)
        themes_chars = len(self.state.get("market_themes_block", "") or "")
        macro_chars = len(self.state.get("macro_summary", "") or "")
        total_input_chars = (
            reviews_block_chars + cc_block_chars + ranker_chars
            + history_chars + themes_chars + macro_chars
        )
        # Rough estimate: ~4 chars per token for English with markdown.
        approx_input_tokens = total_input_chars // 4
        logger.info(
            "Rebalancer input estimate: %d total chars (~%d tokens). "
            "Breakdown: reviews=%d, cc_block=%d, ranker=%d, history=%d, "
            "themes=%d, macro=%d. (Opus 4.7 input limit: 200,000 tokens.)",
            total_input_chars, approx_input_tokens,
            reviews_block_chars, cc_block_chars, ranker_chars,
            history_chars, themes_chars, macro_chars,
        )
        if approx_input_tokens > 150_000:
            logger.warning(
                "Rebalancer input is approaching the 200K-token context "
                "limit (estimated %d tokens). Consider reducing the number "
                "of holdings reviewed, or shortening reviewer.full_text.",
                approx_input_tokens,
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
        )
        try:
            from ..discover.cc_backfill import backfill_option_writes
            from ..discover.cc_validation import validate_option_writes
            # Synthesize OptionWrite entries for any WRITE_CALL actions
            # that Opus emitted without matching option_writes (observed
            # in production — Opus sometimes only fills the actions list
            # and the prose, dropping the structured field).
            plan = backfill_option_writes(
                plan, chains=self.state.get("cc_chains") or {},
            )
            plan, cc_warnings = validate_option_writes(
                plan, eligibility=self.state.get("cc_eligibility") or {},
            )
            if cc_warnings:
                self.state["cc_warnings"] = cc_warnings
                for w in cc_warnings:
                    logger.warning("CC plan validation: %s", w)

            n_write_calls = sum(1 for a in plan.actions if a.action == "WRITE_CALL")
            if n_write_calls > 0:
                total_premium = sum(
                    ow.contracts * ow.est_premium_per_share * 100.0
                    for ow in plan.option_writes
                )
                logger.info(
                    "CC validation passed: %d WRITE_CALL action(s), "
                    "$%s gross premium estimated. Details:",
                    n_write_calls, f"{total_premium:,.0f}",
                )
                for ow in plan.option_writes:
                    contract_premium = ow.contracts * ow.est_premium_per_share * 100.0
                    logger.info(
                        "  - %s: %d contracts @ $%.2f strike, expires %s, "
                        "Δ=%.2f, ~$%s premium, assignment %.0f%%",
                        ow.ticker, ow.contracts, ow.strike, ow.expiry,
                        ow.delta, f"{contract_premium:,.0f}",
                        ow.assignment_probability * 100,
                    )
            else:
                # Distinguish "rebalancer chose not to" from "CC was disabled / data missing"
                cc_block = self.state.get("cc_context_block") or ""
                if not cc_block:
                    logger.info(
                        "CC: no WRITE_CALL recommendations — CC context was empty "
                        "this run (no eligible holdings or CC_ENABLED=false)."
                    )
                else:
                    logger.warning(
                        "CC: rebalancer received CC context (%d chars) but emitted "
                        "0 WRITE_CALL actions. Possible reasons: every eligible chain "
                        "failed the liquidity guard (bid<$0.20, OI<100, spread>15%%), "
                        "every eligible position is a SELL verdict, or Opus declined.",
                        len(cc_block),
                    )
        except Exception as e:
            logger.error(
                "CC validation crashed (%s) — using unvalidated plan. "
                "WRITE_CALL orphans / oversized contracts may slip through.",
                e, exc_info=True,
            )
            self.state["cc_warnings"] = [f"validation crashed: {e}"]
        self.state["rebalance_plan"] = plan
        # `rebalance_text` is the prose rendering, kept under the same key
        # so the PDF/email layer and the log-dump fallback need no change.
        self.state["rebalance_text"] = plan.full_text
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
        from ..discover.schemas import HoldingReview
        reviews_text = "\n\n".join(
            f"=== {ticker} ===\n"
            f"{r.full_text if isinstance(r, HoldingReview) else r}"
            for ticker, r in self.state.get("holdings_reviews", {}).items()
        )
        agent = PreMortemAgent("claude", self.settings.discover_opus_model)
        premortem = agent.run(
            rebalance_plan_text=plan.full_text,
            ranker_text=self.state.get("ranker_text", ""),
            holdings_reviews_text=reviews_text,
        )
        self.state["premortem"] = premortem
        if premortem is None:
            return StepOutput(content="premortem: agent returned no content")
        return StepOutput(
            content=(
                f"Pre-mortem: verdict={premortem.overall_verdict}, "
                f"{len(premortem.failures)} failure mode(s)"
            )
        )

    def step_persist_and_email_rebalance(self, step_input: StepInput) -> StepOutput:
        # Defensive .get() reads throughout: any earlier discover stage
        # (universe / screen / ranker / sizer) can exhaust its retry budget
        # without setting state, and persisting + emailing the holdings
        # reviews is still valuable when that happens. KeyError-ing the
        # whole pipeline at the last step costs the user the entire run.
        candidates = self.state.get("candidates") or []
        survivors = self.state.get("survivors") or []
        picks = self.state.get("picks") or []
        analyses = self.state.get("analyses") or {}
        ranker_text = self.state.get("ranker_text") or ""
        redteam_text = self.state.get("redteam_text") or ""
        sizer_text = self.state.get("sizer_text") or ""

        # Persist run + candidates + scorecards + picks + outputs.
        with connect(self.settings.discover_db_path) as conn:
            run_id = insert_run(
                conn,
                universe_size=len(candidates),
                survivors=len(survivors),
                picks=len(picks),
                opus_model=self.settings.discover_opus_model,
                sonnet_model=self.settings.discover_sonnet_model,
                cash_budget=self.state.get("cash_balance"),
                kind="rebalance",
            )
            for c in candidates:
                insert_candidate(
                    conn,
                    run_id,
                    c["ticker"],
                    passed_filter=c["passed_filter"],
                    fail_reasons=c["fail_reasons"],
                    score=c["score"],
                    score_components=c["score_components"],
                    score_breakdown=c["score_breakdown"],
                    sources=c["sources"],
                    conviction=c["conviction"],
                    sector=c["sector"],
                    price=c["price"],
                )
            for ticker, report in analyses.items():
                analyst_text = getattr(report, "full_text", None) or (
                    report if isinstance(report, str) else ""
                )
                insert_scorecard(conn, run_id, ticker, analyst_text)
            for ticker, review in self.state.get("holdings_reviews", {}).items():
                if not review:
                    continue
                # `review` is a HoldingReview (Phase 4a). Fall back gracefully
                # if a legacy str leaks through.
                review_text = getattr(review, "full_text", None) or (
                    review if isinstance(review, str) else ""
                )
                insert_holdings_review(
                    conn,
                    run_id,
                    ticker,
                    verdict=parse_verdict(review),
                    confidence=parse_confidence(review),
                    review_text=review_text,
                )
            for rank, ticker, _ in picks:
                insert_pick(
                    conn,
                    run_id,
                    rank=rank,
                    ticker=ticker,
                    ranker_text=ranker_text,
                    bear_case_text=redteam_text,
                    allocation_text=sizer_text,
                )
            plan = self.state.get("rebalance_plan")
            # Use .get() with empty-string defaults for the rebalancer's
            # outputs so a failed rebalance step (e.g. LLM error after
            # max retries) doesn't compound into a KeyError that
            # tanks persistence too.
            insert_run_outputs(
                conn,
                run_id,
                ranker_full=ranker_text,
                redteam_full=redteam_text,
                sizer_full=sizer_text,
                holdings_summary=self.state.get("holdings_summary", "") or "",
                rebalance_text=self.state.get("rebalance_text", "") or "",
                dashboard_data=plan.model_dump(mode="json") if plan else None,
            )

        # Charts for the BUY candidates (discover picks). If Ranker
        # failed, picks list is empty and we just skip — the holdings
        # reviews still render fine without ticker charts.
        pick_tickers = [t for _, t, _ in picks]
        charts: dict[str, bytes] = {}
        try:
            charts = fetch_charts(pick_tickers) if pick_tickers else {}
        except Exception as e:
            logger.warning("Chart fetch failed (%s) — report will omit charts", e)
        chart_cids = {t: f"chart-{t.replace('.', '-')}" for t in charts}

        sections = _build_rebalance_sections(
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
            rebalance_plan=self.state.get("rebalance_plan"),
            market_themes=self.state.get("market_themes"),
            premortem=self.state.get("premortem"),
            holdings_news=self.state.get("news"),
            cc_eligibility=self.state.get("cc_eligibility") or {},
            cc_round_lot_coverage=self.state.get("cc_round_lot_coverage") or {},
            cc_stub_pool_total_usd=self.state.get("cc_stub_pool_total_usd") or 0.0,
            cc_warnings=self.state.get("cc_warnings") or [],
            cc_slippage_buffer=self.settings.cc_slippage_buffer,
        )
        html_body = render_html_email(sections, chart_cids)
        pdf_bytes = render_pdf(sections, charts)

        today = date.today()
        plan = self.state.get("rebalance_plan")
        gross_premium = 0.0
        if plan is not None and getattr(plan, "option_writes", None):
            gross_premium = sum(
                ow.contracts * ow.est_premium_per_share * 100.0
                for ow in plan.option_writes
            )
        action_count = len(plan.actions) if plan else 0

        if gross_premium > 0:
            logger.info(
                "CC summary at email time: %d WRITE_CALL(s), $%s gross premium "
                "across %d total action(s)",
                sum(1 for a in plan.actions if a.action == "WRITE_CALL"),
                f"{gross_premium:,.0f}", action_count,
            )

        subject = build_email_subject(
            action_count=action_count, gross_premium_usd=gross_premium,
        )
        pdf_filename = f"rebalance-{today.isoformat()}.pdf"

        # Save PDF locally BEFORE the email attempt so a delivery failure
        # (SMTP outage, wrong creds, etc.) never costs the user the report.
        local_pdf_path = _save_local_pdf(pdf_bytes, pdf_filename)
        logger.info("Saved rebalance PDF locally: %s", local_pdf_path)

        delivered = False
        delivery_error: str | None = None
        if self.settings.email_to:
            try:
                SmtpServer().send_email(
                    self.settings.email_to,
                    subject,
                    html_body,
                    content_type="html",
                    inline_images={
                        chart_cids[t]: data for t, data in charts.items()
                    } or None,
                    attachments=[(pdf_filename, pdf_bytes, "pdf")],
                )
                delivered = True
                logger.info(
                    "Sent rebalance email to %s", self.settings.email_to
                )
            except Exception as e:
                delivery_error = str(e)
                logger.error("Email delivery failed: %s", e)
        else:
            logger.warning("EMAIL_TO not set; skipping email")

        # Always dump the full analysis to the log so the user can recover
        # every section — even when email is offline or wasn't configured.
        _log_full_analysis(
            delivered=delivered,
            delivery_error=delivery_error,
            local_pdf_path=str(local_pdf_path),
            rebalance_text=self.state.get("rebalance_text", "") or "",
            ranker_text=ranker_text,
            redteam_text=redteam_text,
            sizer_text=sizer_text,
            holdings_reviews=self.state.get("holdings_reviews", {}),
        )

        # CC summary — visible in terminal so the user sees at-a-glance
        # whether the covered-call flow triggered.
        plan = self.state.get("rebalance_plan")
        if plan is not None:
            n_writes = sum(1 for a in plan.actions if a.action == "WRITE_CALL")
            print("\n" + "=" * 60)
            print("COVERED-CALL SUMMARY")
            print("=" * 60)
            if n_writes > 0:
                gross = sum(
                    ow.contracts * ow.est_premium_per_share * 100.0
                    for ow in plan.option_writes
                )
                print(f"  Recommendations: {n_writes} WRITE_CALL action(s)")
                print(f"  Gross premium:   ${gross:,.0f}")
                for ow in plan.option_writes:
                    print(
                        f"    {ow.ticker}: {ow.contracts}x ${ow.strike:.2f}C "
                        f"expires {ow.expiry}, Δ={ow.delta:.2f}, "
                        f"premium ${ow.contracts * ow.est_premium_per_share * 100:,.0f}"
                    )
            else:
                cc_block = self.state.get("cc_context_block") or ""
                if not cc_block:
                    print("  No recommendations: CC context was empty.")
                    print("  (No eligible ≥100-share holdings, CC_ENABLED=0, or chain fetch failed.)")
                else:
                    print("  No recommendations: rebalancer declined to write calls this run.")
                    print(f"  CC context ({len(cc_block)} chars) WAS provided to Opus.")

        print_terminal_summary(ranker_text, sizer_text)
        print("\n" + "=" * 60)
        print("REBALANCE PLAN")
        print("=" * 60)
        print(self.state.get("rebalance_text") or "(no plan produced)")
        print(f"\nPDF saved: {local_pdf_path}")
        log_path = current_log_file()
        if log_path:
            print(f"Log file:  {log_path}")

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

        return Workflow(
            name="Portfolio Rebalance",
            description=(
                "Discover new picks + review current holdings + emit "
                "aggressive rebalance plan"
            ),
            db=SqliteDb(
                db_file=str(db_path), session_table="workflow_session"
            ),
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
                Step(name="rebalance", executor=self.step_rebalance),
                Step(name="premortem", executor=self.step_premortem),
                Step(
                    name="persist_and_email_rebalance",
                    executor=self.step_persist_and_email_rebalance,
                ),
            ],
        )


def _build_rebalance_sections(
    *,
    rebalance_text: str,
    holdings_reviews: dict[str, Any],
    ranker_text: str,
    redteam_text: str,
    sizer_text: str,
    candidates: list[dict[str, Any]],
    cash_balance: float | None,
    macro_summary: str,
    sector_rotation: dict[str, Any] | None,
    holdings_positions: dict[str, dict[str, Any]],
    holdings_technicals: dict[str, dict[str, Any]],
    holdings_fundamentals: dict[str, dict[str, Any]],
    track_record_block: str = "",
    rebalance_plan: object = None,
    market_themes: object = None,
    premortem: object = None,
    holdings_news: dict[str, list[dict[str, Any]]] | None = None,
    cc_eligibility: dict[str, Any] | None = None,
    cc_round_lot_coverage: dict[str, Any] | None = None,
    cc_stub_pool_total_usd: float = 0.0,
    cc_warnings: list[str] | None = None,
    cc_slippage_buffer: float = 0.10,
) -> list[Section]:
    """Rebalance-specific layout — status banner + metrics + dashboard +
    sector pie at the top, then the LLM's plan + per-holding reviews +
    discover-picks appendix.

    `rebalance_plan` is the structured RebalancePlan from the LLM (Phase 3);
    `rebalance_text` is the prose rendering (plan.full_text). The plan is
    used to determine status without regex; the prose is what we render."""
    today = date.today().isoformat()

    # ---- Parse status + collect dashboard rows ----------------------------
    # Prefer the structured plan; fall back to regex on text only when not
    # available (legacy or partial runs).
    status = parse_rebalance_status(rebalance_plan or rebalance_text)
    status_label = (
        "STATUS: NO ACTION RECOMMENDED" if status == "NO_ACTION"
        else "STATUS: ACTION RECOMMENDED" if status == "ACTION"
        else "STATUS: REVIEW REQUIRED"
    )

    dashboard_rows: list[dict[str, Any]] = []
    total_value = 0.0
    total_cost = 0.0
    sector_value: dict[str, float] = {}
    for ticker in sorted(holdings_positions.keys()):
        pos = holdings_positions[ticker]
        tech = holdings_technicals.get(ticker, {})
        fund = holdings_fundamentals.get(ticker, {})
        review = holdings_reviews.get(ticker) or ""
        current = tech.get("price")
        units = pos.get("units", 0)
        cost = pos.get("cost_basis", 0)
        value = (current or 0) * units
        total_value += value
        total_cost += cost
        pnl_pct = ((value - cost) / cost * 100) if cost else None
        sector = fund.get("sector") or "Unknown"
        if value > 0:
            sector_value[sector] = sector_value.get(sector, 0) + value
        dashboard_rows.append({
            "ticker": ticker,
            "verdict": parse_verdict(review),
            "confidence": parse_confidence(review),
            "pnl_pct": pnl_pct,
            "sector": sector,
            "note": "",
        })

    total_pnl_pct = ((total_value - total_cost) / total_cost * 100) if total_cost else None

    # ---- Build sections ---------------------------------------------------
    sections: list[Section] = [
        Section(kind="heading", text=f"Portfolio Rebalance — {today}", level=1),
        Section(kind="status_banner", text=status_label, status=status),
    ]

    metrics: list[tuple[str, str]] = [
        ("Holdings", f"{len(holdings_positions)}"),
        ("Portfolio value", f"${total_value:,.0f}" if total_value else "—"),
        ("Total P/L", f"{total_pnl_pct:+.1f}%" if total_pnl_pct is not None else "—"),
        ("Cash", f"${cash_balance:,.0f}" if cash_balance is not None else "—"),
    ]
    sections.append(Section(kind="metric_strip", metrics=metrics))

    if dashboard_rows:
        sections.append(Section(kind="heading", text="Holdings dashboard", level=2))
        sections.append(
            Section(kind="holdings_dashboard", holdings=dashboard_rows)
        )

    # Recent catalysts — informational only. Surfaces fresh headlines per
    # holding so the user can spot catalysts that postdate the latest 10-Q
    # or earnings transcript. Does NOT drive sizing or verdicts — those
    # decisions stay with the Reviewer / Rebalancer.
    catalyst_rows: list[list[str]] = []
    if holdings_news:
        for ticker in sorted(holdings_positions.keys()):
            items = holdings_news.get(ticker) or []
            for item in items[:2]:
                title = (item.get("title") or "").strip()
                if title:
                    catalyst_rows.append([ticker, title])
    if catalyst_rows:
        sections.append(Section(
            kind="heading", text="Recent catalysts (informational)", level=2,
        ))
        sections.append(Section(
            kind="para",
            text=(
                "Headlines worth scanning. Not used to compute verdicts or "
                "position sizing — your Reviewer/Rebalancer reads news as "
                "qualitative context only."
            ),
        ))
        sections.append(Section(
            kind="table",
            table_header=["Ticker", "Headline"],
            table_rows=catalyst_rows,
        ))

    if sector_value:
        pie_data = sorted(sector_value.items(), key=lambda x: x[1], reverse=True)
        sections.append(Section(kind="heading", text="Sector allocation", level=2))
        sections.append(Section(kind="sector_pie", pie_data=pie_data))

    if track_record_block:
        sections.append(Section(kind="heading", text="Track record", level=2))
        sections.append(Section(kind="preformatted", text=track_record_block))

    from ..discover.schemas import MarketThemes
    if isinstance(market_themes, MarketThemes) and market_themes.themes:
        sections.append(Section(kind="heading", text="Current market themes", level=2))
        sections.append(Section(
            kind="market_themes_panel",
            data={
                "themes": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "strength": t.strength,
                        "trending": t.trending,
                        "member_tickers": list(t.member_tickers),
                    }
                    for t in market_themes.themes
                ],
            },
        ))

    if macro_summary:
        sections.append(Section(kind="heading", text="Macro regime", level=2))
        sections.append(Section(kind="blockquote", text=macro_summary))

    sections.append(Section(kind="page_break"))
    sections.append(Section(kind="heading", text="Rebalance plan (action list)", level=1))
    # Structured action table when status=ACTION — one styled row per
    # action with a SELL/TRIM/ADD/BUY badge. NO_ACTION runs have no
    # actions; skip the table entirely (status banner already conveys
    # the verdict).
    from ..discover.rebalance_schema import RebalancePlan
    plan = rebalance_plan if isinstance(rebalance_plan, RebalancePlan) else None
    if plan and plan.actions:
        sections.append(Section(
            kind="rebalance_action_table",
            data={
                "actions": [
                    {"action": a.action, "ticker": a.ticker, "sizing": a.sizing}
                    for a in plan.actions
                ],
                "summary": plan.summary,
            },
        ))

    if isinstance(premortem, PreMortem) and (premortem.failures or premortem.summary):
        sections.append(Section(
            kind="heading", text="Pre-mortem (adversarial hindsight)", level=2,
        ))
        sections.append(Section(
            kind="premortem_panel",
            data={
                "overall_verdict": premortem.overall_verdict,
                "summary": premortem.summary,
                "failures": [
                    {
                        "likelihood": f.likelihood,
                        "severity": f.severity,
                        "triggering_action": f.triggering_action,
                        "failure_narrative": f.failure_narrative,
                        "early_warning": f.early_warning,
                    }
                    for f in premortem.failures
                ],
            },
        ))

    # Covered-call report sections — rendered only when relevant.
    from ..discover.cc_render import (
        compute_premium_deployment,
        compute_premium_income,
        compute_round_lot_summary,
    )
    if plan is not None and plan.option_writes:
        sections.append(Section(
            kind="premium_income",
            data=compute_premium_income(plan, slippage_buffer=cc_slippage_buffer),
        ))
    if cc_round_lot_coverage:
        rls = compute_round_lot_summary(cc_round_lot_coverage)
        if rls["rows"]:
            sections.append(Section(kind="round_lot_coverage", data=rls))
    if plan is not None and (
        plan.option_writes
        or any(
            a.action in ("ADD", "BUY")
            or (a.action == "TRIM" and "stub" in a.sizing.lower())
            for a in plan.actions
        )
    ):
        stub_usd = 0.0
        if cc_round_lot_coverage:
            for a in plan.actions:
                if a.action == "TRIM" and "stub" in a.sizing.lower():
                    rec = cc_round_lot_coverage.get(a.ticker)
                    if rec is not None:
                        stub_usd += getattr(rec, "stub_dollar_value", 0.0)
        deployment = compute_premium_deployment(
            plan, cash_balance=cash_balance, slippage_buffer=cc_slippage_buffer,
            stub_consolidation_usd=stub_usd,
        )
        if (
            deployment["gross_premium_usd"] > 0
            or deployment["deployments"]
            or stub_usd > 0
        ):
            sections.append(Section(kind="premium_deployment", data=deployment))

    if cc_warnings:
        sections.append(Section(
            kind="para",
            text="CC plan adjustments: " + "; ".join(cc_warnings),
        ))

    sections.append(Section(kind="preformatted", text=rebalance_text))

    sections.append(Section(kind="page_break"))
    sections.append(Section(kind="heading", text="Per-holding reviews", level=1))
    from ..discover.schemas import HoldingReview
    for ticker in sorted(holdings_reviews.keys()):
        review = holdings_reviews[ticker]
        if isinstance(review, HoldingReview):
            # Structured card — pills, labeled sections, tax lot list,
            # wash-sale callout. Renders much nicer than the monospace
            # preformatted dump.
            sections.append(Section(
                kind="holding_review_card",
                data={
                    "ticker": ticker,
                    "verdict": review.verdict,
                    "confidence": review.confidence,
                    "trim_pct": review.trim_pct,
                    "position_context": review.position_context,
                    "forward_outlook": review.forward_outlook,
                    "reasoning": review.reasoning,
                    "tax_lot_plan": list(review.tax_lot_plan),
                    "what_would_change_mind": review.what_would_change_mind,
                    "wash_sale_notice": review.wash_sale_notice,
                },
            ))
        else:
            # Legacy free-text path — keep the old layout.
            text = review or ""
            sections.append(Section(kind="heading", text=ticker, level=2))
            sections.append(Section(kind="preformatted", text=text))

    # Discover picks as appendix — drop their own H1 + summary since we have ours.
    discover_sections = build_sections(
        ranker_text=ranker_text,
        redteam_text=redteam_text,
        sizer_text=sizer_text,
        candidates=candidates,
        universe_size=len(candidates),
        holdings_summary="",
        macro_summary="",
        sector_rotation=sector_rotation,
    )
    sections.append(Section(kind="page_break"))
    sections.append(
        Section(kind="heading", text="Discover picks (input to rebalancer)", level=1)
    )
    sections.extend(discover_sections[2:])
    return sections


def run() -> None:
    load_dotenv()
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
    workflow.print_response(input="rebalance", stream=True)
    if pipeline.state.get("run_id"):
        print(
            f"\nRun #{pipeline.state['run_id']} stored in {settings.discover_db_path}"
        )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
