"""Discover steps that reason over the data: analyst, holdings, ranker, macro
veto, red team, sizer."""

from __future__ import annotations

from typing import Any

from agno.workflow.types import StepInput, StepOutput

from ...data.brokerage import fetch_portfolio_holdings
from ...data.fundamentals import batch_fundamentals
from ...discover.analyst import Analyst, analyze_tiered, plan_under_budget
from ...discover.catalysts import repair_catalysts
from ...discover.data_reconciliation import reconcile_price_targets
from ...discover.macro_filter import apply_macro_veto
from ...discover.output_validation import validate_pick_scenarios
from ...discover.ranker import Ranker
from ...discover.redteam import RedTeam
from ...discover.report import (
    parse_picks,
)
from ...discover.sizer import (
    Sizer,
    enforce_correlation_caps,
    enforce_earnings_blackout,
    enforce_sector_caps,
    format_sector_exposure_block,
)
from ...logging import get_logger
from ...model.ranker_model import load_latest_model, score_percentiles, screen_points
from ...serialization import dumps_pretty
from ...usage import BUDGET
from .helpers import (
    _QUARTERLY_MDA_CHARS,
    _RISK_FACTORS_CHARS,
    _TRANSCRIPT_CHARS,
    _format_agreement_block,
    _format_ev_table,
    _format_risk_parity_block,
    _holdings_summary,
    _holdings_table_rows,
    _holdings_value_by_sector,
    _trim,
)

logger = get_logger("stock_analyzer.cli.discover")


class AnalysisSteps:
    def step_analyst(self, step_input: StepInput) -> StepOutput:
        survivors = self.state.get("survivors") or []
        if not survivors:
            # Empty after screen short-circuited. Set everything downstream
            # depends on so the rest of the pipeline degrades cleanly.
            self.state["analyses"] = {}
            return StepOutput(content="analyst: no survivors; skipping")
        payloads = self._analyst_payloads(survivors)
        analyses, catalyst_warnings = self._run_analysts(survivors, payloads)
        self.state["analyses"] = analyses
        self.state["catalyst_warnings"] = catalyst_warnings
        if not self.state["analyses"]:
            logger.error("Analyst: all calls failed; downstream LLM stages will skip")
            return StepOutput(content="Analyst: all calls failed; downstream will skip")
        return StepOutput(content=f"Analyst: {len(self.state['analyses'])} scorecards")

    def _analyst_payloads(self, survivors: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Everything the Analyst sees about each survivor."""
        fundamentals = self.state.get("fundamentals", {})
        technicals = self.state.get("technicals", {})
        risk_factors = self.state.get("risk_factors", {})
        news = self.state.get("news", {})
        recent_news = self.state.get("recent_news") or {}

        earnings_alerts = self.state.get("earnings_alerts", {})
        insider_selling = self.state.get("insider_selling", {})
        share_trades = self.state.get("share_trades", {})
        finnhub_signals = self.state.get("finnhub_signals", {})
        eps_revisions = self.state.get("eps_revisions", {})

        payloads: dict[str, dict[str, Any]] = {}
        for c in survivors:
            ticker = c["ticker"]
            fh = finnhub_signals.get(ticker) or {}
            # Prefer Finnhub's Form 4 record when available; fall back to
            # the Tavily news-mention count for tickers Finnhub doesn't cover.
            insider_activity: Any = fh.get("insider_activity") or {
                "mention_count": insider_selling.get(ticker, 0)
            }
            reconciliation_flags = [
                flag
                for flag in [
                    reconcile_price_targets(fundamentals.get(ticker), fh.get("price_targets")),
                ]
                if flag
            ]
            payloads[ticker] = {
                "fundamentals": fundamentals.get(ticker) or {},
                "data_reconciliation_flags": reconciliation_flags,
                "technicals": technicals.get(ticker) or {},
                "universe_signals": {
                    "sources": c["sources"],
                    "conviction": c["conviction"],
                },
                "score": c["score"],
                "score_breakdown": c["score_breakdown"],
                "sector_bias": c.get("sector_bias"),
                "market_themes": c.get("themes") or [],
                "earnings_alert": earnings_alerts.get(ticker),
                "insider_activity": insider_activity,
                "earnings_surprise_history": fh.get("earnings_surprise") or [],
                "recommendation_trend": fh.get("recommendation_trend") or [],
                "analyst_price_targets": fh.get("price_targets") or {},
                "eps_revisions": eps_revisions.get(ticker) or {},
                # Contracted, not forecast: revenue already under order.
                "contracted_book": (self.state.get("contracted_book") or {}).get(ticker),
                "share_trades": share_trades.get(ticker),
                "risk_factors_10k": _trim(
                    (risk_factors.get(ticker) or {}).get("risk_factors"),
                    _RISK_FACTORS_CHARS,
                ),
                "quarterly_mda": _trim(
                    (self.state.get("quarterly_mda", {}).get(ticker) or {}).get("mda"),
                    _QUARTERLY_MDA_CHARS,
                ),
                "peers": self.state.get("peer_comparison", {}).get(ticker),
                "earnings_transcript": _trim(
                    (self.state.get("earnings_transcripts", {}).get(ticker) or {}).get("snippet"),
                    _TRANSCRIPT_CHARS,
                ),
                "recent_news": recent_news.get(ticker, []),
                "news": news.get(ticker, []),
            }
            for flag in reconciliation_flags:
                logger.warning("Data reconciliation (%s): %s", ticker, flag)
        return payloads

    def _run_analysts(
        self, survivors: list[dict[str, Any]], payloads: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, Any], list[str]]:
        """Deep (Sonnet) and light (Haiku) tiers under the cost cap, then
        the catalyst repair pass."""
        recent_news = self.state.get("recent_news") or {}
        fallback = (
            self.settings.discover_fallback_provider,
            self.settings.resolve_fallback_model(),
        )
        deep = Analyst("claude", self.settings.discover_sonnet_model, fallback=fallback)
        deep_count = self.settings.discover_analyst_deep_count
        light = (
            Analyst("claude", self.settings.discover_haiku_model, fallback=fallback)
            if deep_count > 0 and self.settings.discover_haiku_model
            else None
        )
        # `survivors` is already in screen-score order.
        deep_tickers = {c["ticker"] for c in survivors[:deep_count]}
        keep, deep_tickers, cuts = plan_under_budget(
            [c["ticker"] for c in survivors],
            {t: len(dumps_pretty(p)) for t, p in payloads.items()},
            deep_tickers,
            self.settings.discover_sonnet_model,
            self.settings.discover_haiku_model if light is not None else None,
            BUDGET.available_for(self.ANALYST_BUDGET_SHARE),
        )
        for cut in cuts:
            BUDGET.note(cut)
        payloads = {t: payloads[t] for t in keep}
        return repair_catalysts(analyze_tiered(deep, light, payloads, deep_tickers), recent_news)

    def step_holdings(self, step_input: StepInput) -> StepOutput:
        try:
            # step_universe already fetched these to build the sampling
            # frame; reuse so the brokerage is hit once per run.
            holdings = self.state.get("holdings_raw")
            if holdings is None:
                holdings = fetch_portfolio_holdings()
            self.state["holdings_summary"] = _holdings_summary(holdings)
            self.state["holdings_table_rows"] = _holdings_table_rows(holdings)
        except Exception as e:
            logger.warning("Could not fetch holdings (%s) — proceeding without", e)
            self.state["holdings_summary"] = ""
            self.state["holdings_table_rows"] = []
        n = self.state["holdings_summary"].count("\n") + 1 if self.state["holdings_summary"] else 0
        return StepOutput(content=f"Holdings: {n} positions" if n else "Holdings: none")

    def step_ranker(self, step_input: StepInput) -> StepOutput:
        analyses = self.state.get("analyses") or {}
        if not analyses:
            self.state["ranker_output"] = None
            self.state["ranker_text"] = ""
            self.state["picks"] = []
            return StepOutput(content="ranker: no analyses; skipping")
        ranker = Ranker(
            self.settings.resolve_ranker_rounds(),
            fallback=(
                self.settings.discover_fallback_provider,
                self.settings.resolve_fallback_model(),
            ),
        )
        output = ranker.rank(
            analyses,
            self.state.get("holdings_summary", ""),
            macro_context=self.state.get("macro_summary", ""),
            track_record_block=self.state.get("track_record_block", ""),
            market_themes_block=self.state.get("market_themes_block", ""),
            calibration_block=self.state.get("calibration_block", ""),
        )
        self.state["ranker_output"] = output
        self.state["ranker_text"] = output.full_text
        self.state["picks"] = parse_picks(output)
        picked = [t for _, t, _ in self.state["picks"]]

        # Pure-arithmetic sanity check — no LLM call — against data the
        # pipeline already fetched. Never blocks the run; only surfaced in
        # the report/log so a human can weigh in on an outlier target.
        technicals = self.state.get("technicals") or {}
        hv_data = self.state.get("historical_volatility") or {}
        warnings: list[str] = []
        for pick in output.picks:
            warnings.extend(
                validate_pick_scenarios(
                    pick,
                    (technicals.get(pick.ticker) or {}).get("price"),
                    hv_data.get(pick.ticker),
                )
            )
        self.state["output_validation_warnings"] = warnings
        for w in warnings:
            logger.warning("Output sanity check: %s", w)

        return StepOutput(content=f"Ranker picked {len(picked)}: {picked}")

    def step_macro_veto(self, step_input: StepInput) -> StepOutput:
        output = self.state.get("ranker_output")
        if output is None:
            return StepOutput(content="macro_veto: no ranker output; skipping")
        trimmed, reasons = apply_macro_veto(
            output,
            self.state.get("macro_data"),
            self.state.get("technicals") or {},
        )
        self.state["ranker_output"] = trimmed
        self.state["ranker_text"] = trimmed.full_text
        self.state["picks"] = parse_picks(trimmed)
        self.state["macro_veto_reasons"] = reasons
        if not reasons:
            return StepOutput(content="macro_veto: no suppressions")
        return StepOutput(content=f"macro_veto: suppressed {len(reasons)} pick(s)")

    def step_redteam(self, step_input: StepInput) -> StepOutput:
        ranker_text = self.state.get("ranker_text") or ""
        if not ranker_text:
            self.state["redteam_output"] = None
            self.state["redteam_text"] = ""
            return StepOutput(content="redteam: no picks; skipping")
        redteam = RedTeam(
            self.settings.discover_redteam_provider,
            self.settings.resolve_redteam_model(),
            fallback=(
                self.settings.discover_fallback_provider,
                self.settings.resolve_fallback_model(),
            ),
        )
        try:
            redteam_output = redteam.critique(ranker_text)
        except Exception as e:
            # Never let a critique failure cost the run its ranker/sizer
            # output — those already-paid-for Opus calls still get
            # persisted and emailed, just without a bear-case section.
            logger.warning("Red-team critique failed (%s) — report will omit bear cases", e)
            self.state["redteam_output"] = None
            self.state["redteam_text"] = ""
            return StepOutput(content="redteam: failed; continuing without bear cases")
        self.state["redteam_output"] = redteam_output
        self.state["redteam_text"] = redteam_output.full_text
        return StepOutput(content="Red-team critique complete")

    def step_sizer(self, step_input: StepInput) -> StepOutput:
        ranker_text = self.state.get("ranker_text") or ""
        if not ranker_text:
            self.state["sizer_output"] = None
            self.state["sizer_text"] = ""
            return StepOutput(content="sizer: no picks; skipping")
        # Build deterministic EV table from the ranker's probability-weighted
        # scenarios — feeds Sizer as primary ranking signal.
        from ...models.llm import RankerOutput

        ev_table = _format_ev_table(self.state.get("ranker_output"))
        agreement_block = _format_agreement_block(self.state.get("ranker_output"))
        ranker_output = self.state.get("ranker_output")
        correlated_pairs = (
            ranker_output.pairs_not_to_hold_together
            if isinstance(ranker_output, RankerOutput)
            else []
        )
        risk_parity_block = _format_risk_parity_block(
            ranker_output, self.state.get("historical_volatility") or {}
        )
        picked = {t for _, t, _ in self.state.get("picks") or []}
        earnings_alerts = {
            t: a for t, a in (self.state.get("earnings_alerts") or {}).items() if t in picked
        }
        earnings_block = "\n".join(
            f"  {t}: reports {a.get('earnings_date')} (in {a.get('days_until')}d)"
            for t, a in sorted(earnings_alerts.items())
        )
        pick_sectors, holdings_by_sector = self._sector_exposure(picked)
        sector_block = format_sector_exposure_block(
            pick_sectors,
            holdings_by_sector,
            max_book_pct=self.settings.discover_max_sector_pct,
            max_new_pct=self.settings.discover_max_sector_new_pct,
        )
        sizer = Sizer(
            "claude",
            self.settings.discover_opus_model,
            fallback=(
                self.settings.discover_fallback_provider,
                self.settings.resolve_fallback_model(),
            ),
        )
        try:
            sizer_output = sizer.allocate(
                ranker_text,
                self.state.get("redteam_text", ""),
                self.state.get("holdings_summary", ""),
                self.settings.discover_cash_budget,
                ev_table=ev_table,
                agreement_block=agreement_block,
                correlated_pairs=correlated_pairs,
                risk_parity_block=risk_parity_block,
                earnings_block=earnings_block,
                sector_block=sector_block,
            )
        except Exception as e:
            # Same rationale as step_redteam: a sizing failure shouldn't
            # discard the ranker's (already-paid-for) picks.
            logger.warning("Sizer failed (%s) — report will omit position sizing", e)
            self.state["sizer_output"] = None
            self.state["sizer_text"] = ""
            return StepOutput(content="sizer: failed; continuing without sizing")
        if correlated_pairs:
            sizer_output = enforce_correlation_caps(
                sizer_output,
                correlated_pairs,
                cash_budget=self.settings.discover_cash_budget,
            )
        if earnings_alerts:
            sizer_output = enforce_earnings_blackout(
                sizer_output,
                earnings_alerts,
                cash_budget=self.settings.discover_cash_budget,
                max_pct=self.settings.discover_earnings_blackout_max_pct,
            )
        if pick_sectors:
            sizer_output = enforce_sector_caps(
                sizer_output,
                pick_sectors,
                holdings_by_sector,
                cash_budget=self.settings.discover_cash_budget,
                max_book_pct=self.settings.discover_max_sector_pct,
                max_new_pct=self.settings.discover_max_sector_new_pct,
            )
        self.state["sizer_output"] = sizer_output
        self.state["sizer_text"] = sizer_output.full_text
        return StepOutput(content="Position sizing complete")

    def _apply_model_scores(
        self, candidates: list[dict[str, Any]], technicals: dict[str, dict[str, Any]]
    ) -> None:
        """Score screen survivors with the latest forward-return model.
        An ACCEPTED model adds up to +/-5 points to the composite; any other
        model is recorded in shadow (score_breakdown only) so it can be
        graded on live runs before it is ever allowed to move a ranking."""
        try:
            model = load_latest_model(self.settings.discover_db_path)
        except Exception as e:
            logger.warning("model load failed (%s) — screening without it", e)
            return
        if model is None:
            return
        passed = [c for c in candidates if c["passed_filter"] and c["score"] is not None]
        feats = {
            c["ticker"]: (technicals.get(c["ticker"]) or {}).get("model_features") or {}
            for c in passed
        }
        pct = score_percentiles(model, {t: f for t, f in feats.items() if f})
        for c in passed:
            if c["ticker"] not in pct:
                continue
            p = round(pct[c["ticker"]], 1)
            meta = {"version": model.version, "percentile": p, "accepted": model.accepted}
            if model.accepted:
                points = screen_points(p)
                c["score"] = round(c["score"] + points, 1)
                c["score_components"] = {**c["score_components"], "model": points}
            c["score_breakdown"] = {**c["score_breakdown"], "model": meta}
        logger.info(
            "Forward-return model v%d %s: scored %d survivors",
            model.version,
            "applied" if model.accepted else "in shadow",
            len(pct),
        )

    def _sector_exposure(self, picked: set[str]) -> tuple[dict[str, str], dict[str, float]]:
        """(pick -> sector, current holdings value per sector) for the sector
        caps. Sectors come from the screen's fundamentals; held names the
        screen never fetched (prescreened out) are looked up once here."""
        sector_of = {
            str(c["ticker"]).upper(): c["sector"]
            for c in self.state.get("candidates") or []
            if c.get("sector")
        }
        holdings = self.state.get("holdings_raw") or {}
        missing = sorted(
            {
                str(h.get("ticker") or "").upper()
                for items in holdings.values()
                for h in items
                if h.get("ticker")
            }
            - set(sector_of)
        )
        if missing:
            try:
                for t, f in batch_fundamentals(missing).items():
                    if (f or {}).get("sector"):
                        sector_of[t.upper()] = f["sector"]
            except Exception as e:
                logger.warning("sector lookup for holdings failed (%s) — cap uses picks only", e)
        pick_sectors = {t: sector_of[t] for t in picked if t in sector_of}
        from ...data.pricing import reconcile_prices

        prices, _ = reconcile_prices(holdings)
        return pick_sectors, _holdings_value_by_sector(holdings, sector_of, prices)
