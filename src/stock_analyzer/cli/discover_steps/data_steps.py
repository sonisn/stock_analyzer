"""Discover steps that gather data: universe, screen inputs, the screen, enrichment. No LLM
except the market-themes pass."""

from __future__ import annotations

from typing import Any

from agno.workflow.types import StepInput, StepOutput

from ...data.brokerage import fetch_portfolio_holdings
from ...data.earnings_calendar import batch_earnings_flags
from ...data.eps_revisions import batch_eps_revisions
from ...data.finnhub import batch_finnhub_signals
from ...data.fred_macro import fetch_regime_data, regime_summary_text
from ...data.fundamentals import batch_fundamentals
from ...data.historical_volatility import fetch_realized_volatility
from ...data.insider_selling import insider_selling_mentions
from ...data.sec_edgar import batch_quarterly_mda, batch_risk_factors
from ...data.sector_rotation import sector_bias, sector_rotation_summary
from ...data.share_trades import batch_share_trade_data
from ...data.technical_indicators import batch_technicals
from ...data.ticker_news import batch_ticker_news
from ...data.transcripts import batch_transcript_snippets
from ...discover.calibration import (
    format_calibration_block,
    format_similar_setups_block,
    measure_calibration,
    similar_past_setups,
)
from ...discover.catalyst_grading import format_catalyst_grading_block, grade_catalysts
from ...discover.market_themes import (
    MarketThemesAgent,
    theme_score_bonus,
    themes_by_ticker,
)
from ...discover.paper_ledger import build_ledger, ledger_report_data, load_tranches
from ...discover.peers import batch_peer_comparison
from ...discover.screen import (
    IDEAL_ENTRY_DRAWDOWN,
    passes_hard_filter,
    passes_trend_gate,
    score_candidate,
)
from ...discover.thesis_tracker import check_theses, load_open_picks, thesis_report_data
from ...discover.track_record import (
    format_track_record_block,
    format_track_record_summary,
    measure_track_record,
)
from ...discover.universe import build_universe
from ...logging import get_logger
from ...usage import BudgetExceededError
from .helpers import (
    MAX_CANDIDATES_FOR_LLM,
    _batch_news,
    _flatten_score_breakdown,
    _top_fail_reasons,
    _validate_and_correct_themes,
)

logger = get_logger("stock_analyzer.cli.discover")


class DataSteps:
    # --- step executors ------------------------------------------------

    def step_universe(self, step_input: StepInput) -> StepOutput:
        # Holdings belong in the sampling frame: a name you already own is
        # always worth re-evaluating. Fetched here (not in step_holdings,
        # which runs later) and cached in state so the brokerage is only
        # called once per run.
        holdings_tickers: tuple[str, ...] = ()
        try:
            holdings = fetch_portfolio_holdings()
            self.state["holdings_raw"] = holdings
            holdings_tickers = tuple(
                sorted(
                    {
                        str(item.get("symbol") or "").upper()
                        for items in holdings.values()
                        for item in items
                        if item.get("symbol")
                    }
                )
            )
        except Exception as e:
            logger.info(
                "Holdings unavailable for the universe frame (%s) — "
                "continuing with index + watchlist only",
                e,
            )

        universe = build_universe(
            watchlist=self.settings.discover_watchlist,
            holdings=holdings_tickers,
        )
        if not universe:
            raise RuntimeError(
                "Universe empty — the base universe file, watchlist, holdings "
                "and news feeds all came back empty. Check the bundled "
                "S&P 500 snapshot (or DISCOVER_UNIVERSE_FILE), "
                "DISCOVER_WATCHLIST, and TAVILY_API_KEY."
            )
        self.state["universe"] = universe
        self.state["tickers"] = list(universe.keys())
        frame_size = sum(1 for d in universe.values() if d.get("in_base_universe"))
        return StepOutput(
            content=(
                f"Universe: {len(universe)} candidates "
                f"({frame_size} in frame, {len(universe) - frame_size} news-only)"
            )
        )

    def step_technicals(self, step_input: StepInput) -> StepOutput:
        tickers = self.state["tickers"]
        self.state["technicals"] = batch_technicals(tickers)
        return StepOutput(content=f"Technicals: {len(self.state['technicals'])}/{len(tickers)}")

    def step_prescreen(self, step_input: StepInput) -> StepOutput:
        """Narrow the frame to names that can still pass the hard filter.

        Technicals cost one request per ticker; fundamentals and EPS
        revisions cost three more. The hard filter's trend rules are
        decidable from the technicals alone, so anything that fails them
        is eliminated before the expensive fetches — which is both the
        "stricter criteria" the funnel needed and the bulk of the
        rate-limit pressure removed.
        """
        tickers: list[str] = self.state["tickers"]
        technicals = self.state.get("technicals") or {}
        universe = self.state.get("universe") or {}

        # The user's own names are analyzed regardless of trend: a holding
        # that broke down is exactly the one worth a sell/trim opinion.
        always: set[str] = {
            t
            for t in tickers
            if {"holding", "watchlist"} & set(universe.get(t, {}).get("sources") or [])
        }

        reasons: dict[str, list[str]] = {}
        passed: list[str] = []
        for ticker in tickers:
            ok, why = passes_trend_gate(technicals.get(ticker), self.settings.discover_trend_gate)
            if ok or ticker in always:
                passed.append(ticker)
            else:
                reasons[ticker] = why

        # Cap the survivors, keeping the user's names outside the cap. The
        # strict gate ranks by 6-month relative strength; the soft gate by
        # closeness to the screen's ideal entry (10% below the high), so
        # the cap doesn't quietly reintroduce a momentum filter.
        cap = self.settings.discover_max_screen_candidates
        capped_out: list[str] = []
        if len(passed) > cap:
            if self.settings.discover_trend_gate == "strict":

                def rank_key(t: str) -> float:
                    return (technicals.get(t) or {}).get("rs_6mo") or 0.0
            else:

                def rank_key(t: str) -> float:
                    dist = (technicals.get(t) or {}).get("dist_from_52w_high")
                    return -abs(dist - IDEAL_ENTRY_DRAWDOWN) if dist is not None else -1.0

            ranked = sorted((t for t in passed if t not in always), key=rank_key, reverse=True)
            keep = set(ranked[: max(0, cap - len(always))]) | always
            capped_out = [t for t in passed if t not in keep]
            for ticker in capped_out:
                reasons[ticker] = ["outside the screen cap for deep analysis"]
            passed = [t for t in passed if t in keep]

        self.state["screen_tickers"] = passed
        self.state["prescreen_reasons"] = reasons
        logger.info(
            "Prescreen: %d/%d names pass the trend gate%s — %d go on to "
            "fundamentals + EPS revisions (~%d requests saved)",
            len(passed),
            len(tickers),
            f" (capped at {cap})" if capped_out else "",
            len(passed),
            3 * (len(tickers) - len(passed)),
        )
        return StepOutput(
            content=f"Prescreen: {len(passed)}/{len(tickers)} names cleared the trend gate"
        )

    def step_fundamentals(self, step_input: StepInput) -> StepOutput:
        tickers = self.state.get("screen_tickers") or self.state["tickers"]
        self.state["fundamentals"] = batch_fundamentals(tickers)
        return StepOutput(content=f"Fundamentals: {len(self.state['fundamentals'])}/{len(tickers)}")

    def step_historical_volatility(self, step_input: StepInput) -> StepOutput:
        # Used post-ranker to sanity-check stated scenario returns against
        # each ticker's own realized volatility (output_validation.py) —
        # not part of the score or any LLM prompt.
        tickers = self.state.get("screen_tickers") or self.state["tickers"]
        self.state["historical_volatility"] = fetch_realized_volatility(tickers)
        return StepOutput(
            content=f"Historical volatility: {len(self.state['historical_volatility'])}/{len(tickers)}"
        )

    def step_sector_rotation(self, step_input: StepInput) -> StepOutput:
        self.state["sector_rotation"] = sector_rotation_summary(months=6)
        leaders = self.state["sector_rotation"].get("leaders", [])
        laggards = self.state["sector_rotation"].get("laggards", [])
        return StepOutput(content=f"Sector leaders (6mo): {leaders}; laggards: {laggards}")

    def step_macro_regime(self, step_input: StepInput) -> StepOutput:
        data = fetch_regime_data(self.settings.fred_api_key)
        self.state["macro_data"] = data
        summary = regime_summary_text(data)
        # FRED describes the US only. Semiconductor demand is priced in
        # Taipei and Seoul overnight, and the dollar decides what foreign
        # revenue is worth, so the Ranker sees those too.
        try:
            from ...data.world_markets import fetch_world_markets, world_markets_text

            rows = fetch_world_markets()
            self.state["world_markets"] = rows
            if rows:
                held = set(self.state.get("holdings_tickers") or [])
                summary = f"{summary}\n\n{world_markets_text(rows, held)}"
        except Exception as e:  # noqa: BLE001 — context, not a dependency
            logger.warning("World markets unavailable (%s) — US macro only", e)
        self.state["macro_summary"] = summary
        # Truncate for terminal preview.
        return StepOutput(content=self.state["macro_summary"][:200])

    def step_track_record(self, step_input: StepInput) -> StepOutput:
        record = measure_track_record(self.settings.discover_db_path)
        self.state["track_record"] = record
        self.state["track_record_summary"] = format_track_record_summary(record)
        self.state["track_record_block"] = format_track_record_block(record)

        # Calibration is a separate read over the same DB: the track record
        # says whether the picks worked, calibration says whether the
        # ranker's own stated confidence and EV meant anything. Both go into
        # the ranker prompt; a failure here must not abort a run.
        try:
            calibration = measure_calibration(self.settings.discover_db_path)
            self.state["calibration"] = calibration
            self.state["calibration_block"] = format_calibration_block(calibration)
        except Exception as e:
            logger.warning("calibration pass failed (%s) — continuing without", e)
            self.state["calibration"] = None
            self.state["calibration_block"] = ""
        try:
            catalyst_block = format_catalyst_grading_block(
                grade_catalysts(self.settings.discover_db_path)
            )
            self.state["calibration_block"] = "\n\n".join(
                b for b in (self.state["calibration_block"], catalyst_block) if b
            )
        except Exception as e:
            logger.warning("catalyst grading failed (%s) — continuing without", e)
        try:
            self.state["paper_ledger"] = ledger_report_data(
                build_ledger(load_tranches(self.settings.discover_db_path))
            )
        except Exception as e:
            logger.warning("paper ledger failed (%s) — report will omit it", e)
            self.state["paper_ledger"] = None
        return StepOutput(content=self.state["track_record_summary"])

    def step_thesis_check(self, step_input: StepInput) -> StepOutput:
        """Re-check every recent pick's thesis (no LLM). Runs after the
        screen so this run's EPS revisions are available."""
        try:
            checks = check_theses(
                load_open_picks(self.settings.discover_db_path),
                eps_revisions=self.state.get("eps_revisions") or {},
            )
        except Exception as e:
            logger.warning("thesis check failed (%s) — report will omit it", e)
            self.state["thesis_checks"] = []
            return StepOutput(content="thesis check: failed; skipping")
        self.state["thesis_checks"] = thesis_report_data(checks)
        flagged = [f"{c.ticker} {c.status}" for c in checks if c.status != "INTACT"]
        for line in flagged:
            logger.info("Thesis check: %s", line)
        return StepOutput(content=f"Thesis check: {len(checks)} open picks, {len(flagged)} flagged")

    def step_market_themes(self, step_input: StepInput) -> StepOutput:
        """Detect 3-8 named market themes that are visible in the
        universe's actual price action + EPS revisions. Grounded in
        real data (top/bottom performers, revision direction) rather
        than the LLM's training memory."""
        agent = MarketThemesAgent(
            "claude",
            self.settings.discover_sonnet_model,
            fallback=(
                self.settings.discover_fallback_provider,
                self.settings.resolve_fallback_model(),
            ),
        )
        try:
            themes = agent.detect(
                macro_summary=self.state.get("macro_summary", ""),
                sector_rotation=self.state.get("sector_rotation"),
                technicals=self.state.get("technicals", {}),
                fundamentals=self.state.get("fundamentals", {}),
                eps_revisions=self.state.get("eps_revisions", {}),
            )
        except BudgetExceededError:
            themes = None
        # Anti-hallucination pass: filter unknown tickers + recompute
        # strength against the actual cohort relative-strength data.
        themes = _validate_and_correct_themes(
            themes,
            universe_tickers=set(self.state.get("tickers") or []),
            technicals=self.state.get("technicals", {}),
        )
        self.state["market_themes"] = themes
        self.state["themes_by_ticker"] = themes_by_ticker(themes)
        if themes is None:
            self.state["market_themes_block"] = ""
            return StepOutput(content="market_themes: detection failed; skipping bias")
        self.state["market_themes_block"] = themes.full_text
        names = [t.name for t in themes.themes]
        return StepOutput(
            content=f"Market themes: {len(themes.themes)} detected ({', '.join(names[:5])})"
        )

    def step_screen(self, step_input: StepInput) -> StepOutput:
        universe = self.state["universe"]
        fundamentals = self.state["fundamentals"]
        technicals = self.state["technicals"]

        themes_by_t = self.state.get("themes_by_ticker") or {}
        revisions_by_t = self.state.get("eps_revisions") or {}

        prescreen_reasons = self.state.get("prescreen_reasons") or {}

        candidates = [
            self._screen_candidate(
                ticker,
                fundamentals=fundamentals,
                technicals=technicals,
                universe=universe,
                themes_by_t=themes_by_t,
                revisions_by_t=revisions_by_t,
                prescreen_reasons=prescreen_reasons,
            )
            for ticker in self.state["tickers"]
        ]
        self._apply_model_scores(candidates, technicals)
        survivors = sorted(
            [c for c in candidates if c["passed_filter"]],
            key=lambda c: c["score"] or 0,
            reverse=True,
        )[:MAX_CANDIDATES_FOR_LLM]
        passed = self._log_screen_funnel(candidates, survivors, universe)
        self.state["candidates"] = candidates
        self.state["survivors"] = survivors
        self.state["survivor_tickers"] = [c["ticker"] for c in survivors]

        self._add_similar_setups(survivors)

        if not survivors:
            # Don't raise — agno doesn't propagate state from a step that
            # raises, which leaves every enrichment step in the next
            # parallel block reading a missing survivor_tickers key and
            # cascading 4 retry attempts × 10 steps of KeyError noise.
            # Log loudly + return so state is preserved; downstream
            # steps short-circuit on the empty list and the run lands as
            # an honest 0-candidates row in the DB.
            logger.error(
                "Screen: no candidates passed hard filters out of %d "
                "(top fail reasons: %s). Continuing with empty survivors "
                "so downstream steps degrade cleanly.",
                len(candidates),
                _top_fail_reasons(candidates),
            )
            return StepOutput(content=f"Screen: 0/{len(candidates)} passed — no survivors")
        return StepOutput(
            content=f"Screen: {passed}/{len(candidates)} passed; top {len(survivors)} → LLM"
        )

    def _screen_candidate(
        self,
        ticker: str,
        *,
        fundamentals: dict[str, Any],
        technicals: dict[str, Any],
        universe: dict[str, Any],
        themes_by_t: dict[str, Any],
        revisions_by_t: dict[str, Any],
        prescreen_reasons: dict[str, Any],
    ) -> dict[str, Any]:
        """One ticker through the hard filter and, if it passes, the score."""
        f = fundamentals.get(ticker)
        t = technicals.get(ticker)
        u = universe[ticker]
        if ticker in prescreen_reasons:
            # Eliminated before the fundamentals fetch — report why it
            # actually failed rather than "no fundamentals data".
            passes, reasons = False, list(prescreen_reasons[ticker])
        else:
            passes, reasons = passes_hard_filter(f, t, self.settings.discover_trend_gate)
        cand: dict[str, Any] = {
            "ticker": ticker,
            "passed_filter": passes,
            "fail_reasons": reasons,
            "sources": u["sources"],
            "conviction": u["conviction"],
            "sector": (f or {}).get("sector"),
            "price": (t or {}).get("price"),
            "score": None,
            "score_components": None,
            "score_breakdown": None,
            "themes": [m["name"] for m in (themes_by_t.get(ticker.upper()) or [])],
        }
        if passes and f and t:
            scored = score_candidate(
                f,
                t,
                u,
                revisions=revisions_by_t.get(ticker),
            )
            bonus, theme_meta = theme_score_bonus(ticker, themes_by_t)
            cand["score"] = round(scored["score"] + bonus, 1)
            cand["score_components"] = {
                **scored["components"],
                "theme_bonus": bonus,
            }
            cand["score_breakdown"] = {
                **scored["breakdown"],
                "theme": theme_meta,
            }
        cand["sector_bias"] = sector_bias(cand["sector"], self.state.get("sector_rotation", {}))

        return cand

    def _log_screen_funnel(
        self,
        candidates: list[dict[str, Any]],
        survivors: list[dict[str, Any]],
        universe: dict[str, Any],
    ) -> int:
        """Log the screen funnel; returns how many passed the hard filter."""
        passed = sum(1 for c in candidates if c["passed_filter"])
        # Funnel visibility: the hard filter is a momentum style bet, so a
        # thin tape can collapse the survivor set to a handful. Logged
        # explicitly because "top 5 of 6" is not a comparison, and the
        # ranker's prompt ("pick the top N from M candidates") reads the
        # same either way.
        frame_n = sum(
            1 for t in self.state["tickers"] if universe.get(t, {}).get("in_base_universe")
        )
        logger.info(
            "Screen funnel: %d universe (%d in frame) -> %d cleared the trend "
            "gate and were fully fetched -> %d passed hard filters -> %d sent "
            "to the LLM stages (cap %d)",
            len(candidates),
            frame_n,
            len(self.state.get("screen_tickers") or self.state["tickers"]),
            passed,
            len(survivors),
            MAX_CANDIDATES_FOR_LLM,
        )
        if 0 < passed < 10:
            logger.warning(
                "Only %d candidate(s) passed the hard filter. The 'top picks' "
                "are nearly the whole surviving set, so treat the ranking as "
                "weak discrimination rather than selection.",
                passed,
            )
        return passed

    def _add_similar_setups(self, survivors: list[dict[str, Any]]) -> None:
        # Factor-similarity few-shot retrieval: for the handful of
        # top-scored survivors, find past candidates with a similar
        # score_breakdown and what they actually did. Appended onto the
        # aggregate calibration_block computed earlier (step_track_record)
        # so the ranker sees both "how has my calibration been overall"
        # and "here's a concrete precedent for this specific setup".
        # Capped at 5 survivors — each lookup does a yfinance forward-
        # return fetch per neighbor, so this isn't free.
        try:
            top_for_similarity = sorted(survivors, key=lambda c: c["score"], reverse=True)[:5]
            similar_lines = [
                line
                for c in top_for_similarity
                if (
                    line := format_similar_setups_block(
                        c["ticker"],
                        similar_past_setups(
                            _flatten_score_breakdown(c["score_components"], c["score_breakdown"]),
                            self.settings.discover_db_path,
                        ),
                    )
                )
            ]
            if similar_lines:
                existing = self.state.get("calibration_block", "")
                similar_block = (
                    "Similar past setups (factor-nearest, with what happened):\n"
                    + "\n".join(similar_lines)
                )
                self.state["calibration_block"] = (
                    f"{existing}\n\n{similar_block}" if existing else similar_block
                )
        except Exception as e:
            logger.warning("similar-past-setups lookup failed (%s) — continuing without", e)

    def step_risk_factors(self, step_input: StepInput) -> StepOutput:
        tickers = self.state.get("survivor_tickers") or []
        if not tickers:
            self.state["risk_factors"] = {}
            return StepOutput(content="risk_factors: no survivors; skipping")
        self.state["risk_factors"] = batch_risk_factors(tickers)
        return StepOutput(content=f"SEC 10-K: {len(self.state['risk_factors'])}/{len(tickers)}")

    def step_news(self, step_input: StepInput) -> StepOutput:
        tickers = self.state.get("survivor_tickers") or []
        if not tickers:
            self.state["news"] = {}
            self.state["recent_news"] = {}
            return StepOutput(content="news: no survivors; skipping")
        self.state["news"] = _batch_news(tickers)
        self.state["recent_news"] = self._fetch_recent_news(tickers)
        return StepOutput(content=f"News fetched for {len(tickers)}")

    def _fetch_recent_news(self, tickers: list[str]) -> dict[str, list[dict[str, Any]]]:
        fundamentals = {
            **(self.state.get("holdings_fundamentals") or {}),
            **(self.state.get("fundamentals") or {}),
        }
        names = {t: (fundamentals.get(t) or {}).get("name") for t in tickers}
        return batch_ticker_news(
            list(tickers), names, days=self.settings.discover_catalyst_news_days
        )

    def step_earnings(self, step_input: StepInput) -> StepOutput:
        tickers = self.state.get("survivor_tickers") or []
        if not tickers:
            self.state["earnings_alerts"] = {}
            return StepOutput(content="earnings: no survivors; skipping")
        self.state["earnings_alerts"] = batch_earnings_flags(
            tickers, within_days=5, db_path=self.settings.discover_db_path
        )
        return StepOutput(
            content=(
                f"Earnings within 5d: {len(self.state['earnings_alerts'])}/{len(tickers)} flagged"
            )
        )

    def step_insider_selling(self, step_input: StepInput) -> StepOutput:
        # Kept for back-compat when FINNHUB_API_KEY is unset; the new
        # Finnhub-backed insider activity in `step_finnhub_signals` is
        # strictly richer (real Form 4 filings vs news-mention heuristic).
        tickers = set(self.state.get("survivor_tickers") or [])
        if not tickers:
            self.state["insider_selling"] = {}
            return StepOutput(content="insider_selling: no survivors; skipping")
        self.state["insider_selling"] = insider_selling_mentions(tickers, days=14)
        return StepOutput(
            content=f"Insider selling: {len(self.state['insider_selling'])} survivors flagged"
        )

    def step_finnhub_signals(self, step_input: StepInput) -> StepOutput:
        tickers = list(self.state.get("survivor_tickers") or [])
        if not tickers:
            self.state["finnhub_signals"] = {}
            return StepOutput(content="finnhub_signals: no survivors; skipping")
        self.state["finnhub_signals"] = batch_finnhub_signals(tickers)
        n = sum(1 for v in self.state["finnhub_signals"].values() if v)
        return StepOutput(content=f"Finnhub signals: {n}/{len(tickers)} tickers covered")

    def step_eps_revisions(self, step_input: StepInput) -> StepOutput:
        """Analyst EPS-estimate revisions over the last 7 and 30 days.
        One of the strongest forward-thesis signals available.

        Runs before the screen so the score can pick up a +/-5 bonus
        from direction_30d. Fetched for the prescreened set — names the
        trend gate already eliminated cannot pass the hard filter, so
        their revisions would never be read."""
        tickers = list(self.state.get("screen_tickers") or self.state.get("tickers") or [])
        if not tickers:
            self.state["eps_revisions"] = {}
            return StepOutput(content="eps_revisions: empty universe; skipping")
        self.state["eps_revisions"] = batch_eps_revisions(tickers)
        raising = sum(
            1 for v in self.state["eps_revisions"].values() if v.get("direction_30d") == "raising"
        )
        lowering = sum(
            1 for v in self.state["eps_revisions"].values() if v.get("direction_30d") == "lowering"
        )
        return StepOutput(
            content=(
                f"EPS revisions: {len(self.state['eps_revisions'])}/{len(tickers)} "
                f"covered ({raising} raising, {lowering} lowering, "
                f"rest stable or no coverage)"
            )
        )

    def step_quarterly_mda(self, step_input: StepInput) -> StepOutput:
        tickers = self.state.get("survivor_tickers") or []
        if not tickers:
            self.state["quarterly_mda"] = {}
            return StepOutput(content="quarterly_mda: no survivors; skipping")
        self.state["quarterly_mda"] = batch_quarterly_mda(tickers)
        return StepOutput(content=f"10-Q MD&A: {len(self.state['quarterly_mda'])}/{len(tickers)}")

    def step_peer_comparison(self, step_input: StepInput) -> StepOutput:
        tickers = self.state.get("survivor_tickers") or []
        if not tickers:
            self.state["peer_comparison"] = {}
            return StepOutput(content="peer_comparison: no survivors; skipping")
        fundamentals = self.state.get("fundamentals", {})
        target_meta = {
            t: {
                "name": (fundamentals.get(t) or {}).get("name"),
                "sector": (fundamentals.get(t) or {}).get("sector"),
            }
            for t in tickers
        }
        self.state["peer_comparison"] = batch_peer_comparison(
            tickers,
            target_meta,
            fallback=(
                self.settings.discover_fallback_provider,
                self.settings.resolve_fallback_model(),
            ),
        )
        return StepOutput(
            content=f"Peer comparison: {len(self.state['peer_comparison'])}/{len(tickers)}"
        )

    def step_earnings_transcripts(self, step_input: StepInput) -> StepOutput:
        tickers = self.state.get("survivor_tickers") or []
        if not tickers:
            self.state["earnings_transcripts"] = {}
            return StepOutput(content="earnings_transcripts: no survivors; skipping")
        self.state["earnings_transcripts"] = batch_transcript_snippets(tickers)
        return StepOutput(
            content=f"Transcripts: {len(self.state['earnings_transcripts'])}/{len(tickers)}"
        )

    def step_share_trades(self, step_input: StepInput) -> StepOutput:
        tickers = self.state.get("survivor_tickers") or []
        if not tickers:
            self.state["share_trades"] = {}
            return StepOutput(content="share_trades: no survivors; skipping")
        self.state["share_trades"] = batch_share_trade_data(tickers)
        signals = {
            (data.get("insider_summary_6mo") or {}).get("insider_signal", "neutral")
            for data in self.state["share_trades"].values()
        }
        return StepOutput(
            content=(
                f"Share trades fetched for {len(self.state['share_trades'])}"
                f"/{len(tickers)}; signals seen: {sorted(signals)}"
            )
        )

    def step_contracted_book(self, step_input: StepInput) -> StepOutput:
        """Signed orders each survivor has not delivered yet (data/backlog).

        The only forward number in the payload that is not a forecast:
        analyst targets, forward P/E and EPS revisions are all opinions
        about the future, while remaining performance obligations are
        contracts already signed. Free, from the SEC's XBRL API.

        Roughly half of any universe never tags the concept, so this is
        evidence where present and never a filter — a name without a book
        is not penalized for it.
        """
        survivors = [c["ticker"] for c in (self.state.get("survivors") or [])]
        if not survivors:
            self.state["contracted_book"] = {}
            return StepOutput(content="contracted_book: no survivors")
        try:
            from ...data.backlog import batch_rpo

            books = batch_rpo(survivors)
        except Exception as e:  # noqa: BLE001 — context, not a dependency
            logger.warning("Contracted-book fetch failed (%s) — continuing without it", e)
            books = {}
        self.state["contracted_book"] = books
        growing = sum(1 for b in books.values() if (b.get("yoy_pct") or 0) > 0)
        return StepOutput(
            content=(
                f"contracted_book: {len(books)}/{len(survivors)} tag one, "
                f"{growing} growing year-on-year"
            )
        )
