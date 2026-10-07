"""Discover steps that gather data: universe, screen inputs, the screen, enrichment. No LLM
except the market-themes pass."""

from __future__ import annotations

from datetime import date
from typing import Any

from ...data.brokerage import fetch_portfolio_holdings, listed_tickers
from ...data.earnings_calendar import batch_earnings_flags
from ...data.eps_revisions import batch_eps_revisions
from ...data.filing_evidence import filing_features, red_flags_from
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
    diversify_shortlist,
    passes_hard_filter,
    prescreen,
    rank_within_size_bands,
    score_candidate,
    typical_book_points,
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
from ..pipeline_base import PipelineBase
from .helpers import (
    MAX_CANDIDATES_FOR_LLM,
    _batch_news,
    _flatten_score_breakdown,
    _top_fail_reasons,
    _validate_and_correct_themes,
)

logger = get_logger("stock_analyzer.cli.discover")


def holdings_frame(holdings: dict[str, list[dict[str, Any]]]) -> tuple[str, ...]:
    """The held tickers that belong in the universe frame.

    Holdings rows carry "ticker" (data/brokerage.py); reading "symbol" left
    every holding out of the frame ("holdings 0") until 2026-09-26.
    listed_tickers drops option symbols and dead CUSIPs, and a cash sweep is
    dropped too: holdings skip the trend gate, so SPAXX would take one of the
    analysis slots.
    """
    from ...reporting.health import is_cash_like

    listed, _ = listed_tickers(holdings)
    return tuple(sorted({t.upper() for t in listed if not is_cash_like(t, None)}))


class DataSteps(PipelineBase):
    # --- step executors ------------------------------------------------

    def step_universe(self) -> str:
        # Holdings belong in the sampling frame: a name you already own is
        # always worth re-evaluating. Fetched here (not in step_holdings,
        # which runs later) and cached in state so the brokerage is only
        # called once per run.
        holdings_tickers: tuple[str, ...] = ()
        try:
            holdings = fetch_portfolio_holdings()
            self.state["holdings_raw"] = holdings
            holdings_tickers = holdings_frame(holdings)
        except Exception as e:
            logger.info(
                "Holdings unavailable for the universe frame (%s) — "
                "continuing with index + watchlist only",
                e,
            )

        try:
            from ...discover.earnings_standouts import DISCOVER_DAYS, recent_standouts

            standouts = tuple(
                s["ticker"]
                for s in recent_standouts(self.settings.discover_db_path, days=DISCOVER_DAYS)
            )
        except Exception as e:  # noqa: BLE001 — an idea source, not a requirement
            logger.info("Earnings standouts unavailable for the universe (%s)", e)
            standouts = ()
        try:
            from ...data.insider_buying import clusters

            insider = tuple(
                c["ticker"] for c in clusters(self.settings.discover_db_path, today=date.today())
            )
        except Exception as e:  # noqa: BLE001 — an idea source, not a requirement
            logger.info("Insider clusters unavailable for the universe (%s)", e)
            insider = ()
        try:
            from ..ibd import top_leaders

            leaders = top_leaders(
                self.settings.discover_db_path,
                self.settings.discover_ibd_leaders,
                today=date.today(),
            )
        except Exception as e:  # noqa: BLE001 — an idea source, not a requirement
            logger.info("IBD-style leaders unavailable for the universe (%s)", e)
            leaders = ()
        try:
            from ...reporting.filing_alert import activist_targets

            activists = tuple(activist_targets(self.settings.discover_db_path, today=date.today()))
        except Exception as e:  # noqa: BLE001 — an idea source, not a requirement
            logger.info("Activist 13D targets unavailable for the universe (%s)", e)
            activists = ()
        try:
            from ...data.hedge_funds_13f import consensus_buys

            consensus = tuple(consensus_buys(self.settings.discover_db_path, today=date.today()))
        except Exception as e:  # noqa: BLE001 — an idea source, not a requirement
            logger.info("13F consensus unavailable for the universe (%s)", e)
            consensus = ()
        universe = build_universe(
            watchlist=self.settings.discover_watchlist,
            holdings=holdings_tickers,
            standouts=standouts,
            insider_clusters=insider,
            ibd_leaders=leaders,
            activist_targets=activists,
            fund_consensus=consensus,
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
        return (
            f"Universe: {len(universe)} candidates "
            f"({frame_size} in frame, {len(universe) - frame_size} news-only)"
        )

    def step_technicals(self) -> str:
        tickers = self.state["tickers"]
        self.state["technicals"] = batch_technicals(tickers)
        return f"Technicals: {len(self.state['technicals'])}/{len(tickers)}"

    def step_prescreen(self) -> str:
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

        cap = self.settings.discover_max_screen_candidates
        passed, reasons, n_capped = prescreen(
            tickers, technicals, gate=self.settings.discover_trend_gate, cap=cap, always=always
        )

        self.state["screen_tickers"] = passed
        self.state["prescreen_reasons"] = reasons
        logger.info(
            "Prescreen: %d/%d names pass the trend gate%s — %d go on to "
            "fundamentals + EPS revisions (~%d requests saved)",
            len(passed),
            len(tickers),
            f" (capped at {cap})" if n_capped else "",
            len(passed),
            3 * (len(tickers) - len(passed)),
        )
        return f"Prescreen: {len(passed)}/{len(tickers)} names cleared the trend gate"

    def step_fundamentals(self) -> str:
        tickers = self.state.get("screen_tickers") or self.state["tickers"]
        self.state["fundamentals"] = batch_fundamentals(tickers)
        return f"Fundamentals: {len(self.state['fundamentals'])}/{len(tickers)}"

    def step_historical_volatility(self) -> str:
        # Used post-ranker to sanity-check stated scenario returns against
        # each ticker's own realized volatility (output_validation.py) —
        # not part of the score or any LLM prompt.
        tickers = self.state.get("screen_tickers") or self.state["tickers"]
        self.state["historical_volatility"] = fetch_realized_volatility(tickers)
        return f"Historical volatility: {len(self.state['historical_volatility'])}/{len(tickers)}"

    def step_sector_rotation(self) -> str:
        self.state["sector_rotation"] = sector_rotation_summary(months=6)
        leaders = self.state["sector_rotation"].get("leaders", [])
        laggards = self.state["sector_rotation"].get("laggards", [])
        return f"Sector leaders (6mo): {leaders}; laggards: {laggards}"

    def step_macro_regime(self) -> str:
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
        return self.state["macro_summary"][:200]

    def step_track_record(self) -> str:
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
        return self.state["track_record_summary"]

    def step_thesis_check(self) -> str:
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
            return "thesis check: failed; skipping"
        self.state["thesis_checks"] = thesis_report_data(checks)
        flagged = [f"{c.ticker} {c.status}" for c in checks if c.status != "INTACT"]
        for line in flagged:
            logger.info("Thesis check: %s", line)
        return f"Thesis check: {len(checks)} open picks, {len(flagged)} flagged"

    def step_market_themes(self) -> str:
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
            return "market_themes: detection failed; skipping bias"
        self.state["market_themes_block"] = themes.full_text
        names = [t.name for t in themes.themes]
        return f"Market themes: {len(themes.themes)} detected ({', '.join(names[:5])})"

    def step_screen(self) -> str:
        universe = self.state["universe"]
        fundamentals = self.state["fundamentals"]
        technicals = self.state["technicals"]

        themes_by_t = self.state.get("themes_by_ticker") or {}
        revisions_by_t = self.state.get("eps_revisions") or {}

        prescreen_reasons = self.state.get("prescreen_reasons") or {}
        books = self._screen_books(fundamentals, technicals, prescreen_reasons)
        no_book_points = typical_book_points(books.values())
        filing = filing_features(
            self.settings.discover_db_path, list(self.state["tickers"]), today=date.today()
        )
        self.state["filing_features"] = filing
        flags = red_flags_from(filing)
        if flags:
            logger.info(
                "Screen: SEC filing red flags on %d names (%s)",
                len(flags),
                ", ".join(f"{t} {'/'.join(c)}" for t, c in sorted(flags.items())[:20]),
            )

        candidates = [
            self._screen_candidate(
                ticker,
                fundamentals=fundamentals,
                technicals=technicals,
                universe=universe,
                themes_by_t=themes_by_t,
                revisions_by_t=revisions_by_t,
                books=books,
                no_book_points=no_book_points,
                prescreen_reasons=prescreen_reasons,
                filing_flags=flags,
            )
            for ticker in self.state["tickers"]
        ]
        self._apply_model_scores(candidates, technicals)
        self._apply_evidence_scores(candidates, books)
        passed_all = [c for c in candidates if c["passed_filter"]]
        for c in passed_all:
            f = fundamentals.get(c["ticker"]) or {}
            c["market_cap"], c["industry"] = f.get("market_cap"), f.get("industry")
        # Slots go by percentile within size band, not raw score, and no
        # more than a few per sector and industry: see
        # screen.rank_within_size_bands and screen.diversify_shortlist.
        ranked = rank_within_size_bands(passed_all)
        survivors = diversify_shortlist(ranked, MAX_CANDIDATES_FOR_LLM)
        skipped = [c for c in ranked if c.get("shortlist_skipped")]
        if skipped:
            logger.info(
                "Shortlist: %d skipped for sector/industry balance (%s)",
                len(skipped),
                ", ".join(
                    f"{c['ticker']} ({c.get('industry') or c.get('sector')})" for c in skipped
                ),
            )
        passed = self._log_screen_funnel(candidates, survivors, universe)
        self.state["candidates"] = candidates
        self.state["survivors"] = survivors
        self.state["survivor_tickers"] = [c["ticker"] for c in survivors]

        self._add_similar_setups(survivors)

        if not survivors:
            # Don't raise — a step that raises stops the run, and an empty
            # shortlist is a result, not a failure. Log loudly + return;
            # downstream steps short-circuit on the empty list and the run
            # lands as an honest 0-candidates row in the DB.
            logger.error(
                "Screen: no candidates passed hard filters out of %d "
                "(top fail reasons: %s). Continuing with empty survivors "
                "so downstream steps degrade cleanly.",
                len(candidates),
                _top_fail_reasons(candidates),
            )
            return f"Screen: 0/{len(candidates)} passed — no survivors"
        return f"Screen: {passed}/{len(candidates)} passed; top {len(survivors)} → LLM"

    def _screen_books(
        self,
        fundamentals: dict[str, Any],
        technicals: dict[str, Any],
        prescreen_reasons: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Contracted books for every name that will be scored, since
        book growth is part of the score (screen._score_book). Free SEC
        requests, cached for a week; a failed fetch scores as no book."""
        gate = self.settings.discover_trend_gate
        scored = [
            t
            for t in self.state["tickers"]
            if t not in prescreen_reasons
            and fundamentals.get(t)
            and technicals.get(t)
            and passes_hard_filter(fundamentals.get(t), technicals.get(t), gate)[0]
        ]
        if not scored:
            return {}
        try:
            from ...data.backlog import batch_rpo

            return batch_rpo(scored)
        except Exception as e:  # noqa: BLE001 — a score input, not a dependency
            logger.warning("Contracted-book fetch for the screen failed (%s)", e)
            return {}

    def _screen_candidate(
        self,
        ticker: str,
        *,
        fundamentals: dict[str, Any],
        technicals: dict[str, Any],
        universe: dict[str, Any],
        themes_by_t: dict[str, Any],
        revisions_by_t: dict[str, Any],
        books: dict[str, dict[str, Any]],
        no_book_points: float,
        prescreen_reasons: dict[str, Any],
        filing_flags: dict[str, list[str]] | None = None,
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
                book=books.get(ticker.upper()),
                no_book_points=no_book_points,
                filing_flags=(filing_flags or {}).get(ticker.upper()),
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

    def _apply_evidence_scores(
        self, candidates: list[dict[str, Any]], books: dict[str, dict[str, Any]]
    ) -> None:
        """Record each survivor's evidence score (discover/evidence.py) in
        its score_breakdown. It never moves the ranking and the LLM stages
        never see it: it is the control the pick scorecard grades the
        picks against."""
        from ...data.insider_buying import clusters
        from ...discover.evidence import evidence_scores
        from ...model.sec_history import current_gross_profitability

        passed = [c for c in candidates if c["passed_filter"] and c["score_breakdown"]]
        if not passed:
            return
        tickers = [c["ticker"].upper() for c in passed]
        today = date.today()
        try:
            gp = current_gross_profitability(tickers, self.settings.model_cache_dir, today=today)
        except Exception as e:  # noqa: BLE001 — a benchmark, not a dependency
            logger.warning("Evidence score: gross profitability unavailable (%s)", e)
            gp = {}
        try:
            insiders = {
                c["ticker"].upper() for c in clusters(self.settings.discover_db_path, today=today)
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("Evidence score: insider clusters unavailable (%s)", e)
            insiders = set()
        book_yoy = {
            t: rec["yoy_pct"] for t, rec in books.items() if rec and rec.get("yoy_pct") is not None
        }
        scores = evidence_scores(
            tickers, book_yoy=book_yoy, gross_profitability=gp, insider_clusters=insiders
        )
        for c in passed:
            c["score_breakdown"] = {**c["score_breakdown"], "evidence": scores[c["ticker"].upper()]}
        logger.info(
            "Evidence score: %d survivors (%d with gross profitability, %d with a book, "
            "%d in an insider cluster)",
            len(passed),
            sum(t in gp for t in tickers),
            sum(t in book_yoy for t in tickers),
            sum(t in insiders for t in tickers),
        )

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

    def step_risk_factors(self) -> str:
        tickers = self.state.get("survivor_tickers") or []
        if not tickers:
            self.state["risk_factors"] = {}
            return "risk_factors: no survivors; skipping"
        self.state["risk_factors"] = batch_risk_factors(tickers)
        return f"SEC 10-K: {len(self.state['risk_factors'])}/{len(tickers)}"

    def step_news(self) -> str:
        tickers = self.state.get("survivor_tickers") or []
        if not tickers:
            self.state["news"] = {}
            self.state["recent_news"] = {}
            return "news: no survivors; skipping"
        self.state["news"] = _batch_news(tickers)
        self.state["recent_news"] = self._fetch_recent_news(tickers)
        return f"News fetched for {len(tickers)}"

    def _fetch_recent_news(self, tickers: list[str]) -> dict[str, list[dict[str, Any]]]:
        fundamentals = {
            **(self.state.get("holdings_fundamentals") or {}),
            **(self.state.get("fundamentals") or {}),
        }
        names = {t: (fundamentals.get(t) or {}).get("name") for t in tickers}
        return batch_ticker_news(
            list(tickers), names, days=self.settings.discover_catalyst_news_days
        )

    def step_earnings(self) -> str:
        tickers = self.state.get("survivor_tickers") or []
        if not tickers:
            self.state["earnings_alerts"] = {}
            return "earnings: no survivors; skipping"
        self.state["earnings_alerts"] = batch_earnings_flags(
            tickers, within_days=5, db_path=self.settings.discover_db_path
        )
        return f"Earnings within 5d: {len(self.state['earnings_alerts'])}/{len(tickers)} flagged"

    def step_insider_selling(self) -> str:
        # Kept for back-compat when FINNHUB_API_KEY is unset; the new
        # Finnhub-backed insider activity in `step_finnhub_signals` is
        # strictly richer (real Form 4 filings vs news-mention heuristic).
        tickers = set(self.state.get("survivor_tickers") or [])
        if not tickers:
            self.state["insider_selling"] = {}
            return "insider_selling: no survivors; skipping"
        self.state["insider_selling"] = insider_selling_mentions(tickers, days=14)
        return f"Insider selling: {len(self.state['insider_selling'])} survivors flagged"

    def step_finnhub_signals(self) -> str:
        tickers = list(self.state.get("survivor_tickers") or [])
        if not tickers:
            self.state["finnhub_signals"] = {}
            return "finnhub_signals: no survivors; skipping"
        self.state["finnhub_signals"] = batch_finnhub_signals(tickers)
        n = sum(1 for v in self.state["finnhub_signals"].values() if v)
        return f"Finnhub signals: {n}/{len(tickers)} tickers covered"

    def step_eps_revisions(self) -> str:
        """Analyst EPS-estimate revisions over the last 7 and 30 days.
        One of the strongest forward-thesis signals available.

        Runs before the screen so the score can pick up a +/-5 bonus
        from direction_30d. Fetched for the prescreened set — names the
        trend gate already eliminated cannot pass the hard filter, so
        their revisions would never be read."""
        tickers = list(self.state.get("screen_tickers") or self.state.get("tickers") or [])
        if not tickers:
            self.state["eps_revisions"] = {}
            return "eps_revisions: empty universe; skipping"
        self.state["eps_revisions"] = batch_eps_revisions(tickers)
        raising = sum(
            1 for v in self.state["eps_revisions"].values() if v.get("direction_30d") == "raising"
        )
        lowering = sum(
            1 for v in self.state["eps_revisions"].values() if v.get("direction_30d") == "lowering"
        )
        return (
            f"EPS revisions: {len(self.state['eps_revisions'])}/{len(tickers)} "
            f"covered ({raising} raising, {lowering} lowering, "
            f"rest stable or no coverage)"
        )

    def step_quarterly_mda(self) -> str:
        tickers = self.state.get("survivor_tickers") or []
        if not tickers:
            self.state["quarterly_mda"] = {}
            return "quarterly_mda: no survivors; skipping"
        self.state["quarterly_mda"] = batch_quarterly_mda(tickers)
        return f"10-Q MD&A: {len(self.state['quarterly_mda'])}/{len(tickers)}"

    def step_peer_comparison(self) -> str:
        tickers = self.state.get("survivor_tickers") or []
        if not tickers:
            self.state["peer_comparison"] = {}
            return "peer_comparison: no survivors; skipping"
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
        return f"Peer comparison: {len(self.state['peer_comparison'])}/{len(tickers)}"

    def step_earnings_transcripts(self) -> str:
        tickers = self.state.get("survivor_tickers") or []
        if not tickers:
            self.state["earnings_transcripts"] = {}
            return "earnings_transcripts: no survivors; skipping"
        self.state["earnings_transcripts"] = batch_transcript_snippets(tickers)
        return f"Transcripts: {len(self.state['earnings_transcripts'])}/{len(tickers)}"

    def step_share_trades(self) -> str:
        tickers = self.state.get("survivor_tickers") or []
        if not tickers:
            self.state["share_trades"] = {}
            return "share_trades: no survivors; skipping"
        self.state["share_trades"] = batch_share_trade_data(tickers)
        signals = {
            (data.get("insider_summary_6mo") or {}).get("insider_signal", "neutral")
            for data in self.state["share_trades"].values()
        }
        return (
            f"Share trades fetched for {len(self.state['share_trades'])}"
            f"/{len(tickers)}; signals seen: {sorted(signals)}"
        )

    def step_contracted_book(self) -> str:
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
            return "contracted_book: no survivors"
        try:
            from ...data.backlog import batch_rpo

            books = batch_rpo(survivors)
        except Exception as e:  # noqa: BLE001 — context, not a dependency
            logger.warning("Contracted-book fetch failed (%s) — continuing without it", e)
            books = {}
        self.state["contracted_book"] = books
        growing = sum(1 for b in books.values() if (b.get("yoy_pct") or 0) > 0)
        return (
            f"contracted_book: {len(books)}/{len(survivors)} tag one, "
            f"{growing} growing year-on-year"
        )
