"""Stock discovery pipeline — Agno Workflow version.

Run via:   python -m stock_analyzer.cli.discover

Declarative shape:
  universe
  ├ Parallel(fundamentals, technicals)
  screen
  ├ Parallel(risk_factors, news)
  analyst (Sonnet, parallel fan-out inside step)
  holdings
  ranker (multi-provider consensus — one round per DISCOVER_RANKER_PROVIDERS entry)
  macro_veto (deterministic, suppresses high-momentum picks in a risk-off regime)
  redteam (DISCOVER_REDTEAM_PROVIDER, default a different provider than the ranker's)
  sizer (Opus)
  persist_and_report

Workflow's SqliteDb logs every run (per-step input/output/timing) into the
SAME discover.db file we use for domain tables. Single source of truth.

State is shared across steps via a DiscoverPipeline instance — each step is a
bound method that reads/writes self.state. Cleaner than threading dicts
through StepOutput.content.
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
from ..data.brokerage import fetch_portfolio_holdings
from ..data.chart_img import fetch_charts
from ..data.earnings_calendar import batch_earnings_flags
from ..data.eps_revisions import batch_eps_revisions
from ..data.finnhub import batch_finnhub_signals
from ..data.fred_macro import fetch_regime_data, regime_summary_text
from ..data.fundamentals import batch_fundamentals
from ..data.historical_volatility import fetch_realized_volatility
from ..data.insider_selling import insider_selling_mentions
from ..data.sec_edgar import batch_quarterly_mda, batch_risk_factors
from ..data.sector_rotation import sector_bias, sector_rotation_summary
from ..data.share_trades import batch_share_trade_data
from ..data.technical_indicators import batch_technicals
from ..data.ticker_news import batch_ticker_news
from ..data.transcripts import batch_transcript_snippets
from ..db.repository import (
    insert_candidate,
    insert_pick,
    insert_pick_catalysts,
    insert_run,
    insert_run_outputs,
    insert_scorecard,
)
from ..db.session import get_session
from ..discover.analyst import Analyst, analyze_tiered
from ..discover.calibration import (
    format_calibration_block,
    format_similar_setups_block,
    measure_calibration,
    similar_past_setups,
)
from ..discover.catalyst_grading import format_catalyst_grading_block, grade_catalysts
from ..discover.catalysts import catalysts_to_dicts, repair_catalysts
from ..discover.data_reconciliation import reconcile_price_targets
from ..discover.factor_tilt import average_factor_tilts, compute_factor_tilt
from ..discover.macro_filter import apply_macro_veto
from ..discover.market_themes import (
    MarketThemesAgent,
    theme_score_bonus,
    themes_by_ticker,
)
from ..discover.output_validation import validate_pick_scenarios
from ..discover.paper_ledger import build_ledger, ledger_report_data, load_tranches
from ..discover.peers import batch_peer_comparison
from ..discover.ranker import Ranker
from ..discover.redteam import RedTeam
from ..discover.report import (
    build_sections,
    parse_picks,
    print_terminal_summary,
    render_html_email,
    render_pdf,
)
from ..discover.screen import passes_hard_filter, passes_trend_gate, score_candidate
from ..discover.sizer import (
    Sizer,
    enforce_correlation_caps,
    enforce_earnings_blackout,
    enforce_sector_caps,
    format_sector_exposure_block,
)
from ..discover.track_record import (
    format_track_record_block,
    format_track_record_summary,
    measure_track_record,
)
from ..discover.universe import build_universe
from ..logging import current_log_file, get_logger
from ..preflight import PreflightError, preflight
from ..reporting.smtp import SmtpServer
from ..usage import TRACKER, log_usage_summary

logger = get_logger(__name__)


def _save_local_pdf(pdf_bytes: bytes, filename: str) -> Path:
    """Persist the PDF to ~/.stock_analyzer/reports/ so a missed email
    never costs the user the report. Override via REPORTS_DIR env."""
    reports_dir = Path(os.path.expanduser(os.getenv("REPORTS_DIR", "~/.stock_analyzer/reports")))
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / filename
    path.write_bytes(pdf_bytes)
    return path


def _pick_forecasts(ranker_output: object) -> dict[str, dict[str, Any]]:
    """Per-ticker forecast extracted from the structured ranker output.

    Returns {ticker: {conviction, ev_pct, time_horizon, scenarios,
    agreement_ratio, voting_providers}} where `scenarios` is a list of
    plain dicts ready for the repository layer. EV is computed by the same
    deterministic helper the report and Sizer use, so the stored number is
    exactly the one the pipeline acted on. `agreement_ratio`/
    `voting_providers` are the Ranker's multi-provider consensus vote for
    this pick (None on single-round runs).
    """
    from ..models.llm import RankerOutput, expected_return_pct

    if not isinstance(ranker_output, RankerOutput):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for pick in ranker_output.picks:
        out[pick.ticker] = {
            "conviction": pick.conviction,
            "ev_pct": expected_return_pct(pick),
            "time_horizon": pick.time_horizon,
            "scenarios": [
                {
                    "label": s.label,
                    "probability": s.probability,
                    "target_return_pct": s.target_return_pct,
                }
                for s in pick.scenarios
            ],
            "agreement_ratio": pick.agreement_ratio,
            "voting_providers": pick.voting_providers,
        }
    return out


def _format_ev_table(ranker_output: object) -> str:
    """Build the expected-return table the Sizer reads as ranking signal.

    Format per row: ticker | E[ret] | bull P/ret | base P/ret | bear P/ret.
    Bear probabilities surfaced explicitly so the Sizer can see when a
    high-EV pick has high dispersion."""
    from ..models.llm import RankerOutput, expected_return_pct

    if not isinstance(ranker_output, RankerOutput):
        return ""
    rows: list[str] = []
    for pick in sorted(ranker_output.picks, key=lambda p: p.rank):
        ev = expected_return_pct(pick)
        if ev is None or not pick.scenarios:
            rows.append(f"  {pick.ticker:6s}  conviction={pick.conviction}  (no scenarios)")
            continue
        sc_map = {s.label: s for s in pick.scenarios}
        bull = sc_map.get("bull")
        base = sc_map.get("base")
        bear = sc_map.get("bear")
        bull_s = f"{bull.probability:.0%}/{bull.target_return_pct:+.0f}%" if bull else "—"
        base_s = f"{base.probability:.0%}/{base.target_return_pct:+.0f}%" if base else "—"
        bear_s = f"{bear.probability:.0%}/{bear.target_return_pct:+.0f}%" if bear else "—"
        rows.append(
            f"  {pick.ticker:6s}  E[ret]={ev:+5.1f}%  "
            f"bull {bull_s:>10s}  base {base_s:>10s}  bear {bear_s:>10s}  "
            f"(conv {pick.conviction})"
        )
    if not rows:
        return ""
    header = "  Ticker  E[return]    Bull P/Ret    Base P/Ret    Bear P/Ret"
    return header + "\n" + "\n".join(rows)


def _format_agreement_block(ranker_output: object) -> str:
    """Per-ticker consensus agreement ratio, for the Sizer's prompt.

    Empty when the ranker ran a single round (agreement_ratio is None on
    every pick in that case) — single-round runs have no agreement signal."""
    from ..models.llm import RankerOutput

    if not isinstance(ranker_output, RankerOutput):
        return ""
    rows: list[str] = []
    for pick in sorted(ranker_output.picks, key=lambda p: p.rank):
        if pick.agreement_ratio is None:
            continue
        n = len(pick.voting_providers) if pick.voting_providers else 0
        total = round(n / pick.agreement_ratio) if pick.agreement_ratio else n
        providers = ", ".join(pick.voting_providers or [])
        rows.append(f"  {pick.ticker:6s}  {n}/{total} rounds agreed ({providers})")
    return "\n".join(rows)


def _flatten_score_breakdown(
    components: dict[str, Any] | None, breakdown: dict[str, Any] | None
) -> dict[str, float]:
    """Every numeric sub-score as a flat {name: value} map, matching the
    key convention `score_validation.py::_flatten_components` uses for
    stored past candidates ("total.<group>" for group totals, "<group>.
    <leaf>" for individual sub-scores) — so `similar_past_setups()`'s
    nearest-neighbor distance is comparing like-shaped dicts instead of a
    flat dict (past) against a nested one (today's candidate)."""
    out: dict[str, float] = {}
    for key, value in (components or {}).items():
        if isinstance(value, int | float) and not isinstance(value, bool):
            out[f"total.{key}"] = float(value)
    for group, leaves in (breakdown or {}).items():
        if not isinstance(leaves, dict):
            continue
        for key, value in leaves.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                out[f"{group}.{key}"] = float(value)
    return out


def _format_risk_parity_block(ranker_output: object, hv_data: dict[str, Any]) -> str:
    """Inverse-volatility (risk-parity) weight per pick, as a third sizing
    input alongside EV/conviction — high-volatility picks should carry
    less weight than EV alone implies, for equal risk contribution.

    Skips picks with missing/zero HV. Empty string if fewer than 2 picks
    have usable HV (nothing to normalize against)."""
    from ..models.llm import RankerOutput

    if not isinstance(ranker_output, RankerOutput):
        return ""
    inv_vols: dict[str, float] = {}
    for pick in ranker_output.picks:
        hv = hv_data.get(pick.ticker)
        if hv is None or not hv.hv_annualized:
            continue
        inv_vols[pick.ticker] = 1.0 / hv.hv_annualized
    if len(inv_vols) < 2:
        return ""
    total = sum(inv_vols.values())
    rows = [
        f"  {ticker:6s}  HV={hv_data[ticker].hv_annualized:.0%}  "
        f"risk-parity weight={inv_v / total:.0%}"
        for ticker, inv_v in inv_vols.items()
    ]
    header = "  Ticker  HV        Risk-parity weight (equal risk contribution)"
    return header + "\n" + "\n".join(rows)


def _validate_and_correct_themes(
    themes: object,
    *,
    universe_tickers: set[str],
    technicals: dict[str, Any],
) -> object:
    """Anti-hallucination pass on the MarketThemes output.

    1. Drop member_tickers that aren't in the universe (LLM may invent
       tickers or pull them from training cutoff). Keep them only if we
       have data on them; otherwise we can't ground anything.
    2. Compute data-derived strength from the avg rs_6mo of the SURVIVING
       members. If the LLM's claimed strength is materially off (delta>3),
       blend toward the data value and log a warning so drift is visible.
    3. Drop themes that have <3 surviving members (LLM hallucinated the
       whole theme).

    Returns a NEW MarketThemes instance (or None if all themes were
    dropped). The original LLM output is left untouched.
    """
    if themes is None:
        return None
    from ..models.llm import MarketTheme, MarketThemes

    if not isinstance(themes, MarketThemes):
        return themes

    upper_universe = {t.upper() for t in universe_tickers}
    rs6_by_ticker: dict[str, float] = {}
    for ticker, t in technicals.items():
        rs6 = t.get("rs_6mo")
        if rs6 is not None:
            rs6_by_ticker[ticker.upper()] = float(rs6)

    corrected: list[MarketTheme] = []
    for theme in themes.themes:
        valid_members = [t for t in theme.member_tickers if t.upper() in upper_universe]
        dropped = [t for t in theme.member_tickers if t.upper() not in upper_universe]
        if dropped:
            logger.info(
                "Theme '%s': dropped %d/%d tickers not in universe: %s",
                theme.name,
                len(dropped),
                len(theme.member_tickers),
                ", ".join(dropped[:10]),
            )

        if len(valid_members) < 3:
            logger.warning(
                "Theme '%s': only %d valid member(s) survive — dropping "
                "theme entirely (likely hallucinated).",
                theme.name,
                len(valid_members),
            )
            continue

        # Data-derived strength: avg rs_6mo across surviving members,
        # mapped 0..10 via a sigmoid-ish curve. SPY-neutral → ~5, +15% → ~8,
        # +25% → ~9, -10% → ~3, -20% → ~1.
        rs6_values = [rs6_by_ticker[m.upper()] for m in valid_members if m.upper() in rs6_by_ticker]
        if rs6_values:
            avg_rs = sum(rs6_values) / len(rs6_values)
            data_strength = max(1, min(10, round(5 + avg_rs * 25)))
        else:
            data_strength = theme.strength

        # Reconcile: if claimed strength diverges from data by > 3, log
        # warning and blend (60% data, 40% LLM).
        delta = abs(theme.strength - data_strength)
        if delta > 3:
            corrected_strength = round(0.6 * data_strength + 0.4 * theme.strength)
            logger.warning(
                "Theme '%s': LLM claimed strength=%d, data says %d "
                "(avg rs_6mo of members = %.1f%%). Adjusting to %d.",
                theme.name,
                theme.strength,
                data_strength,
                (sum(rs6_values) / len(rs6_values) * 100) if rs6_values else 0,
                corrected_strength,
            )
            new_strength = corrected_strength
        else:
            new_strength = theme.strength

        # Reconcile trending against data: if avg rs_6mo strongly negative,
        # force 'down'; strongly positive → 'up'.
        if rs6_values:
            avg_rs = sum(rs6_values) / len(rs6_values)
            if avg_rs < -0.05 and theme.trending == "up":
                logger.warning(
                    "Theme '%s': LLM said trending=up but avg rs_6mo "
                    "of members is %.1f%% — flipping to 'down'.",
                    theme.name,
                    avg_rs * 100,
                )
                new_trending = "down"
            elif avg_rs > 0.10 and theme.trending == "down":
                logger.warning(
                    "Theme '%s': LLM said trending=down but avg rs_6mo "
                    "of members is %.1f%% — flipping to 'up'.",
                    theme.name,
                    avg_rs * 100,
                )
                new_trending = "up"
            else:
                new_trending = theme.trending
        else:
            new_trending = theme.trending

        corrected.append(
            MarketTheme(
                name=theme.name,
                description=theme.description,
                strength=new_strength,
                trending=new_trending,
                member_tickers=valid_members,
            )
        )

    if not corrected:
        logger.warning("All themes were invalidated; returning None.")
        return None

    # Rebuild full_text to reflect the corrections.
    parts: list[str] = []
    for t in corrected:
        parts.append(
            f"THEME: {t.name} [strength {t.strength}/10, trending {t.trending}]\n"
            f"{t.description}\n"
            f"Members: {', '.join(t.member_tickers)}"
        )
    return MarketThemes(themes=corrected, full_text="\n\n".join(parts))


def _top_fail_reasons(candidates: list[dict[str, Any]], *, k: int = 3) -> str:
    """Aggregate the top-K fail-reason strings across all candidates, so
    a 'no survivors' log line is actionable (e.g. tells you debt/equity
    or market-cap was the dominant filter)."""
    from collections import Counter

    counter: Counter[str] = Counter()
    for c in candidates:
        for r in c.get("fail_reasons") or []:
            counter[r] += 1
    if not counter:
        return "(no fail_reasons captured)"
    return ", ".join(f"{r} ({n})" for r, n in counter.most_common(k))


def _log_discover_analysis(
    *,
    delivered: bool,
    delivery_error: str | None,
    local_pdf_path: str,
    ranker_text: str,
    redteam_text: str,
    sizer_text: str,
) -> None:
    """Dump every analyst-produced section to the logger so the user can
    recover the full report from the log file when email fails."""
    bar = "=" * 70
    if not delivered:
        logger.error(
            "%s\nEMAIL NOT DELIVERED — full analysis follows in this log.\nReason: %s\nPDF: %s\n%s",
            bar,
            delivery_error or "EMAIL_TO not configured",
            local_pdf_path,
            bar,
        )
    logger.info("%s\nRANKER — discover picks\n%s\n%s", bar, bar, ranker_text)
    logger.info("%s\nRED TEAM — bear cases\n%s\n%s", bar, bar, redteam_text)
    logger.info("%s\nSIZER — allocation\n%s\n%s", bar, bar, sizer_text)


MAX_CANDIDATES_FOR_LLM = 25

# Per-field char caps applied to analyst/reviewer payloads to keep each call
# well under Sonnet's 30k input-tokens/min rate limit. 10-K text is mostly
# boilerplate; MD&A and transcripts retain most signal at these sizes.
_RISK_FACTORS_CHARS = 3500
_QUARTERLY_MDA_CHARS = 4000
_TRANSCRIPT_CHARS = 2500


def _trim(text: str | None, max_chars: int) -> str | None:
    if not text:
        return text
    return text[:max_chars]


# --- small helpers (used by step executors) ----------------------------------


def _fetch_news(ticker: str, limit: int = 3) -> list[dict[str, Any]]:
    items = (
        yf_gateway.ticker_call(ticker, "discover.news", lambda t: t.news or [], default=[]) or []
    )
    out: list[dict[str, Any]] = []
    for it in items[:limit]:
        title = it.get("title") or (it.get("content") or {}).get("title")
        link = it.get("link") or ((it.get("content") or {}).get("canonicalUrl") or {}).get("url")
        if title:
            out.append({"title": title, "link": link})
    return out


def _batch_news(tickers: list[str]) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    for ticker, news in yf_gateway.map_symbols(_fetch_news, tickers, workers=5):
        results[ticker] = news
    return results


def _aggregate_holdings(holdings: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
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
    return agg


def _holdings_summary(holdings: dict[str, list[dict[str, Any]]]) -> str:
    """Plain-text rendering — this is what the LLM stages (Sizer, Ranker,
    etc.) read as prompt input, so its format stays stable independent of
    how the report renders the same data (see `_holdings_table_rows`)."""
    agg = _aggregate_holdings(holdings)
    if not agg:
        return ""
    lines: list[str] = []
    for ticker, v in sorted(agg.items()):
        avg = v["cost"] / v["units"] if v["units"] else 0
        lines.append(f"  - {ticker}: {v['units']:.0f} shares @ avg ${avg:,.2f}")
    return "\n".join(lines)


def _holdings_table_rows(holdings: dict[str, list[dict[str, Any]]]) -> list[list[str]]:
    """Same aggregated holdings as `_holdings_summary`, shaped as table
    rows for the report instead of a monospace bullet list."""
    agg = _aggregate_holdings(holdings)
    rows: list[list[str]] = []
    for ticker, v in sorted(agg.items()):
        avg = v["cost"] / v["units"] if v["units"] else 0
        rows.append([ticker, f"{v['units']:.0f}", f"${avg:,.2f}", f"${v['cost']:,.0f}"])
    return rows


def _holdings_value_by_sector(
    holdings: dict[str, list[dict[str, Any]]],
    sector_of: dict[str, str],
) -> dict[str, float]:
    """Market value of current holdings per sector (units x brokerage price).
    Positions with no price or no known sector are left out."""
    out: dict[str, float] = {}
    for items in holdings.values():
        for h in items:
            ticker = str(h.get("ticker") or "").upper()
            value = float(h.get("units") or 0) * float(h.get("price") or 0)
            sector = sector_of.get(ticker)
            if value > 0 and sector:
                out[sector] = out.get(sector, 0.0) + value
    return out


# --- pipeline ----------------------------------------------------------------


class DiscoverPipeline:
    """Holds shared state across Workflow steps.

    Each `step_*` method is bound to this instance, so steps read/write
    self.state instead of round-tripping data through StepOutput content.
    Fatal conditions (empty universe, no survivors) raise RuntimeError —
    the Workflow aborts cleanly and the run shows as failed in workflow_session.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.state: dict[str, Any] = {}
        TRACKER.reset()

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
            ok, why = passes_trend_gate(technicals.get(ticker))
            if ok or ticker in always:
                passed.append(ticker)
            else:
                reasons[ticker] = why

        # Cap the survivors by 6-month relative strength, keeping the
        # user's names outside the cap.
        cap = self.settings.discover_max_screen_candidates
        capped_out: list[str] = []
        if len(passed) > cap:
            ranked = sorted(
                (t for t in passed if t not in always),
                key=lambda t: (technicals.get(t) or {}).get("rs_6mo") or 0.0,
                reverse=True,
            )
            keep = set(ranked[: max(0, cap - len(always))]) | always
            capped_out = [t for t in passed if t not in keep]
            for ticker in capped_out:
                reasons[ticker] = ["below the relative-strength cap for deep analysis"]
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
        self.state["macro_summary"] = regime_summary_text(data)
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
        themes = agent.detect(
            macro_summary=self.state.get("macro_summary", ""),
            sector_rotation=self.state.get("sector_rotation"),
            technicals=self.state.get("technicals", {}),
            fundamentals=self.state.get("fundamentals", {}),
            eps_revisions=self.state.get("eps_revisions", {}),
        )
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

        candidates: list[dict[str, Any]] = []
        for ticker in self.state["tickers"]:
            f = fundamentals.get(ticker)
            t = technicals.get(ticker)
            u = universe[ticker]
            if ticker in prescreen_reasons:
                # Eliminated before the fundamentals fetch — report why it
                # actually failed rather than "no fundamentals data".
                passes, reasons = False, list(prescreen_reasons[ticker])
            else:
                passes, reasons = passes_hard_filter(f, t)
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
            candidates.append(cand)

        survivors = sorted(
            [c for c in candidates if c["passed_filter"]],
            key=lambda c: c["score"] or 0,
            reverse=True,
        )[:MAX_CANDIDATES_FOR_LLM]
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
        self.state["candidates"] = candidates
        self.state["survivors"] = survivors
        self.state["survivor_tickers"] = [c["ticker"] for c in survivors]

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
        self.state["earnings_alerts"] = batch_earnings_flags(tickers, within_days=5)
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

    def step_analyst(self, step_input: StepInput) -> StepOutput:
        survivors = self.state.get("survivors") or []
        if not survivors:
            # Empty after screen short-circuited. Set everything downstream
            # depends on so the rest of the pipeline degrades cleanly.
            self.state["analyses"] = {}
            return StepOutput(content="analyst: no survivors; skipping")
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
        analyses, catalyst_warnings = repair_catalysts(
            analyze_tiered(deep, light, payloads, deep_tickers), recent_news
        )
        self.state["analyses"] = analyses
        self.state["catalyst_warnings"] = catalyst_warnings
        if not self.state["analyses"]:
            logger.error("Analyst: all calls failed; downstream LLM stages will skip")
            return StepOutput(content="Analyst: all calls failed; downstream will skip")
        return StepOutput(content=f"Analyst: {len(self.state['analyses'])} scorecards")

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
        from ..models.llm import RankerOutput

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
        return pick_sectors, _holdings_value_by_sector(holdings, sector_of)

    def step_persist_and_report(self, step_input: StepInput) -> StepOutput:
        # 1. SQLite persistence (same as before)
        with get_session(self.settings.discover_db_path) as session:
            run_id = insert_run(
                session,
                universe_size=len(self.state["candidates"]),
                survivors=len(self.state["survivors"]),
                picks=len(self.state["picks"]),
                opus_model=self.settings.discover_opus_model,
                sonnet_model=self.settings.discover_sonnet_model,
                cash_budget=self.settings.discover_cash_budget,
                kind="discover",
            )
            for c in self.state["candidates"]:
                insert_candidate(
                    session,
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
            for ticker, report in self.state["analyses"].items():
                analyst_text = getattr(report, "full_text", None) or (
                    report if isinstance(report, str) else ""
                )
                insert_scorecard(session, run_id, ticker, analyst_text)
            # Forecast fields travel with the pick so calibration can grade
            # them later; `entry_price` is the screen-time price, never a
            # refetch, so a historical pick is never repriced with new data.
            forecasts = _pick_forecasts(self.state.get("ranker_output"))
            prices = {c["ticker"]: c.get("price") for c in self.state["candidates"]}
            for rank, ticker, _ in self.state["picks"]:
                forecast = forecasts.get(ticker, {})
                insert_pick(
                    session,
                    run_id,
                    rank=rank,
                    ticker=ticker,
                    conviction=forecast.get("conviction"),
                    ev_pct=forecast.get("ev_pct"),
                    entry_price=prices.get(ticker),
                    time_horizon=forecast.get("time_horizon"),
                    scenarios=forecast.get("scenarios"),
                    agreement_ratio=forecast.get("agreement_ratio"),
                    voting_providers=forecast.get("voting_providers"),
                )
                analysis = self.state["analyses"].get(ticker)
                if analysis is not None and getattr(analysis, "upcoming_catalysts", None):
                    insert_pick_catalysts(
                        session,
                        run_id,
                        ticker,
                        catalysts_to_dicts(analysis.upcoming_catalysts),
                    )
            insert_run_outputs(
                session,
                run_id,
                ranker_full=self.state["ranker_text"],
                redteam_full=self.state["redteam_text"],
                sizer_full=self.state["sizer_text"],
                holdings_summary=self.state["holdings_summary"],
            )

        # 2. Fetch a chart for each pick (existing chart-img.com client).
        pick_tickers = [t for _, t, _ in self.state["picks"]]
        charts: dict[str, bytes] = {}
        try:
            charts = fetch_charts(pick_tickers)
        except Exception as e:
            logger.warning("Chart fetch failed (%s) — report will omit charts", e)
        chart_cids = {t: f"chart-{t.replace('.', '-')}" for t in charts}

        # 2b. Style factor tilt — remap each pick's existing score_breakdown
        # leaves into named growth/value/quality/momentum/low_vol buckets
        # for reporting only (no rescoring).
        candidates_by_ticker = {c["ticker"]: c for c in self.state["candidates"]}
        hv_data = self.state.get("historical_volatility") or {}
        pick_tilts: dict[str, dict[str, float]] = {}
        for ticker in pick_tickers:
            cand = candidates_by_ticker.get(ticker)
            if cand is None:
                continue
            tilt = compute_factor_tilt(cand.get("score_breakdown"), hv_data.get(ticker))
            if tilt:
                pick_tilts[ticker] = tilt
        portfolio_tilt = average_factor_tilts(list(pick_tilts.values()))
        analyses = self.state.get("analyses") or {}
        pick_catalysts = {
            t: catalysts_to_dicts(analyses[t].upcoming_catalysts)
            for t in pick_tickers
            if t in analyses
        }

        # 3. Build shared section list, then render both HTML and PDF from it.
        sections = build_sections(
            ranker_text=self.state["ranker_text"],
            redteam_text=self.state["redteam_text"],
            sizer_text=self.state["sizer_text"],
            candidates=self.state["candidates"],
            universe_size=len(self.state["candidates"]),
            holdings_summary=self.state["holdings_summary"],
            holdings_rows=self.state.get("holdings_table_rows"),
            macro_summary=self.state.get("macro_summary", ""),
            sector_rotation=self.state.get("sector_rotation"),
            track_record_block=self.state.get("track_record_block", ""),
            track_record=self.state.get("track_record"),
            ranker_output=self.state.get("ranker_output"),
            redteam_output=self.state.get("redteam_output"),
            sizer_output=self.state.get("sizer_output"),
            market_themes=self.state.get("market_themes"),
            data_warnings=(
                (self.state.get("output_validation_warnings") or [])
                + (self.state.get("macro_veto_reasons") or [])
                + (self.state.get("catalyst_warnings") or [])
            ),
            pick_tilts=pick_tilts,
            portfolio_tilt=portfolio_tilt,
            pick_catalysts=pick_catalysts,
            usage=TRACKER.report_data(),
            paper_ledger=self.state.get("paper_ledger"),
        )
        html_body = render_html_email(sections, chart_cids)
        pdf_bytes = render_pdf(sections, charts)

        # 4. Send email (or fall back to logging if EMAIL_TO unset).
        today = date.today()
        picks_summary = ", ".join(pick_tickers[:5])
        subject = (
            f"Stock Discovery — {today.strftime('%b-%d')}: {picks_summary}"
            if pick_tickers
            else f"Stock Discovery — {today.strftime('%b-%d')}"
        )
        pdf_filename = f"discover-{today.isoformat()}.pdf"

        # Save PDF locally BEFORE the email attempt so a delivery failure
        # (SMTP outage, wrong creds, etc.) never costs the user the report.
        local_pdf_path = _save_local_pdf(pdf_bytes, pdf_filename)
        logger.info("Saved discover PDF locally: %s", local_pdf_path)

        delivered = False
        delivery_error: str | None = None
        if self.settings.email_to:
            try:
                SmtpServer().send_email(
                    self.settings.email_to,
                    subject,
                    html_body,
                    content_type="html",
                    inline_images={chart_cids[t]: data for t, data in charts.items()} or None,
                    attachments=[(pdf_filename, pdf_bytes, "pdf")],
                )
                delivered = True
                logger.info("Sent discovery email to %s", self.settings.email_to)
            except Exception as e:
                delivery_error = str(e)
                logger.error("Email delivery failed: %s", e)
        else:
            logger.warning(
                "EMAIL_TO not set; skipping email delivery. "
                "Run %d's HTML/PDF available via state if you want to inspect them.",
                run_id,
            )

        # Always dump the full analysis to the log so the user can recover
        # every section — even when email is offline or wasn't configured.
        _log_discover_analysis(
            delivered=delivered,
            delivery_error=delivery_error,
            local_pdf_path=str(local_pdf_path),
            ranker_text=self.state["ranker_text"],
            redteam_text=self.state["redteam_text"],
            sizer_text=self.state["sizer_text"],
        )

        self.state["run_id"] = run_id
        self.state["pdf_bytes"] = pdf_bytes
        self.state["html_body"] = html_body
        self.state["local_pdf_path"] = str(local_pdf_path)
        print_terminal_summary(self.state["ranker_text"], self.state["sizer_text"])
        print(f"\nPDF saved: {local_pdf_path}")
        log_path = current_log_file()
        if log_path:
            print(f"Log file:  {log_path}")
        status = "emailed" if delivered else "persisted (no email)"
        return StepOutput(
            content=(
                f"Run #{run_id} {status}; PDF {len(pdf_bytes)} bytes (saved to {local_pdf_path})"
            )
        )

    # --- workflow assembly --------------------------------------------

    def build_workflow(self) -> Workflow:
        db_path = Path(os.path.expanduser(self.settings.discover_db_path))
        db_path.parent.mkdir(parents=True, exist_ok=True)

        return Workflow(
            name="Stock Discovery",
            description="Find mid-long term holds via screen + Sonnet + Opus reasoning",
            db=SqliteDb(
                db_file=str(db_path),
                session_table="workflow_session",
            ),
            steps=[
                Step(name="universe", executor=self.step_universe),
                # Technicals first, alone among the Yahoo-backed steps: one
                # request per name buys the trend gate, which decides who is
                # worth the three-requests-per-name fetches below.
                Parallel(
                    Step(name="technicals", executor=self.step_technicals),
                    Step(name="sector_rotation", executor=self.step_sector_rotation),
                    Step(name="macro_regime", executor=self.step_macro_regime),
                    Step(name="track_record", executor=self.step_track_record),
                    name="market_data",
                ),
                Step(name="prescreen", executor=self.step_prescreen),
                Parallel(
                    Step(name="fundamentals", executor=self.step_fundamentals),
                    # EPS revisions run here so the score function can pick
                    # up the +/-5 trend bonus from direction_30d.
                    Step(name="eps_revisions", executor=self.step_eps_revisions),
                    Step(
                        name="historical_volatility",
                        executor=self.step_historical_volatility,
                    ),
                    name="candidate_data",
                ),
                # Market themes need sector_rotation + macro_regime as input,
                # so it runs sequentially after the market_data block.
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
                    name="enrichment",
                ),
                Step(name="analyst", executor=self.step_analyst),
                Step(name="holdings", executor=self.step_holdings),
                Step(name="ranker", executor=self.step_ranker),
                Step(name="macro_veto", executor=self.step_macro_veto),
                Step(name="redteam", executor=self.step_redteam),
                Step(name="sizer", executor=self.step_sizer),
                Step(name="persist_and_report", executor=self.step_persist_and_report),
            ],
        )


def run() -> None:
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
            needs_discover_providers=True,
        )
    except PreflightError as e:
        logger.error("%s", e)
        raise SystemExit(2) from e
    pipeline = DiscoverPipeline(settings)
    workflow = pipeline.build_workflow()
    logger.info("=== Stock discovery pipeline starting ===")
    try:
        workflow.print_response(input="discover", stream=True)
    finally:
        # Request budget for the run: how much Yahoo traffic it took, how
        # often it was throttled, and what the pacer settled on. Read this
        # before touching YF_RATE_LIMIT_PER_MIN.
        yf_gateway.log_stats("discover run")
        log_usage_summary()
        unavailable = yf_gateway.unavailable_symbols()
        if unavailable:
            logger.info(
                "Symbols Yahoo had no data for this run (skipped after the first miss): %s",
                ", ".join(sorted(unavailable)),
            )
    if pipeline.state.get("run_id"):
        print(f"\nRun #{pipeline.state['run_id']} stored in {settings.discover_db_path}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
