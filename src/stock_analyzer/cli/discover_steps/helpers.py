"""Formatting and aggregation helpers shared by the discover steps."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agno.workflow import Workflow

from ...data import yf_gateway
from ...logging import get_logger

logger = get_logger("stock_analyzer.cli.discover")


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
    from ...models.llm import RankerOutput, expected_return_pct

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
    from ...models.llm import RankerOutput, expected_return_pct

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
        # 3-5 year picks quote per-year (annualized) returns.
        per = "/yr" if "year" in (pick.time_horizon or "").lower() else ""
        bull_s = f"{bull.probability:.0%}/{bull.target_return_pct:+.0f}%{per}" if bull else "—"
        base_s = f"{base.probability:.0%}/{base.target_return_pct:+.0f}%{per}" if base else "—"
        bear_s = f"{bear.probability:.0%}/{bear.target_return_pct:+.0f}%{per}" if bear else "—"
        rows.append(
            f"  {pick.ticker:6s}  E[ret]={ev:+5.1f}%{per}  "
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
    from ...models.llm import RankerOutput

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
    from ...models.llm import RankerOutput

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
    from ...models.llm import MarketThemes

    if not isinstance(themes, MarketThemes):
        return themes

    upper_universe = {t.upper() for t in universe_tickers}
    rs6_by_ticker: dict[str, float] = {}
    for ticker, t in technicals.items():
        rs6 = t.get("rs_6mo")
        if rs6 is not None:
            rs6_by_ticker[ticker.upper()] = float(rs6)

    corrected = [
        fixed
        for theme in themes.themes
        if (fixed := _correct_theme(theme, upper_universe, rs6_by_ticker)) is not None
    ]
    if not corrected:
        logger.warning("All themes were invalidated; returning None.")
        return None
    return MarketThemes(themes=corrected, full_text=_themes_full_text(corrected))


def _correct_theme(
    theme: Any, upper_universe: set[str], rs6_by_ticker: dict[str, float]
) -> Any | None:
    """The theme with unknown members dropped and strength/trend checked
    against the members' data, or None if fewer than 3 members survive."""
    from ...models.llm import MarketTheme

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
        return None

    rs6_values = [rs6_by_ticker[m.upper()] for m in valid_members if m.upper() in rs6_by_ticker]
    return MarketTheme(
        name=theme.name,
        description=theme.description,
        strength=_reconciled_strength(theme, rs6_values),
        trending=_reconciled_trend(theme, rs6_values),
        member_tickers=valid_members,
    )


def _reconciled_strength(theme: Any, rs6_values: list[float]) -> int:
    """Data-derived strength: avg rs_6mo across surviving members, mapped
    0..10 via a sigmoid-ish curve (SPY-neutral → ~5, +15% → ~8, +25% → ~9,
    -10% → ~3, -20% → ~1). If the claimed strength diverges from it by more
    than 3, log a warning and blend (60% data, 40% LLM)."""
    if rs6_values:
        avg_rs = sum(rs6_values) / len(rs6_values)
        data_strength = max(1, min(10, round(5 + avg_rs * 25)))
    else:
        data_strength = theme.strength

    delta = abs(theme.strength - data_strength)
    if delta <= 3:
        return theme.strength
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
    return corrected_strength


def _reconciled_trend(theme: Any, rs6_values: list[float]) -> str:
    """Reconcile trending against data: if avg rs_6mo is strongly negative,
    force 'down'; strongly positive → 'up'."""
    if not rs6_values:
        return theme.trending
    avg_rs = sum(rs6_values) / len(rs6_values)
    if avg_rs < -0.05 and theme.trending == "up":
        logger.warning(
            "Theme '%s': LLM said trending=up but avg rs_6mo "
            "of members is %.1f%% — flipping to 'down'.",
            theme.name,
            avg_rs * 100,
        )
        return "down"
    if avg_rs > 0.10 and theme.trending == "down":
        logger.warning(
            "Theme '%s': LLM said trending=down but avg rs_6mo "
            "of members is %.1f%% — flipping to 'up'.",
            theme.name,
            avg_rs * 100,
        )
        return "up"
    return theme.trending


def _themes_full_text(corrected: list[Any]) -> str:
    """full_text rebuilt to reflect the corrections."""
    parts: list[str] = []
    for t in corrected:
        parts.append(
            f"THEME: {t.name} [strength {t.strength}/10, trending {t.trending}]\n"
            f"{t.description}\n"
            f"Members: {', '.join(t.member_tickers)}"
        )
    return "\n\n".join(parts)


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
    prices: dict[str, float] | None = None,
) -> dict[str, float]:
    """Market value of current holdings per sector. One price per ticker
    (data/pricing.py) where it is known, so a stale account feed can't tip
    a sector over its cap; the account's own price otherwise. Positions
    with no price or no known sector are left out."""
    out: dict[str, float] = {}
    prices = prices or {}
    for items in holdings.values():
        for h in items:
            ticker = str(h.get("ticker") or "").upper()
            price = prices.get(ticker) or float(h.get("price") or 0)
            value = float(h.get("units") or 0) * price
            sector = sector_of.get(ticker)
            if value > 0 and sector:
                out[sector] = out.get(sector, 0.0) + value
    return out


# --- pipeline ----------------------------------------------------------------


def without_step_retries(workflow: Workflow) -> Workflow:
    """Turn off agno's automatic step re-run (default: 3 retries).

    A step that raises would otherwise be executed again from the top —
    re-paying every LLM call it already made (the Analyst fan-out, all
    Ranker rounds, the Opus Rebalancer). Provider failures are already
    retried once on the fallback provider inside the step (llm.py
    run_with_fallback); data fetches retry in the HTTP/yfinance layers.
    """

    def walk(steps) -> None:
        for step in steps or []:
            if hasattr(step, "max_retries"):
                step.max_retries = 0
            walk(getattr(step, "steps", None))

    walk(workflow.steps)
    return workflow
