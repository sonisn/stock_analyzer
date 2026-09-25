"""CRUD repository for the SQLite analytics DB.

Signatures mirror the legacy discover/persistence.py API exactly — same
keyword args, same return types — except the first argument is now a
Session instead of a sqlite3.Connection. JSON marshalling lives here:
the table classes hold raw TEXT, the repository converts to/from
Python types at the boundary.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlmodel import Session, col, select

from .tables import (
    Candidate,
    CandidateSnapshot,
    HoldingReviewRow,
    Pick,
    PickCatalyst,
    PickScenario,
    PortfolioSnapshot,
    Run,
    RunOutput,
    Scorecard,
    Suggestion,
)

# --- runs -----------------------------------------------------------------


def insert_run(
    session: Session,
    *,
    universe_size: int,
    survivors: int,
    picks: int,
    opus_model: str,
    sonnet_model: str,
    cash_budget: float | None,
    kind: str = "discover",
) -> int:
    """Insert a run row, return the assigned run_id."""
    row = Run(
        run_at=datetime.now().isoformat(timespec="seconds"),
        kind=kind,
        universe_size=universe_size,
        survivors=survivors,
        picks=picks,
        opus_model=opus_model,
        sonnet_model=sonnet_model,
        cash_budget=cash_budget,
    )
    session.add(row)
    session.flush()  # populate row.id without committing
    if row.id is None:
        raise RuntimeError("Run.id was None after session.flush() — SQLite autoincrement failed")
    return int(row.id)


# --- candidates -----------------------------------------------------------

# Numeric fundamentals worth keeping point-in-time (see CandidateSnapshot).
SNAPSHOT_FIELDS: tuple[str, ...] = (
    "market_cap",
    "revenue_growth_yoy",
    "earnings_growth_yoy",
    "fcf_yield",
    "debt_to_equity",
    "gross_margin",
    "operating_margin",
    "profit_margin",
    "forward_pe",
    "trailing_pe",
    "peg_ratio",
    "analyst_target_upside_pct",
    "analyst_recommendation_mean",
    "analyst_count",
    "shares_short_pct",
    "short_ratio_days",
)


def insert_candidate_snapshot(
    session: Session,
    run_id: int,
    ticker: str,
    fundamentals: dict[str, Any] | None,
    revisions: dict[str, Any] | None = None,
) -> None:
    """Store the numeric fundamentals (4 significant figures) and EPS
    revision counts this run saw. Skipped when there is nothing to keep."""
    data: dict[str, float] = {}
    for key in SNAPSHOT_FIELDS:
        v = (fundamentals or {}).get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
            data[key] = float(f"{v:.4g}")
    for key in ("net_revisions_30d", "net_revisions_7d"):
        v = (revisions or {}).get(key)
        if isinstance(v, int):
            data[key] = v
    if data:
        session.add(
            CandidateSnapshot(run_id=run_id, ticker=ticker, data=json.dumps(data, sort_keys=True))
        )


def insert_candidate(
    session: Session,
    run_id: int,
    ticker: str,
    *,
    passed_filter: bool,
    fail_reasons: list[str],
    score: float | None,
    score_components: dict[str, Any] | None,
    score_breakdown: dict[str, Any] | None,
    sources: list[str],
    conviction: int,
    sector: str | None,
    price: float | None,
) -> None:
    session.add(
        Candidate(
            run_id=run_id,
            ticker=ticker,
            passed_filter=int(passed_filter),
            fail_reasons=json.dumps(fail_reasons),
            score=score,
            score_components=json.dumps(score_components) if score_components else None,
            score_breakdown=json.dumps(score_breakdown) if score_breakdown else None,
            sources=json.dumps(sources),
            conviction=conviction,
            sector=sector,
            price=price,
        )
    )


# --- scorecards -----------------------------------------------------------


def insert_scorecard(session: Session, run_id: int, ticker: str, text: str) -> None:
    session.add(Scorecard(run_id=run_id, ticker=ticker, analyst_text=text))


# --- picks ----------------------------------------------------------------


def insert_pick(
    session: Session,
    run_id: int,
    *,
    rank: int,
    ticker: str,
    conviction: int | None = None,
    ev_pct: float | None = None,
    entry_price: float | None = None,
    time_horizon: str | None = None,
    scenarios: list[dict[str, Any]] | None = None,
    agreement_ratio: float | None = None,
    voting_providers: list[str] | None = None,
) -> None:
    """Persist one pick plus the forecast behind it.

    The forecast args are what make calibration possible: without
    `conviction`, `ev_pct` and the `scenarios` rows there is no way to ask
    later whether conviction-9 picks actually beat conviction-6 ones, or
    whether the stated bear probabilities were too low. They default to
    None so older callers keep working, but the discover pipeline always
    passes them.

    `agreement_ratio`/`voting_providers` are the multi-provider Ranker
    consensus vote for this pick (None for single-round runs) — see
    `discover/track_record.py::_compute_provider_breakdown`.

    Each scenario dict is {"label", "probability", "target_return_pct"}.

    The ranker / red-team / sizer prose is stored once per run in
    `run_outputs`; the per-pick text columns used to hold a full copy of it
    for every pick and are left empty now.
    """
    session.add(
        Pick(
            run_id=run_id,
            rank=rank,
            ticker=ticker,
            ranker_text="",
            conviction=conviction,
            ev_pct=ev_pct,
            entry_price=entry_price,
            time_horizon=time_horizon,
            agreement_ratio=agreement_ratio,
            voting_providers=",".join(voting_providers) if voting_providers else None,
        )
    )
    for scenario in scenarios or []:
        label = str(scenario.get("label") or "").strip().lower()
        if label not in ("bull", "base", "bear"):
            continue
        probability = scenario.get("probability")
        target = scenario.get("target_return_pct")
        if probability is None or target is None:
            continue
        session.add(
            PickScenario(
                run_id=run_id,
                rank=rank,
                label=label,
                ticker=ticker,
                probability=float(probability),
                target_return_pct=float(target),
            )
        )


# --- holdings reviews -----------------------------------------------------


def insert_pick_catalysts(
    session: Session, run_id: int, ticker: str, catalysts: list[dict[str, Any]]
) -> None:
    """Persist a pick's validated upcoming catalysts (catalysts_to_dicts shape)."""
    for seq, c in enumerate(catalysts):
        session.add(
            PickCatalyst(
                run_id=run_id,
                ticker=ticker,
                seq=seq,
                event=str(c.get("event") or ""),
                expected_date=c.get("expected_date"),
                direction=str(c.get("direction") or "uncertain"),
                impact=str(c.get("impact") or "low"),
                source=str(c.get("source") or ""),
            )
        )


def insert_holdings_review(
    session: Session,
    run_id: int,
    ticker: str,
    *,
    verdict: str | None,
    confidence: int | None,
    review_text: str,
) -> None:
    session.add(
        HoldingReviewRow(
            run_id=run_id,
            ticker=ticker,
            verdict=verdict,
            confidence=confidence,
            review_text=review_text,
        )
    )


def fetch_recent_holdings_history(
    session: Session, *, n_runs: int = 3, kind: str = "rebalance"
) -> dict[str, list[dict[str, Any]]]:
    """Return {ticker: [{run_at, verdict, confidence}, ...]} oldest-first for
    the last `n_runs` runs of `kind`. Same shape as the legacy function."""
    # DESC + LIMIT to grab the most recent N rows; then reverse to chronological
    # (ascending) order so the LLM reads them oldest-first.
    recent_runs = list(
        session.exec(
            select(Run.id, Run.run_at)
            .where(Run.kind == kind)
            .order_by(col(Run.id).desc())
            .limit(n_runs)
        )
    )
    if not recent_runs:
        return {}
    recent_runs.reverse()
    out: dict[str, list[dict[str, Any]]] = {}
    for run_row in recent_runs:
        rows = session.exec(
            select(
                HoldingReviewRow.ticker,
                HoldingReviewRow.verdict,
                HoldingReviewRow.confidence,
            ).where(HoldingReviewRow.run_id == run_row.id)
        )
        for review_row in rows:
            out.setdefault(review_row.ticker, []).append(
                {
                    "run_at": run_row.run_at,
                    "verdict": review_row.verdict,
                    "confidence": review_row.confidence,
                }
            )
    return out


def fetch_recent_picks(session: Session, *, n_runs: int = 3) -> list[tuple[str, int, str]]:
    """[(ticker, rank, run_at), ...] for every pick of the last `n_runs`
    runs that made picks (discover or rebalance), newest run first. Used
    to find cash-secured-put candidates."""
    run_ids = list(
        session.exec(
            select(Run.id)
            .where(col(Run.id).in_(select(Pick.run_id).distinct()))
            .order_by(col(Run.id).desc())
            .limit(n_runs)
        )
    )
    if not run_ids:
        return []
    rows = session.exec(
        select(Pick.ticker, Pick.rank, Run.run_at)
        .join(Run, col(Run.id) == col(Pick.run_id))
        .where(col(Pick.run_id).in_(run_ids))
        .order_by(col(Run.id).desc(), col(Pick.rank))
    )
    return [(t, r, at) for t, r, at in rows]


# --- suggestions ----------------------------------------------------------


def record_suggestions(session: Session, rows: list[dict[str, Any]]) -> int:
    """Store advice given today. Returns how many rows were added.

    A (day, source, action, ticker) already stored is not duplicated, so
    re-running the daily email is harmless. But a LATER run on the same
    day is a revision, not a repeat: on 2026-09-20 two rebalances both
    said BUY LLY, the first at ~$26,000 and the second — after it could
    see the covered calls — at $18,450, and the ledger kept the first.
    That is the sizing the grading would later assume was acted on. So a
    row from a newer run replaces the one on record, and the count
    returned stays the count of genuinely new advice.
    """
    added = 0
    for row in rows:
        existing = session.exec(
            select(Suggestion).where(
                Suggestion.suggested_on == row["suggested_on"],
                Suggestion.source == row["source"],
                Suggestion.action == row["action"],
                Suggestion.ticker == row["ticker"],
            )
        ).first()
        if existing is None:
            session.add(Suggestion(**row))
            added += 1
            continue
        new_run, old_run = row.get("run_id"), existing.run_id
        if new_run is not None and (old_run is None or new_run > old_run):
            for field, value in row.items():
                if field != "id":
                    setattr(existing, field, value)
    session.flush()
    return added


def fetch_suggestions(session: Session, *, start: str, end: str) -> list[Suggestion]:
    """Suggestions made on dates in [start, end] (ISO), oldest first."""
    return list(
        session.exec(
            select(Suggestion)
            .where(Suggestion.suggested_on >= start, Suggestion.suggested_on <= end)
            .order_by(col(Suggestion.suggested_on), col(Suggestion.id))
        )
    )


# --- portfolio snapshots --------------------------------------------------


def record_snapshot(
    session: Session,
    *,
    day: str,
    holdings_value: float,
    cash: float,
    accounts: dict[str, dict[str, float]] | None = None,
) -> None:
    """Store (or replace) the portfolio's value for `day`.

    `accounts` ({label: {"value", "cash"}}) records which accounts the
    total covered, so a later comparison can tell a newly connected
    account's balance apart from a gain."""
    row = session.get(PortfolioSnapshot, day)
    total = holdings_value + cash
    blob = json.dumps(accounts, sort_keys=True) if accounts else None
    if row is None:
        session.add(
            PortfolioSnapshot(
                day=day,
                holdings_value=holdings_value,
                cash=cash,
                total=total,
                accounts=blob,
            )
        )
    else:
        row.holdings_value, row.cash, row.total = holdings_value, cash, total
        # An account map is only ever replaced by another one: a run that
        # could not read the breakdown must not erase a good one.
        if blob:
            row.accounts = blob
    session.flush()


def snapshot_accounts(row: PortfolioSnapshot) -> dict[str, dict[str, float]] | None:
    """The stored account breakdown, or None when the row predates it (or
    holds unreadable JSON)."""
    if not row.accounts:
        return None
    try:
        parsed = json.loads(row.accounts)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def fetch_snapshots(session: Session, *, start: str | None = None) -> list[PortfolioSnapshot]:
    query = select(PortfolioSnapshot).order_by(PortfolioSnapshot.day)
    if start:
        query = query.where(PortfolioSnapshot.day >= start)
    return list(session.exec(query))


# --- run outputs ----------------------------------------------------------


def insert_run_outputs(
    session: Session,
    run_id: int,
    *,
    ranker_full: str,
    redteam_full: str,
    sizer_full: str,
    holdings_summary: str,
    rebalance_text: str | None = None,
    dashboard_data: dict[str, Any] | None = None,
) -> None:
    session.add(
        RunOutput(
            run_id=run_id,
            ranker_full=ranker_full,
            redteam_full=redteam_full,
            sizer_full=sizer_full,
            holdings_summary=holdings_summary,
            rebalance_text=rebalance_text,
            dashboard_data=(json.dumps(dashboard_data) if dashboard_data is not None else None),
        )
    )


__all__ = [
    "insert_run",
    "insert_candidate",
    "insert_scorecard",
    "insert_pick",
    "insert_holdings_review",
    "fetch_recent_holdings_history",
    "insert_run_outputs",
]
