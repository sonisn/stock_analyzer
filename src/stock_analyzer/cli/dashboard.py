"""`dashboard` — one self-contained HTML page of everything on record.

The daily email decides; this answers the questions it cannot. Why did
the reviewer change its mind on a holding? Did last quarter's sale beat
what replaced it? Did I act on any of it? All of that is in the database
and none of it was reachable.

It is a single file with the data embedded, so it makes no requests when
opened: no CORS, no API key in the page, nothing to be down while you are
reading it. Every fetch happens here, once, on the machine that has the
keys and the retry logic. Serve it with any static file server.

    uv run dashboard                 # write it to the reports directory
    uv run dashboard --open          # ...and print the path to open
    uv run dashboard --out /srv/x.html
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlmodel import select

from ..config import Settings
from ..db.session import get_session
from ..db.tables import (
    HoldingReviewRow,
    PickCatalyst,
    PickScenario,
    Run,
    RunOutput,
    StockView,
    Suggestion,
)
from ..logging import get_logger

logger = get_logger(__name__)


def _latest_review_run(db_path: str) -> int | None:
    with get_session(db_path) as session:
        return session.exec(
            select(HoldingReviewRow.run_id).order_by(HoldingReviewRow.run_id.desc())  # type: ignore[attr-defined]
        ).first()


def _reasoning(db_path: str) -> dict[str, dict[str, Any]]:
    """Per ticker: the scenarios behind its expected return, the bull case,
    the red team's bear case, and the catalysts named for it.

    The ledger says what was advised; this says why. All of it was already
    persisted — the scenarios as rows, the prose in `run_outputs` — and
    none of it was reachable outside the PDF of the run that produced it.
    Newest run wins, so a ticker picked twice shows the current thinking.
    """
    from ..discover.report_sections import _split_by_pick_blocks, _split_by_ticker_blocks

    out: dict[str, dict[str, Any]] = {}
    with get_session(db_path) as session:
        scenarios = session.exec(
            select(
                PickScenario.run_id,
                PickScenario.ticker,
                PickScenario.label,
                PickScenario.probability,
                PickScenario.target_return_pct,
            ).order_by(PickScenario.run_id)
        ).all()
        catalysts = session.exec(
            select(
                PickCatalyst.run_id,
                PickCatalyst.ticker,
                PickCatalyst.event,
                PickCatalyst.expected_date,
                PickCatalyst.direction,
                PickCatalyst.impact,
            ).order_by(PickCatalyst.run_id, PickCatalyst.seq)
        ).all()
        prose = session.exec(
            select(RunOutput.run_id, RunOutput.ranker_full, RunOutput.redteam_full).order_by(
                RunOutput.run_id
            )
        ).all()

    for run_id, ticker, label, prob, target in scenarios:
        rec = out.setdefault(ticker, {})
        if rec.get("_scen_run") != run_id:
            rec["_scen_run"], rec["scenarios"] = run_id, []
        rec["scenarios"] = [s for s in rec["scenarios"] if s["label"] != label]
        rec["scenarios"].append(dict(label=label, prob=round(prob * 100), target=round(target, 1)))
    for run_id, ticker, event, when, direction, impact in catalysts:
        rec = out.setdefault(ticker, {})
        if rec.get("_cat_run") != run_id:
            rec["_cat_run"], rec["catalysts"] = run_id, []
        rec["catalysts"].append(dict(event=event, when=when, direction=direction, impact=impact))
    for _run_id, ranker_text, redteam_text in prose:
        for ticker, block in _split_by_pick_blocks(ranker_text or "").items():
            out.setdefault(ticker, {})["bull"] = block.strip()
        for ticker, block in _split_by_ticker_blocks(redteam_text or "").items():
            out.setdefault(ticker, {})["bear"] = block.strip()

    for rec in out.values():
        rec.pop("_scen_run", None)
        rec.pop("_cat_run", None)
        if "scenarios" in rec:
            order = {"bull": 0, "base": 1, "bear": 2}
            rec["scenarios"].sort(key=lambda s: order.get(s["label"], 9))
    return {k: v for k, v in out.items() if v}


def collect(settings: Settings, *, today: date) -> dict[str, Any]:
    """Everything the page shows. Each source degrades on its own: a
    brokerage outage costs the holdings table, not the whole page."""
    from ..data.price_record import coverage, record_prices, stored_history
    from ..reporting.quarterly import grade_suggestions

    db = settings.discover_db_path
    run_id = _latest_review_run(db)

    positions: dict[str, dict[str, Any]] = {}
    obligations: dict[str, dict[str, Any]] = {}
    try:
        from ..data.brokerage import fetch_covered_call_obligations, fetch_portfolio_holdings

        for _acct, items in fetch_portfolio_holdings().items():
            for h in items:
                if t := h.get("ticker"):
                    p = positions.setdefault(t, {"units": 0.0, "cost": 0.0})
                    p["units"] += float(h.get("units") or 0)
                    p["cost"] += float(h.get("units") or 0) * float(
                        h.get("average_purchase_price") or 0
                    )
        obligations = fetch_covered_call_obligations()
    except Exception as e:  # noqa: BLE001
        logger.warning("Holdings unavailable (%s) — the page will say so", e)

    with get_session(db) as session:
        # Columns, not rows: an ORM object read after the session closes
        # raises DetachedInstanceError, and nothing here needs the object.
        reviews = {
            t: (v, c)
            for t, v, c in session.exec(
                select(
                    HoldingReviewRow.ticker,
                    HoldingReviewRow.verdict,
                    HoldingReviewRow.confidence,
                ).where(HoldingReviewRow.run_id == run_id)
            ).all()
        }
        review_text = {
            t: r
            for t, r in session.exec(
                select(HoldingReviewRow.ticker, HoldingReviewRow.review_text).where(
                    HoldingReviewRow.run_id == run_id
                )
            ).all()
        }
        history_rows = session.exec(
            select(
                HoldingReviewRow.ticker,
                HoldingReviewRow.run_id,
                HoldingReviewRow.verdict,
                HoldingReviewRow.confidence,
            ).order_by(HoldingReviewRow.run_id)
        ).all()
        run_days = {r_id: d[:10] for r_id, d in session.exec(select(Run.id, Run.run_at)).all()}
        views = {t: v for t, v in session.exec(select(StockView.ticker, StockView.view)).all()}
        suggestions = [
            dict(
                id=s.id,
                suggested_on=s.suggested_on,
                source=s.source,
                action=s.action,
                ticker=s.ticker,
                detail=s.detail,
                price=s.price,
                units_held=s.units_held,
                reinvest_into=s.reinvest_into,
                run_id=s.run_id,
            )
            for s in session.exec(select(Suggestion).order_by(Suggestion.id)).all()
        ]
        runs = [
            dict(
                id=r.id,
                kind=r.kind,
                d=r.run_at[:10],
                universe=r.universe_size,
                survivors=r.survivors,
                picks=r.picks,
            )
            for r in session.exec(select(Run).order_by(Run.id.desc())).all()[:20]  # type: ignore[attr-defined]
        ]

    # Record today's closes before grading, so the grade is computed from
    # the record rather than from a second, possibly different, fetch.
    tickers = sorted(
        set(positions)
        | {s["ticker"] for s in suggestions}
        | {s["reinvest_into"] for s in suggestions if s["reinvest_into"]}
        | {"SPY"}
    )
    record_prices(db, tickers, today=today)

    books: dict[str, dict[str, Any]] = {}
    try:
        from ..data.backlog import batch_rpo

        books = batch_rpo([t for t in positions if t.isalpha()])
    except Exception as e:  # noqa: BLE001
        logger.warning("Contracted books unavailable (%s)", e)

    fetch = stored_history(db)
    prices_now: dict[str, float] = {}
    for t in positions:
        frame = fetch(t, today.replace(year=today.year - 1), today)
        if frame is not None and not frame.empty:
            prices_now[t] = float(frame["Close"].iloc[-1])

    holdings = []
    for t, p in positions.items():
        units, cost = p["units"], p["cost"]
        px = prices_now.get(t)
        avg = cost / units if units else 0
        holdings.append(
            dict(
                ticker=t,
                units=round(units, 2),
                price=px,
                value=round(units * px) if px else None,
                pl=round((px / avg - 1) * 100, 1) if px and avg else None,
                verdict=(reviews.get(t) or (None, None))[0],
                conf=(reviews.get(t) or (None, None))[1],
                calls=(obligations.get(t) or {}).get("contracts", 0),
                free=round(
                    units - float((obligations.get(t) or {}).get("shares_committed") or 0), 2
                ),
                book=(books.get(t) or {}).get("yoy_pct"),
            )
        )
    holdings.sort(key=lambda h: -(h["value"] or 0))

    history: dict[str, list[dict[str, Any]]] = {}
    for t, rid, verdict, conf in history_rows:
        history.setdefault(t, []).append(
            dict(run=rid, d=run_days.get(rid, ""), v=verdict or "?", c=conf or 0)
        )

    # Only for tickers the page actually references. Every run adds picks,
    # so carrying prose for all of them would grow the file without bound
    # and none of it would be reachable.
    referenced = set(positions) | {s["ticker"] for s in suggestions}
    reasoning = {t: r for t, r in _reasoning(db).items() if t in referenced}
    graded = grade_suggestions(
        suggestions,
        today=today,
        units_now={t: p["units"] for t, p in positions.items()},
        fetch=fetch,
    )
    graded.sort(key=lambda g: (g["suggested_on"], g.get("id") or 0), reverse=True)

    return dict(
        generated=today.isoformat(),
        latest_run=run_id,
        holdings=holdings,
        history=history,
        views=views,
        runs=runs,
        record=coverage(db),
        suggestions=[
            dict(
                d=g["suggested_on"],
                ticker=g["ticker"],
                action=g["action"],
                run=g.get("run_id"),
                ret=g.get("return_pct"),
                spy=g.get("spy_pct"),
                swap=g.get("reinvest_into"),
                swap_pct=g.get("reinvest_pct"),
                edge=g.get("edge_pct"),
                verdict=g.get("verdict"),
                acted=g.get("acted"),
                detail=(g.get("detail") or "")[:150],
            )
            for g in graded
        ],
        holdings_ok=bool(positions),
        reasoning=reasoning,
        reviews={t: (r or "") for t, r in review_text.items()},
    )


def main(argv: list[str] | None = None) -> None:
    from ..dashboard_page import render_page

    parser = argparse.ArgumentParser(prog="dashboard", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=None, help="where to write the page")
    parser.add_argument("--open", action="store_true", help="print the path when done")
    args = parser.parse_args(argv)

    from ..market_time import use_market_timezone

    use_market_timezone()
    load_dotenv()
    settings = Settings.from_env()
    data = collect(settings, today=date.today())
    out = Path(args.out or settings.dashboard_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(data))
    logger.info(
        "Dashboard: %d holding(s), %d graded suggestion(s), %d price rows -> %s (%,d bytes)".replace(
            "%,d", "%d"
        ),
        len(data["holdings"]),
        len(data["suggestions"]),
        data["record"]["rows"],
        out,
        out.stat().st_size,
    )
    if args.open:
        print(out)


if __name__ == "__main__":
    main()
