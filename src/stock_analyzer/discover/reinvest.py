"""Where sale proceeds should go — no LLM.

Every suggestion to sell comes with a destination for the money, drawn
from what the pipeline already stands behind:

  - `reinvest_ideas`: the most recent discover picks the user doesn't
    hold, newest run and best rank first, skipping sectors already over
    the cap and picks whose thesis has broken;
  - `sector_peers`: same-sector names from the latest screen (passed its
    hard filter, best score first) — the swap that keeps a tax-loss
    sale's market exposure without buying back the same stock.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from ..db.session import get_session
from ..logging import get_logger

logger = get_logger(__name__)


def load_pick_pool(db_path: str, *, n_runs: int = 3) -> list[dict[str, Any]]:
    """Picks from the last `n_runs` runs that made picks, newest first,
    one row per ticker (its latest pick), with sector and screen price."""
    with get_session(db_path) as session:
        rows = session.exec(
            text(
                "SELECT p.ticker, p.rank, r.run_at, c.sector, c.price, p.conviction "
                "FROM picks p JOIN runs r ON r.id = p.run_id "
                "LEFT JOIN candidates c ON c.run_id = p.run_id AND c.ticker = p.ticker "
                "WHERE p.run_id IN (SELECT DISTINCT run_id FROM picks "
                "                   ORDER BY run_id DESC LIMIT :n) "
                "ORDER BY p.run_id DESC, p.rank ASC"
            ),
            params={"n": n_runs},
        ).all()
    out: dict[str, dict[str, Any]] = {}
    for ticker, rank, run_at, sector, price, conviction in rows:
        out.setdefault(
            ticker,
            {
                "ticker": ticker,
                "rank": rank,
                "pick_date": str(run_at)[:10],
                "sector": sector,
                "price": price,
                "conviction": conviction,
            },
        )
    return list(out.values())


def reinvest_ideas(
    pool: list[dict[str, Any]],
    *,
    held: set[str],
    avoid_sectors: set[str] = frozenset(),
    exclude: set[str] = frozenset(),
    n: int = 2,
) -> list[dict[str, Any]]:
    """The first `n` pool entries not held, not excluded (e.g. a broken
    thesis) and not in a sector that is already over the cap."""
    out = []
    for idea in pool:
        t = idea["ticker"]
        if t in held or t in exclude or (idea.get("sector") in avoid_sectors):
            continue
        out.append(idea)
        if len(out) >= n:
            break
    return out


def sector_peers(
    db_path: str, tickers: list[str], *, held: set[str], n: int = 3
) -> dict[str, dict[str, list[str]]]:
    """{ticker: {"peers": [same-sector names]}} from the latest run that
    screened each ticker's sector — the shape tax_harvest expects. Held
    names are left out (the swap should add new exposure)."""
    if not tickers:
        return {}
    with get_session(db_path) as session:
        rows = session.exec(
            text(
                "SELECT c.ticker, c.sector, c.score, c.passed_filter, c.run_id "
                "FROM candidates c "
                "WHERE c.sector IS NOT NULL AND c.run_id = "
                "  (SELECT MAX(run_id) FROM candidates WHERE sector IS NOT NULL)"
            )
        ).all()
        sectors = dict(
            session.exec(
                text(
                    "SELECT ticker, sector FROM candidates WHERE sector IS NOT NULL "
                    "ORDER BY run_id ASC"
                )
            ).all()
        )
    by_sector: dict[str, list[tuple[float, str]]] = {}
    for ticker, sector, score, passed, _run in rows:
        if passed and ticker not in held:
            by_sector.setdefault(sector, []).append((score or 0.0, ticker))
    out: dict[str, dict[str, list[str]]] = {}
    for t in tickers:
        mates = sorted(by_sector.get(sectors.get(t) or "", []), reverse=True)
        peers = [m for _, m in mates if m != t][:n]
        if peers:
            out[t] = {"peers": peers}
    return out


def format_idea(idea: dict[str, Any]) -> str:
    """'ANET (pick #1, 2026-09-17, Technology)'."""
    parts = [f"pick #{idea['rank']}", idea["pick_date"]]
    if idea.get("sector"):
        parts.append(idea["sector"])
    return f"{idea['ticker']} ({', '.join(parts)})"


def unfunded_sales(plan: Any) -> list[str]:
    """Tickers a plan SELLs/TRIMs when it deploys nothing (no BUY, ADD or
    SELL_PUT) — the proceeds would otherwise have no named destination."""
    actions = getattr(plan, "actions", None) or []
    if any(a.action in ("BUY", "ADD", "SELL_PUT") for a in actions):
        return []
    return [a.ticker for a in actions if a.action in ("SELL", "TRIM")]
