"""Who among Wall Street's analysts acted on a stock, and how.

yfinance's `Ticker.upgrades_downgrades` is the full log of rating and
price-target actions — for ANET, 459 of them back to 2014, each dated with
the firm, the new and old rating and the new and old target. The nightly
earnings watch stores the part of it around each recorded earnings event
(`analyst_actions`), so a standout in the daily email can say which firms
raised their targets after the report. One request per stock. No LLM.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import text

from ..db.session import exec_sql, get_session
from ..db.tables import AnalystAction
from ..logging import get_logger
from . import yf_gateway

logger = get_logger(__name__)


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except TypeError, ValueError:
        return None
    return f if f > 0 and not pd.isna(f) else None


def fetch_analyst_actions(ticker: str) -> list[dict[str, Any]]:
    """Every action Yahoo has for `ticker`, newest first; [] when none."""
    df = yf_gateway.ticker_call(ticker, "upgrades_downgrades", lambda t: t.upgrades_downgrades)
    if df is None or df.empty:
        return []
    out = []
    for when, r in df.iterrows():
        out.append(
            {
                "graded_at": str(when)[:19],
                "firm": str(r.get("Firm") or "").strip(),
                "action": str(r.get("Action") or ""),
                "to_grade": str(r.get("ToGrade") or ""),
                "from_grade": str(r.get("FromGrade") or ""),
                "target_action": str(r.get("priceTargetAction") or ""),
                "target": _num(r.get("currentPriceTarget")),
                "prior_target": _num(r.get("priorPriceTarget")),
            }
        )
    return [a for a in out if a["firm"]]


def record_actions(db_path: str, ticker: str, actions: list[dict[str, Any]], *, since: date) -> int:
    """Store the actions on or after `since` not stored yet; returns how many."""
    keep = [a for a in actions if a["graded_at"][:10] >= since.isoformat()]
    added = 0
    with get_session(db_path) as session:
        for a in keep:
            if session.get(AnalystAction, (ticker, a["graded_at"], a["firm"])) is not None:
                continue
            session.add(AnalystAction(ticker=ticker, **a))
            added += 1
    return added


def summarize(db_path: str, ticker: str, *, since: date) -> dict[str, Any]:
    """What analysts did from `since` on: counts, the average target change
    among firms that moved theirs, and the actions themselves (newest first)."""
    with get_session(db_path) as session:
        rows = exec_sql(
            session,
            text(
                "SELECT graded_at, firm, action, to_grade, target_action, target, prior_target "
                "FROM analyst_actions WHERE ticker = :t AND graded_at >= :s ORDER BY graded_at DESC"
            ),
            params={"t": ticker, "s": since.isoformat()},
        ).all()
    actions = [
        {
            "day": g[:10],
            "firm": firm,
            "action": action,
            "to_grade": to_grade,
            "target_action": target_action,
            "target": target,
            "prior_target": prior,
        }
        for g, firm, action, to_grade, target_action, target, prior in rows
    ]
    changes = [
        a["target"] / a["prior_target"] - 1
        for a in actions
        if a["target"] and a["prior_target"] and a["target_action"] in ("Raises", "Lowers")
    ]
    return {
        "count": len(actions),
        "raised": sum(a["target_action"] == "Raises" for a in actions),
        "lowered": sum(a["target_action"] == "Lowers" for a in actions),
        "upgrades": sum(a["action"] == "up" for a in actions),
        "downgrades": sum(a["action"] == "down" for a in actions),
        "initiated": sum(a["action"] == "init" for a in actions),
        "avg_target_change_pct": sum(changes) / len(changes) * 100 if changes else None,
        "actions": actions,
    }


def describe(a: dict[str, Any]) -> str:
    """One action as a line: 'Aug 06 Barclays: target $195 → $289 (Overweight)'."""
    day = date.fromisoformat(a["day"]).strftime("%b %d")
    what = {"up": "upgrade", "down": "downgrade", "init": "starts coverage"}.get(a["action"], "")
    target = ""
    if a["target"] and a["prior_target"]:
        target = f"target ${a['prior_target']:,.0f} → ${a['target']:,.0f}"
    elif a["target"]:
        target = f"target ${a['target']:,.0f}"
    bits = ", ".join(b for b in (what, target) if b)
    grade = f" ({a['to_grade']})" if a["to_grade"] else ""
    return f"{day} {a['firm']}: {bits or a['target_action'] or 'maintained'}{grade}"
