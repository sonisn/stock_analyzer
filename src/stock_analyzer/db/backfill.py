"""Fill forecast fields that older runs recorded but never stored as columns.

Picks from before the forecast columns existed (and rebalance-run picks,
which skipped them until 2026-09) have NULL conviction / time_horizon /
entry_price, so calibration and the thesis check ignored them. The values
were decided on the run date and are still on disk:

  - conviction and time horizon: the run's own Ranker text
    (`run_outputs.ranker_full`) states them in each "PICK n:" block;
  - entry price: the run's screen-time price for that ticker
    (`candidates.price`), which is what the pipeline stores today.

Only NULL fields are filled, so this is safe to re-run. Anything the old
runs never produced (EV, scenarios, consensus votes, structured
catalysts) is left empty rather than reconstructed with hindsight.
"""

from __future__ import annotations

import re

from sqlalchemy import text

from .session import get_session

_PICK_BLOCK = re.compile(r"^PICK\s+(\d+):\s+([A-Z][A-Z.\-]{0,5})\b", re.MULTILINE)
_CONVICTION = re.compile(r"^Conviction \(1-10\):\s*(\d+)", re.MULTILINE)
_HORIZON = re.compile(r"^Time horizon:\s*(.+?)\s*$", re.MULTILINE)


def parse_pick_forecasts(ranker_text: str) -> dict[str, dict[str, object]]:
    """{ticker: {"conviction", "time_horizon"}} from a Ranker text's PICK
    blocks; a field is omitted when its line is missing or out of range."""
    out: dict[str, dict[str, object]] = {}
    matches = list(_PICK_BLOCK.finditer(ranker_text or ""))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(ranker_text)
        block = ranker_text[m.start() : end]
        fields: dict[str, object] = {}
        if (c := _CONVICTION.search(block)) and 1 <= int(c.group(1)) <= 10:
            fields["conviction"] = int(c.group(1))
        if h := _HORIZON.search(block):
            fields["time_horizon"] = h.group(1)[:40]
        out[m.group(2)] = fields
    return out


def backfill_pick_forecasts(db_path: str) -> dict[str, int]:
    """Fill NULL conviction / time_horizon / entry_price on past picks.
    Returns how many values were filled per field."""
    filled = {"conviction": 0, "time_horizon": 0, "entry_price": 0}
    with get_session(db_path) as session:
        rows = session.exec(
            text(
                "SELECT p.run_id, p.ticker, p.conviction, p.time_horizon, p.entry_price, "
                "o.ranker_full, c.price FROM picks p "
                "LEFT JOIN run_outputs o ON o.run_id = p.run_id "
                "LEFT JOIN candidates c ON c.run_id = p.run_id AND c.ticker = p.ticker "
                "WHERE p.conviction IS NULL OR p.time_horizon IS NULL OR p.entry_price IS NULL"
            )
        ).all()
        parsed: dict[int, dict[str, dict[str, object]]] = {}
        for run_id, ticker, conviction, horizon, entry, ranker_full, price in rows:
            if run_id not in parsed:
                parsed[run_id] = parse_pick_forecasts(ranker_full or "")
            found = parsed[run_id].get(ticker, {})
            updates: dict[str, object] = {}
            if conviction is None and "conviction" in found:
                updates["conviction"] = found["conviction"]
            if horizon is None and "time_horizon" in found:
                updates["time_horizon"] = found["time_horizon"]
            if entry is None and price:
                updates["entry_price"] = float(price)
            if not updates:
                continue
            assignments = ", ".join(f"{k} = :{k}" for k in updates)
            session.exec(
                text(f"UPDATE picks SET {assignments} WHERE run_id = :run_id AND ticker = :ticker"),
                params={**updates, "run_id": run_id, "ticker": ticker},
            )
            for k in updates:
                filled[k] += 1
    return filled
