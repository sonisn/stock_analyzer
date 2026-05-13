"""Streamlit dashboard — browse past discover + rebalance runs.

Install the optional dependency once:
    uv pip install -e ".[dashboard]"

Run:
    streamlit run src/stock_analyzer/experimental/dashboard.py

Reads from `~/.stock_analyzer/discover.db` (the same SQLite the pipeline
already writes to). No pipeline changes required — the dashboard is a
pure read-side view of what's already persisted.

v1 has two pages:
  - Run history (home)        — sortable table of every past run
  - Run detail (drill-in)     — metric strip + holdings dashboard +
                                sector pie + collapsible LLM-output
                                sections + PDF download

Pick a run from the history table to drill in.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# Honor the project's .env so DISCOVER_DB_PATH / REPORTS_DIR / LOG_FILE
# work the same way they do for the CLI pipelines — no need to re-export
# them in the shell that launches Streamlit.
load_dotenv()


_DB_PATH = Path(
    os.path.expanduser(os.getenv("DISCOVER_DB_PATH", "~/.stock_analyzer/discover.db"))
)
_REPORTS_DIR = Path(
    os.path.expanduser(os.getenv("REPORTS_DIR", "~/.stock_analyzer/reports"))
)


# --- DB helpers -------------------------------------------------------------


@st.cache_resource
def _conn() -> sqlite3.Connection:
    if not _DB_PATH.exists():
        st.error(
            f"Database not found at `{_DB_PATH}`.\n\n"
            "Fixes:\n"
            "1. Run the discover or rebalance pipeline at least once on "
            "**this machine** to populate the DB, OR\n"
            "2. Point the dashboard at a DB that already exists by setting "
            "`DISCOVER_DB_PATH` (in `.env` or as an environment variable "
            "when launching streamlit), e.g.\n\n"
            "```\n"
            "DISCOVER_DB_PATH=/path/to/discover.db streamlit run "
            "src/stock_analyzer/experimental/dashboard.py\n"
            "```\n\n"
            "If the pipeline runs on a different machine, you'll need the "
            "DB file to be reachable from this host (rsync, scp, NFS, "
            "SSHFS, etc.)."
        )
        st.stop()
    conn = sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _q(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    with closing(_conn().cursor()) as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def _list_runs(limit: int = 100) -> list[sqlite3.Row]:
    return _q(
        "SELECT id, run_at, kind, picks, survivors, universe_size, cash_budget "
        "FROM runs ORDER BY id DESC LIMIT ?",
        (limit,),
    )


def _run(run_id: int) -> sqlite3.Row | None:
    rows = _q("SELECT * FROM runs WHERE id = ?", (run_id,))
    return rows[0] if rows else None


def _run_outputs(run_id: int) -> sqlite3.Row | None:
    rows = _q("SELECT * FROM run_outputs WHERE run_id = ?", (run_id,))
    return rows[0] if rows else None


def _picks(run_id: int) -> list[sqlite3.Row]:
    return _q(
        "SELECT rank, ticker FROM picks WHERE run_id = ? ORDER BY rank",
        (run_id,),
    )


def _dashboard_data(run_id: int) -> dict[str, Any]:
    """Parse the dashboard_data JSON column; return {} if absent (older run)."""
    out = _run_outputs(run_id)
    if not out or not out["dashboard_data"]:
        return {}
    try:
        return json.loads(out["dashboard_data"])
    except (json.JSONDecodeError, TypeError):
        return {}


def _status_from_outputs(run_id: int) -> str:
    data = _dashboard_data(run_id)
    return data.get("status") or "—"


# --- Pages ------------------------------------------------------------------


def _page_run_history() -> None:
    st.header("Run history")
    runs = _list_runs(100)
    if not runs:
        st.info("No runs yet. Run the discover or rebalance pipeline first.")
        return

    rows = []
    for r in runs:
        rows.append(
            {
                "Run ID": r["id"],
                "Date": r["run_at"],
                "Type": r["kind"],
                "Picks": r["picks"],
                "Survivors": r["survivors"],
                "Universe": r["universe_size"],
                "Status": (
                    _status_from_outputs(r["id"]) if r["kind"] == "rebalance" else "—"
                ),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Drill in")
    run_ids = [r["id"] for r in runs]
    labels = [
        f"#{r['id']}  {r['run_at']}  ({r['kind']}, {r['picks']} picks)"
        for r in runs
    ]
    choice = st.selectbox(
        "Pick a run", options=run_ids, format_func=lambda rid: labels[run_ids.index(rid)]
    )
    if st.button("Open run", type="primary"):
        st.query_params["run"] = str(choice)
        st.rerun()

    _page_picks_frequency(runs)


def _page_picks_frequency(runs: list[sqlite3.Row]) -> None:
    if not runs:
        return
    st.markdown("---")
    st.subheader("Picks frequency (last 90 days)")
    run_ids = [r["id"] for r in runs[:90]]  # not actually 90d, but recent
    if not run_ids:
        return
    placeholders = ",".join(["?"] * len(run_ids))
    rows = _q(
        f"SELECT ticker, COUNT(*) AS n FROM picks WHERE run_id IN ({placeholders}) "
        f"GROUP BY ticker ORDER BY n DESC LIMIT 20",
        tuple(run_ids),
    )
    if not rows:
        st.caption("No picks in the recent window.")
        return
    st.bar_chart(
        {"count": {r["ticker"]: r["n"] for r in rows}}, horizontal=True
    )


def _page_run_detail(run_id: int) -> None:
    run = _run(run_id)
    if not run:
        st.error(f"Run #{run_id} not found.")
        return

    if st.button("← Back to history"):
        st.query_params.clear()
        st.rerun()

    data = _dashboard_data(run_id)
    outputs = _run_outputs(run_id)

    title_status = data.get("status") or ""
    title_suffix = f"  ·  Status: {title_status}" if title_status else ""
    st.header(f"Run #{run_id}  ·  {run['run_at']}  ·  {run['kind']}{title_suffix}")

    _render_metrics(run, data)
    if run["kind"] == "rebalance" and data.get("holdings"):
        _render_holdings_dashboard(data["holdings"])
    if data.get("sector_breakdown"):
        _render_sector_breakdown(data["sector_breakdown"])

    _render_picks_section(run_id)
    _render_outputs_sections(outputs, data)
    _render_pdf_link(data, run["kind"], run["run_at"])


def _render_metrics(run: sqlite3.Row, data: dict[str, Any]) -> None:
    metrics = data.get("metrics") or {}
    cols = st.columns(4)
    cols[0].metric(
        "Holdings",
        metrics.get("holdings_count") or "—",
    )
    cols[1].metric(
        "Portfolio value",
        f"${metrics['total_value']:,.0f}" if metrics.get("total_value") else "—",
    )
    pnl = metrics.get("total_pnl_pct")
    cols[2].metric(
        "Total P/L",
        f"{pnl:+.1f}%" if pnl is not None else "—",
    )
    cash = metrics.get("cash") if metrics.get("cash") is not None else run["cash_budget"]
    cols[3].metric(
        "Cash",
        f"${cash:,.0f}" if cash is not None else "—",
    )


def _render_holdings_dashboard(holdings: dict[str, dict[str, Any]]) -> None:
    st.subheader("Holdings dashboard")
    rows = []
    for ticker, h in sorted(holdings.items()):
        pnl = h.get("pnl_pct")
        rows.append(
            {
                "Ticker": ticker,
                "Verdict": h.get("verdict") or "—",
                "Conf": h.get("confidence") or "—",
                "P/L %": f"{pnl:+.1f}%" if pnl is not None else "—",
                "Value": (
                    f"${h['value']:,.0f}"
                    if h.get("value") is not None else "—"
                ),
                "Sector": h.get("sector") or "—",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_sector_breakdown(breakdown: list[dict[str, Any]]) -> None:
    st.subheader("Sector allocation")
    by_sector = {row["sector"]: row["value"] for row in breakdown}
    st.bar_chart(by_sector, horizontal=True)


def _render_picks_section(run_id: int) -> None:
    picks = _picks(run_id)
    if not picks:
        return
    with st.expander(f"Picks ({len(picks)})", expanded=False):
        for p in picks:
            st.markdown(f"**Rank {p['rank']}**  ·  `{p['ticker']}`")


def _render_outputs_sections(
    outputs: sqlite3.Row | None, data: dict[str, Any]
) -> None:
    if not outputs:
        return
    if outputs["rebalance_text"]:
        with st.expander("Rebalance plan", expanded=False):
            st.text(outputs["rebalance_text"])
    if outputs["ranker_full"]:
        with st.expander("Ranker (discover picks)", expanded=False):
            st.text(outputs["ranker_full"])
    if outputs["redteam_full"]:
        with st.expander("Red team (bear cases)", expanded=False):
            st.text(outputs["redteam_full"])
    if outputs["sizer_full"]:
        with st.expander("Sizer (allocation)", expanded=False):
            st.text(outputs["sizer_full"])
    holdings = data.get("holdings") or {}
    review_count = sum(1 for h in holdings.values() if h.get("review_text"))
    if review_count:
        with st.expander(f"Per-holding reviews ({review_count})", expanded=False):
            for ticker in sorted(holdings.keys()):
                review = holdings[ticker].get("review_text")
                if not review:
                    continue
                st.markdown(f"### {ticker}")
                st.text(review)


def _render_pdf_link(
    data: dict[str, Any], kind: str, run_at: str
) -> None:
    pdf_filename = data.get("pdf_filename")
    if not pdf_filename:
        # Fall back to the deterministic filename the CLIs use.
        run_date = (run_at or "").split("T")[0]
        prefix = "rebalance" if kind == "rebalance" else "discover"
        pdf_filename = f"{prefix}-{run_date}.pdf"
    pdf_path = _REPORTS_DIR / pdf_filename
    st.markdown("---")
    if pdf_path.exists():
        st.download_button(
            "Download PDF",
            data=pdf_path.read_bytes(),
            file_name=pdf_filename,
            mime="application/pdf",
        )
    else:
        st.caption(f"PDF not on disk at {pdf_path}")


# --- Entry point ------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Stock Analyzer Dashboard",
        layout="wide",
        page_icon="📈",
    )
    st.title("Stock Analyzer Dashboard")
    st.caption(f"Reading from `{_DB_PATH}`")

    params = st.query_params
    run_id = params.get("run")
    if run_id:
        try:
            _page_run_detail(int(run_id))
            return
        except ValueError:
            st.warning(f"Invalid run id: {run_id!r}")

    _page_run_history()


if __name__ == "__main__":
    main()
