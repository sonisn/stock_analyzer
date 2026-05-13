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
import re
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
    # check_same_thread=False is safe here: the handle is opened in
    # read-only URI mode and Streamlit serializes script reruns per
    # session — concurrent reads against a ro handle are the textbook
    # safe SQLite case.
    conn = sqlite3.connect(
        f"file:{_DB_PATH}?mode=ro",
        uri=True,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    return conn


def _q(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    with closing(_conn().cursor()) as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


@st.cache_resource
def _columns_by_table() -> dict[str, set[str]]:
    """Reflect each table's columns so queries can degrade gracefully on
    older DBs that haven't been migrated to include `kind` / `rebalance_text`
    / `dashboard_data`."""
    out: dict[str, set[str]] = {}
    for table in ("runs", "run_outputs", "picks", "candidates", "scorecards"):
        try:
            rows = list(_conn().execute(f"PRAGMA table_info({table})"))
            out[table] = {r[1] for r in rows}
        except sqlite3.OperationalError:
            out[table] = set()
    return out


def _has_col(table: str, col: str) -> bool:
    return col in _columns_by_table().get(table, set())


def _row_get(row: sqlite3.Row | None, key: str, default: Any = None) -> Any:
    """Safe attribute-style access — returns default if the column is absent
    (legacy DB) instead of raising IndexError."""
    if row is None:
        return default
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


def _list_runs(limit: int = 100) -> list[sqlite3.Row]:
    cols = ["id", "run_at"]
    if _has_col("runs", "kind"):
        cols.append("kind")
    cols.extend(["picks", "survivors", "universe_size", "cash_budget"])
    sql = (
        f"SELECT {', '.join(cols)} FROM runs ORDER BY id DESC LIMIT ?"
    )
    return _q(sql, (limit,))


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
    if not _has_col("run_outputs", "dashboard_data"):
        return {}
    out = _run_outputs(run_id)
    if not out or not _row_get(out, "dashboard_data"):
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
        kind = _row_get(r, "kind", "—")
        rows.append(
            {
                "Run ID": r["id"],
                "Date": r["run_at"],
                "Type": kind,
                "Picks": r["picks"],
                "Survivors": r["survivors"],
                "Universe": r["universe_size"],
                "Status": (
                    _status_from_outputs(r["id"]) if kind == "rebalance" else "—"
                ),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Drill in")
    run_ids = [r["id"] for r in runs]
    labels = [
        f"#{r['id']}  {r['run_at']}  "
        f"({_row_get(r, 'kind', 'run')}, {r['picks']} picks)"
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


# --- Visual helpers ---------------------------------------------------------

# Status colors — match the report's status-banner palette
_STATUS_PALETTE = {
    "NO_ACTION": ("#0e6432", "#e6f4ea", "NO ACTION RECOMMENDED"),
    "ACTION":    ("#8a4a00", "#fff4e0", "ACTION RECOMMENDED"),
    "UNKNOWN":   ("#444",    "#f0f0f0", "STATUS UNKNOWN"),
}

# Split LLM output text into per-ticker blocks. The pipeline emits
# `---` separators between blocks; first line is either
# `PICK N: TICKER — one-liner` or `TICKER: XYZ`.
# Match `---` on its own line, optionally at the start of the text.
_BLOCK_SPLIT_RE = re.compile(r"(?:^|\n)-{3,}(?:\n+|$)")
_PICK_FIRST_LINE_RE = re.compile(
    r"^PICK\s+(\d+):\s+([A-Z][A-Z.\-]{0,5})\s*[—–-]\s*(.+)$"
)
_TICKER_FIRST_LINE_RE = re.compile(r"^TICKER:\s*([A-Z][A-Z.\-]{0,5})\s*$")


def _badge(text: str, fg: str, bg: str) -> str:
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:10px;font-size:0.85em;font-weight:600;">{text}</span>'
    )


def _split_blocks(text: str) -> list[str]:
    if not text:
        return []
    raw = _BLOCK_SPLIT_RE.split(text)
    return [b.strip() for b in raw if b and b.strip()]


def _block_title(block: str) -> tuple[str | None, str | None, str]:
    """Return (ticker, subtitle, rest_of_block).

    Recognizes `PICK N: TICKER — line` and `TICKER: XYZ` first lines.
    Anything else returns ticker=None and the whole block as rest."""
    lines = block.split("\n", 1)
    first = lines[0].strip()
    rest = lines[1].strip() if len(lines) > 1 else ""
    m = _PICK_FIRST_LINE_RE.match(first)
    if m:
        return m.group(2), m.group(3), rest
    m = _TICKER_FIRST_LINE_RE.match(first)
    if m:
        return m.group(1), None, rest
    return None, None, block


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
    kind = _row_get(run, "kind", "run")

    _render_run_header(run_id, run, kind)
    _render_status_banner(data, kind)
    _render_metrics(run, data)
    if kind == "rebalance" and data.get("holdings"):
        _render_holdings_dashboard(data["holdings"])
    if data.get("sector_breakdown"):
        _render_sector_breakdown(data["sector_breakdown"])
    _render_picks_grid(run_id)
    _render_analysis_tabs(outputs, data)
    _render_pdf_link(data, kind, run["run_at"])


def _render_run_header(run_id: int, run: sqlite3.Row, kind: str) -> None:
    kind_color = {
        "rebalance": ("#8a4a00", "#fff4e0"),
        "discover":  ("#0c5e7c", "#e8f4f8"),
    }.get(kind, ("#444", "#f0f0f0"))
    badge = _badge(kind.upper(), *kind_color)
    st.markdown(
        f"## Run #{run_id} &nbsp; {badge}",
        unsafe_allow_html=True,
    )
    st.caption(f"{run['run_at']}")


def _render_status_banner(data: dict[str, Any], kind: str) -> None:
    if kind != "rebalance":
        return
    status = data.get("status")
    if not status:
        return
    _, _, label = _STATUS_PALETTE.get(status, _STATUS_PALETTE["UNKNOWN"])
    if status == "NO_ACTION":
        st.success(f"**Status — {label}**")
    elif status == "ACTION":
        st.warning(f"**Status — {label}**")
    else:
        st.info(f"**Status — {label}**")


def _render_metrics(run: sqlite3.Row, data: dict[str, Any]) -> None:
    metrics = data.get("metrics") or {}
    # Only render the metric strip if at least one metric has data.
    cash = metrics.get("cash") if metrics.get("cash") is not None else _row_get(
        run, "cash_budget"
    )
    has_any = (
        metrics.get("holdings_count")
        or metrics.get("total_value")
        or metrics.get("total_pnl_pct") is not None
        or cash is not None
    )
    if not has_any:
        return
    cols = st.columns(4)
    cols[0].metric("Holdings", metrics.get("holdings_count") or "—")
    cols[1].metric(
        "Portfolio value",
        f"${metrics['total_value']:,.0f}" if metrics.get("total_value") else "—",
    )
    pnl = metrics.get("total_pnl_pct")
    cols[2].metric(
        "Total P/L", f"{pnl:+.1f}%" if pnl is not None else "—",
    )
    cols[3].metric("Cash", f"${cash:,.0f}" if cash is not None else "—")


def _render_holdings_dashboard(holdings: dict[str, dict[str, Any]]) -> None:
    st.subheader("Holdings")
    rows = []
    for ticker, h in sorted(holdings.items()):
        pnl = h.get("pnl_pct")
        rows.append(
            {
                "Ticker": ticker,
                "Verdict": h.get("verdict") or "—",
                "Conf": h.get("confidence") or "—",
                "P/L %": pnl if pnl is not None else None,
                "Value": h.get("value") or 0,
                "Sector": h.get("sector") or "—",
            }
        )
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "P/L %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Value": st.column_config.NumberColumn(format="$%,.0f"),
        },
    )


def _render_sector_breakdown(breakdown: list[dict[str, Any]]) -> None:
    st.subheader("Sector allocation")
    by_sector = {row["sector"]: row["value"] for row in breakdown}
    st.bar_chart(by_sector, horizontal=True)


def _render_picks_grid(run_id: int) -> None:
    picks = _picks(run_id)
    if not picks:
        return
    st.subheader(f"Picks ({len(picks)})")
    cols_per_row = 5 if len(picks) >= 5 else len(picks)
    for i in range(0, len(picks), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, pick in enumerate(picks[i:i + cols_per_row]):
            with cols[j]:
                with st.container(border=True):
                    st.markdown(
                        _badge(f"#{pick['rank']}", "#fff", "#3b8fde"),
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"### `{pick['ticker']}`")


def _render_analysis_tabs(
    outputs: sqlite3.Row | None, data: dict[str, Any]
) -> None:
    if not outputs:
        return
    sections: list[tuple[str, str]] = []
    if _row_get(outputs, "rebalance_text"):
        sections.append(("Rebalance plan", outputs["rebalance_text"]))
    if _row_get(outputs, "ranker_full"):
        sections.append(("Ranker — picks", outputs["ranker_full"]))
    if _row_get(outputs, "redteam_full"):
        sections.append(("Red team — bear cases", outputs["redteam_full"]))
    if _row_get(outputs, "sizer_full"):
        sections.append(("Sizer — allocation", outputs["sizer_full"]))
    holdings = data.get("holdings") or {}
    if any(h.get("review_text") for h in holdings.values()):
        sections.append(("Per-holding reviews", _holdings_reviews_as_text(holdings)))
    if not sections:
        return
    st.subheader("Analysis")
    tabs = st.tabs([title for title, _ in sections])
    for tab, (_title, text) in zip(tabs, sections):
        with tab:
            _render_text_as_cards(text)


def _holdings_reviews_as_text(holdings: dict[str, dict[str, Any]]) -> str:
    """Stitch per-holding reviews into one ticker-separated text body so
    the same `_render_text_as_cards` pipeline handles them."""
    parts: list[str] = []
    for ticker in sorted(holdings.keys()):
        review = (holdings[ticker].get("review_text") or "").strip()
        if not review:
            continue
        parts.append(f"TICKER: {ticker}\n{review}")
    return "\n---\n".join(parts)


def _render_text_as_cards(text: str) -> None:
    """Split text by `---` markers and render each block in its own
    bordered card. First line drives the card title (PICK N: TICKER or
    TICKER: XYZ); the rest renders as markdown so paragraph breaks and
    label-style lines look like prose, not a monospace dump."""
    if not text:
        st.caption("No content.")
        return
    blocks = _split_blocks(text)
    if not blocks:
        # Fallback: render the whole text as one card.
        with st.container(border=True):
            st.markdown(text)
        return
    for block in blocks:
        ticker, subtitle, rest = _block_title(block)
        with st.container(border=True):
            if ticker:
                header_bits = [f"### `{ticker}`"]
                if subtitle:
                    header_bits.append(f"_{subtitle}_")
                st.markdown("  ·  ".join(header_bits))
                st.markdown(_lightly_format(rest))
            else:
                st.markdown(_lightly_format(rest))


# Lines like "Bull thesis:" or "Most fragile assumption in the bull thesis:"
# read better as bold labels than plain text. Bound the length and require
# upper-case start so we don't bold every random colon.
_LABEL_LINE_RE = re.compile(
    r"^([A-Z][A-Za-z0-9 ()/&\-]{2,60}):(?=\s*$|\s+\S)", re.MULTILINE
)


def _lightly_format(text: str) -> str:
    """Convert plain-text 'Label:' lines into **Label:** so Streamlit's
    markdown rendering gives the block visible structure."""
    return _LABEL_LINE_RE.sub(r"**\1:**", text)


def _render_pdf_link(
    data: dict[str, Any], kind: str, run_at: str
) -> None:
    pdf_filename = data.get("pdf_filename")
    if not pdf_filename:
        run_date = (run_at or "").split("T")[0]
        prefix = "rebalance" if kind == "rebalance" else "discover"
        pdf_filename = f"{prefix}-{run_date}.pdf"
    pdf_path = _REPORTS_DIR / pdf_filename
    if pdf_path.exists():
        st.divider()
        st.download_button(
            "Download PDF",
            data=pdf_path.read_bytes(),
            file_name=pdf_filename,
            mime="application/pdf",
        )
    # If the PDF isn't on disk we just don't show the button — no noisy
    # "tech error" caption. The browser view IS the report.


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
