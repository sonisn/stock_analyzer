"""Between-run thesis check for open picks (no LLM)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text

from stock_analyzer.db.repository import insert_pick, insert_pick_catalysts, insert_run
from stock_analyzer.db.session import get_session
from stock_analyzer.discover.report_html import render_html_email
from stock_analyzer.discover.report_pdf import render_pdf
from stock_analyzer.discover.report_sections import append_thesis_check_section
from stock_analyzer.discover.thesis_tracker import (
    OpenPick,
    check_theses,
    load_open_picks,
    thesis_report_data,
)

TODAY = date(2026, 9, 1)
PICKED = date(2026, 7, 1)


def _path(before: float, at_pick: float, now: float, *, events=None) -> pd.DataFrame:
    """Business-day closes: flat at `before` for a year, stepping to
    `at_pick` on the pick date, then a straight line to `now`. `events`
    maps a date to an extra multiplicative jump from that day on."""
    idx = pd.bdate_range(PICKED - timedelta(days=400), TODAY)
    span = (TODAY - PICKED).days
    closes = []
    for ts in idx:
        d = ts.date()
        px = before if d < PICKED else at_pick + (now - at_pick) * (d - PICKED).days / span
        for when, jump in (events or {}).items():
            if d >= when:
                px *= jump
        closes.append(px)
    return pd.DataFrame({"Close": closes}, index=idx)


def _pick(ticker: str, *, bear=-20.0, bull=40.0, catalysts=()) -> OpenPick:
    return OpenPick(
        run_id=1,
        ticker=ticker,
        pick_date=PICKED,
        entry_price=100.0,
        bear_target_pct=bear,
        bull_target_pct=bull,
        catalysts=tuple(catalysts),
    )


def _check(pick: OpenPick, frame: pd.DataFrame, spy: pd.DataFrame | None = None, **kw):
    frames = {"SPY": spy if spy is not None else _path(100, 100, 102), pick.ticker: frame}
    return check_theses([pick], today=TODAY, fetch=lambda t, s, e: frames.get(t), **kw)


CUTTING = {"AAA": {"direction_30d": "lowering", "net_revisions_30d": -3}}


def test_bear_target_breach_alone_is_watch():
    # Long-term hold: a price drop alone is a reason to look, not to sell.
    [c] = _check(_pick("AAA"), _path(100, 100, 75))
    assert c.status == "WATCH"
    assert "past its own bear case (-20%)" in c.signals[0].text


def test_bear_target_breach_with_estimate_cuts_is_broken():
    [c] = _check(_pick("AAA"), _path(100, 100, 75), eps_revisions=CUTTING)
    assert c.status == "BROKEN"
    assert "bear case (-20%)" in c.signals[0].text
    assert "analysts cutting EPS estimates (net -3" in c.signals[0].text


def test_annualized_targets_compound_over_at_least_a_year():
    # -12%/yr bear, two months in: judged against a full year (-12%), so
    # a -10% dip is not yet past it; -15% is.
    pick = OpenPick(
        run_id=1,
        ticker="AAA",
        pick_date=PICKED,
        entry_price=100.0,
        bear_target_pct=-12.0,
        bull_target_pct=25.0,
        annualized=True,
    )
    [c] = _check(pick, _path(80, 100, 90))
    assert not any("bear case" in s.text for s in c.signals)
    [c] = _check(pick, _path(80, 100, 85))
    assert "past its own bear case (-12%/yr)" in c.signals[0].text
    [c] = _check(pick, _path(80, 100, 126))
    assert c.status == "TARGET HIT"


def test_bull_target_reached():
    [c] = _check(_pick("AAA"), _path(100, 100, 145))
    assert c.status == "TARGET HIT"


def test_below_200dma_while_lagging_spy():
    # Long history at 120 keeps the 200-day mean above today's 88.
    [c] = _check(_pick("AAA", bear=-30), _path(120, 100, 88))
    assert c.status == "WATCH"
    assert "200-day average" in c.signals[0].text
    [c] = _check(_pick("AAA", bear=-30), _path(120, 100, 88), eps_revisions=CUTTING)
    assert c.status == "BROKEN"


def test_lagging_spy_alone_is_watch():
    [c] = _check(_pick("AAA"), _path(80, 100, 101), spy=_path(100, 100, 115))
    assert c.status == "WATCH"
    assert c.signals[0].text.startswith("Lagging SPY by 14.0 pts")


def test_intact_pick_has_no_signals():
    [c] = _check(_pick("AAA"), _path(80, 100, 110))
    assert c.status == "INTACT" and c.signals == []


def test_failed_positive_catalyst_and_upcoming_event():
    event = date(2026, 8, 3)
    catalysts = [
        {"event": "Q2 earnings", "expected_date": event.isoformat(), "direction": "positive"},
        {"event": "Investor day", "expected_date": "2026-09-08", "direction": "positive"},
    ]
    frame = _path(80, 100, 112, events={event: 0.9})
    [c] = _check(_pick("AAA", catalysts=catalysts), frame)
    assert c.status == "WATCH"
    texts = [s.text for s in c.signals]
    assert any(t.startswith("Catalyst went the wrong way: Q2 earnings") for t in texts)
    assert any(t.startswith("Upcoming in 7d: Investor day") for t in texts)


def test_eps_cuts_flag_watch_and_missing_history_is_skipped():
    frames = {"SPY": _path(100, 100, 102), "AAA": _path(80, 100, 110)}
    checks = check_theses(
        [_pick("AAA"), _pick("GONE")],
        today=TODAY,
        eps_revisions={"AAA": {"direction_30d": "lowering", "net_revisions_30d": -4}},
        fetch=lambda t, s, e: frames.get(t),
    )
    assert [c.ticker for c in checks] == ["AAA"]
    assert checks[0].signals[0].text == "Analysts cutting EPS estimates (net -4 revisions in 30d)"


def test_load_open_picks_keeps_latest_thesis_per_ticker(tmp_path):
    db = str(tmp_path / "t.db")

    def seed(run_at: str, entry: float, bear: float, catalyst_date: str) -> None:
        with get_session(db) as session:
            run_id = insert_run(
                session,
                universe_size=1,
                survivors=1,
                picks=1,
                opus_model="o",
                sonnet_model="s",
                cash_budget=None,
            )
            session.exec(
                text("UPDATE runs SET run_at = :r WHERE id = :i"),
                params={"r": run_at, "i": run_id},
            )
            insert_pick(
                session,
                run_id,
                rank=1,
                ticker="AAA",
                entry_price=entry,
                scenarios=[
                    {"label": "bear", "probability": 0.2, "target_return_pct": bear},
                    {"label": "bull", "probability": 0.3, "target_return_pct": 50},
                ],
            )
            insert_pick_catalysts(
                session,
                run_id,
                "AAA",
                [{"event": "Earnings", "expected_date": catalyst_date, "direction": "positive"}],
            )

    seed("2025-08-01T10:00:00", 50.0, -10, "2025-09-15")  # outside the window
    seed("2026-07-01T10:00:00", 100.0, -20, "2026-08-03")
    seed("2026-08-01T10:00:00", 110.0, -25, "2026-08-03")  # same event re-named

    [pick] = load_open_picks(db, today=TODAY)
    assert (pick.pick_date, pick.entry_price, pick.bear_target_pct) == (
        date(2026, 8, 1),
        110.0,
        -25,
    )
    assert pick.bull_target_pct == 50
    assert [c["expected_date"] for c in pick.catalysts] == ["2026-08-03"]


def test_section_lists_flagged_picks_and_names_intact_ones():
    frames = {
        "SPY": _path(100, 100, 102),
        "BAD": _path(100, 100, 70),
        "OK": _path(80, 100, 110),
    }
    checks = check_theses(
        [_pick("OK"), _pick("BAD")],
        today=TODAY,
        eps_revisions={"BAD": CUTTING["AAA"]},
        fetch=lambda t, s, e: frames.get(t),
    )
    sections = []
    append_thesis_check_section(sections, thesis_report_data(checks))
    assert sections[0].text == "Open picks: thesis check"
    assert "1 broken, 1 intact" in sections[1].text and "Intact: OK." in sections[1].text
    [row] = sections[2].table_rows
    assert row[:2] == ["BROKEN", "BAD"] and row[5] == "-20.0% / +40.0%"
    assert "Open picks" in render_html_email(sections, {})
    assert render_pdf(sections, {}).startswith(b"%PDF")


def test_section_skipped_without_checks():
    sections = []
    append_thesis_check_section(sections, [])
    assert sections == []


def test_reviewer_payload_carries_the_thesis_check_for_former_picks():
    from stock_analyzer.discover.rebalance_holdings import build_holding_review_payloads

    empty: dict = {}
    payloads = build_holding_review_payloads(
        positions={
            t: {"avg_buy_price": 100.0, "units": 1, "cost_basis": 100.0} for t in ("AAPL", "MSFT")
        },
        fund=empty,
        tech={"AAPL": {"price": 90.0}, "MSFT": {"price": 110.0}},
        rfs=empty,
        insider_selling=empty,
        finnhub_signals=empty,
        eps_revisions=empty,
        position_splits=empty,
        account_meta=empty,
        tax_lots_raw=empty,
        share_trades=empty,
        holdings_quarterly_mda=empty,
        holdings_peers=empty,
        holdings_transcripts=empty,
        news=empty,
        risk_factors_chars=10,
        quarterly_mda_chars=10,
        transcript_chars=10,
        thesis_checks=[{"ticker": "AAPL", "status": "BROKEN", "signals": []}],
    )
    assert payloads["AAPL"]["original_pick_thesis_check"]["status"] == "BROKEN"
    assert payloads["MSFT"]["original_pick_thesis_check"] is None
