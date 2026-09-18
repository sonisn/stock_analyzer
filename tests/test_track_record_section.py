"""Structured 'Track record' report section (replaces the monospace dump)."""

from __future__ import annotations

from stock_analyzer.discover.report_html import render_html_email
from stock_analyzer.discover.report_pdf import render_pdf
from stock_analyzer.discover.report_sections import append_track_record_section
from stock_analyzer.models.track_record import (
    DirectionStats,
    HorizonStats,
    PickReturn,
    ProviderStats,
    TrackRecord,
    UnmeasurableDecision,
)


def _stats(n: int, alpha: float | None, beta_adj: float | None = None) -> DirectionStats:
    return DirectionStats(
        n_mature=n,
        n_pending=0,
        mean_return_pct=None,
        mean_spy_return_pct=None,
        mean_alpha_pct=alpha,
        mean_beta_adjusted_alpha_pct=beta_adj,
        n_beta_adjusted=n if beta_adj is not None else 0,
        winners=n // 2,
        losers=n - n // 2,
        flats=0,
        sharpe=0.4 if n >= 5 else None,
    )


def _pick(ticker: str, alpha: float, direction: str = "buy", horizon: int = 90) -> PickReturn:
    return PickReturn(
        ticker=ticker,
        pick_date="2026-05-01",
        age_days=140,
        direction=direction,
        horizon_days=horizon,
        pick_price=10.0,
        measured_price=11.0,
        pick_return_pct=alpha + 3.0,
        spy_return_pct=3.0,
        alpha_pct=alpha,
        beta=1.2,
        beta_adjusted_alpha_pct=alpha - 0.6,
        is_mature=True,
    )


def _record(picks: list[PickReturn], **overrides) -> TrackRecord:
    empty = _stats(0, None)
    buy, hold = _stats(6, 2.5, 1.1), _stats(2, -1.0)
    horizon = HorizonStats(
        horizon_days=90,
        overall=_stats(8, 1.6),
        buy_stats=buy,
        hold_stats=hold,
        trim_stats=empty,
        sell_stats=empty,
        model_breakdown=[],
        provider_breakdown=[
            ProviderStats(provider="gemini", n_mature=4, mean_alpha_pct=3.0, sharpe=None)
        ],
        decisions=picks,
    )
    fields = dict(
        n_picks_total=len(picks) + 2,
        n_mature=8,
        n_pending=1,
        reported_horizon_days=90,
        horizons=[horizon],
        n_unmeasurable=1,
        unmeasurable=[
            UnmeasurableDecision(
                ticker="GONE",
                pick_date="2026-04-01",
                direction="buy",
                age_days=170,
                reason="no_price_data",
            )
        ],
        mean_return_pct=4.6,
        mean_spy_return_pct=3.0,
        mean_alpha_pct=1.6,
        winners=4,
        losers=4,
        flats=0,
        overall_sharpe=0.3,
        buy_stats=buy,
        hold_stats=hold,
        trim_stats=empty,
        sell_stats=empty,
        model_breakdown=[],
        picks=picks,
        pending=[
            PickReturn(
                ticker="NEW",
                pick_date="2026-09-01",
                age_days=17,
                pick_price=5.0,
                measured_price=5.5,
                pick_return_pct=10.0,
                spy_return_pct=None,
                alpha_pct=None,
                is_mature=False,
            )
        ],
    )
    fields.update(overrides)
    return TrackRecord(**fields)


def test_structured_record_replaces_preformatted_dump():
    record = _record([_pick("AAA", 5.0), _pick("BBB", -2.0, "hold")])
    sections = []
    append_track_record_section(sections, record, "RAW BLOCK")

    kinds = [s.kind for s in sections]
    assert "preformatted" not in kinds
    assert kinds[0] == "heading" and sections[0].text == "Track record"
    bar = next(s for s in sections if s.kind == "bar_chart")
    assert [b["label"] for b in bar.data["bars"]] == ["Buy · 6 scored", "Hold · 2 scored"]
    assert bar.data["bars"][0]["note"] == "beta-adj +1.1%"
    tables = [s for s in sections if s.kind == "table"]
    assert tables[0].table_rows[0][:4] == ["90d", "Buy", "6", "+2.5%"]
    assert any(r[1] == "Provider: gemini" for t in tables for r in t.table_rows)
    assert any("GONE" in s.text for s in sections if s.kind == "para")
    assert any(r[1] == "NEW" for t in tables for r in t.table_rows)


def test_decisions_table_shows_best_and_worst_when_long():
    picks = [_pick(f"T{i:02d}", float(i)) for i in range(20)]
    sections = []
    append_track_record_section(sections, _record(picks), "")

    heading = next(s for s in sections if s.kind == "heading" and s.text.startswith("Scored calls"))
    assert "6 best and 6 worst of 20" in heading.text
    table = sections[sections.index(heading) + 1]
    tickers = [r[1] for r in table.table_rows]
    assert tickers[:2] == ["T19", "T18"] and tickers[-1] == "T00"
    assert len(tickers) == 12


def test_falls_back_to_block_without_record_and_skips_empty_record():
    sections = []
    append_track_record_section(sections, None, "RAW BLOCK")
    assert [s.kind for s in sections] == ["heading", "preformatted"]

    sections = []
    append_track_record_section(sections, _record([], n_picks_total=0), "RAW BLOCK")
    assert sections == []


def test_no_mature_decisions_renders_summary_only():
    record = _record(
        [], n_mature=0, horizons=[], buy_stats=_stats(0, None), hold_stats=_stats(0, None)
    )
    sections = []
    append_track_record_section(sections, record, "")
    assert "0 finished decisions" in sections[1].text
    assert not any(s.kind == "bar_chart" for s in sections)


def test_section_renders_in_html_and_pdf():
    sections = []
    append_track_record_section(sections, _record([_pick("AAA", 5.0), _pick("BBB", -2.0)]), "")
    html = render_html_email(sections, {})
    assert "Buy · 6 scored" in html and "<svg" in html
    assert render_pdf(sections, {}).startswith(b"%PDF")
