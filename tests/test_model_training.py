"""Dataset, walk-forward validation, live scoring and label backfill for
the forward-return model — on synthetic prices, no network."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from dateutil.relativedelta import relativedelta
from sqlalchemy import text

from stock_analyzer.db.repository import insert_candidate, insert_run
from stock_analyzer.db.session import get_session
from stock_analyzer.model.dataset import PricePanel, build_dataset, forward_excess
from stock_analyzer.model.labels import label_candidates
from stock_analyzer.model.ranker_model import (
    ActiveModel,
    load_active_model,
    save_model,
    score_percentiles,
    screen_points,
    walk_forward,
)


def _panel(
    n_tickers: int = 60,
    n_days: int = 1300,
    *,
    signal: float,
    seed: int = 0,
    drift_spread: float = 0.0005,
) -> PricePanel:
    """Random walks where (if signal > 0) the last month's excess return
    keeps going: a planted momentum effect the model should find. Per-name
    drift differences are themselves predictable, so pure noise needs
    drift_spread=0 as well."""
    rng = np.random.default_rng(seed)
    idx = _bdays(date(2020, 1, 1), n_days)
    spy_r = rng.normal(0.0003, 0.01, n_days)
    rets = np.empty((n_days, n_tickers))
    rets[:21] = rng.normal(0, 0.02, (21, n_tickers))
    drift = rng.normal(0.0005, drift_spread, n_tickers) if drift_spread else np.zeros(n_tickers)
    for t in range(21, n_days):
        past = rets[t - 21 : t].sum(axis=0) - spy_r[t - 21 : t].sum()
        rets[t] = spy_r[t] + drift + signal * past / 21 + rng.normal(0, 0.02, n_tickers)
    names = [f"T{i}" for i in range(n_tickers)]
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    volume = rng.integers(500_000, 1_500_000, prices.shape).astype(float)
    return PricePanel(
        _wide(idx, prices, names),
        _wide(idx, prices * 1.005, names),
        _wide(idx, volume, names),
        pl.DataFrame({"date": idx, "SPY": 100 * np.exp(np.cumsum(spy_r))}),
    )


def _bdays(start: date, n: int) -> list[date]:
    """`n` weekdays from `start` (pandas' bdate_range)."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _wide(idx: list[date], values: np.ndarray, names: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"date": idx, **{n: values[:, j] for j, n in enumerate(names)}})


def test_forward_label_enters_on_the_next_bar():
    p = _panel(3, 100, signal=0)
    fwd = forward_excess(p, 5)
    c, s = p.close["T0"], p.spy["SPY"]
    expected = (c[16] / c[11] - 1) - (s[16] / s[11] - 1)
    assert fwd["T0"][10] == pytest.approx(expected)
    assert fwd["T0"][-3] is None  # past the data


def test_walk_forward_finds_a_planted_signal_and_rejects_noise():
    signal = _panel(signal=0.6)
    res = walk_forward(build_dataset(signal), signal.spy["date"], horizon=21, population="all")
    assert res.metrics["model"]["ic_mean"] > 0.05
    assert res.accepted
    # Momentum was planted in the last month's return, so its weight leads.
    assert res.coefficients["rs_1mo"] == max(res.coefficients.values())

    noise = _panel(signal=0.0, seed=1, drift_spread=0)
    res = walk_forward(build_dataset(noise), noise.spy["date"], horizon=21, population="all")
    assert not res.accepted


def test_training_rows_are_purged_before_each_test_block(monkeypatch):
    from stock_analyzer.model import ranker_model as rm

    seen = []
    real_fit = rm.fit_ridge
    monkeypatch.setattr(rm, "fit_ridge", lambda x, y, **kw: seen.append(len(x)) or real_fit(x, y))
    p = _panel(20, 900, signal=0.3)
    data = build_dataset(p)
    rm.walk_forward(data, p.spy["date"], horizon=63, population="all")
    first_block = data["date"].min() + relativedelta(years=2)
    # The purge drops at least 63 trading days (~12 weekly dates) before the block.
    unpurged = data.filter((pl.col("date") < first_block) & pl.col("fwd_63").is_not_nan()).height
    assert seen[0] <= unpurged - 12 * 20


def test_live_scoring_and_persistence(tmp_path):
    db = str(tmp_path / "m.db")
    assert load_active_model(db) is None
    p = _panel(signal=0.6)
    res = walk_forward(build_dataset(p), p.spy["date"], horizon=21, population="all")
    version = save_model(db, res)
    model = load_active_model(db)
    assert model is not None and model.version == version

    feats = {f"T{i}": {"rs_1mo": i / 10, "rs_3mo": 0.0} for i in range(6)}
    pct = score_percentiles(ActiveModel(1, 21, {"rs_1mo": 1.0, "rs_3mo": 0.5}), feats)
    assert pct["T5"] == 100 and pct["T0"] == pytest.approx(100 / 6)
    assert score_percentiles(model, dict(list(feats.items())[:3])) == {}
    assert (screen_points(100), screen_points(50), screen_points(0)) == (5.0, 0.0, -5.0)


def test_label_backfill_writes_matured_outcomes_once(tmp_path):
    db = str(tmp_path / "l.db")
    p = _panel(3, 200, signal=0)
    run_day = p.close["date"][100]
    with get_session(db) as session:
        run_id = insert_run(
            session,
            universe_size=1,
            survivors=1,
            picks=0,
            opus_model="o",
            sonnet_model="s",
            cash_budget=None,
        )
        session.execute(
            text("UPDATE runs SET run_at = :r WHERE id = :i"),
            params={"r": run_day.isoformat(), "i": run_id},
        )
        insert_candidate(
            session,
            run_id,
            "T0",
            passed_filter=True,
            fail_reasons=[],
            score=None,
            score_components=None,
            score_breakdown=None,
            sources=[],
            conviction=0,
            sector=None,
            price=None,
        )
    assert label_candidates(db, fetch_panel=lambda t: p) == 2  # 21d and 63d both closed
    assert label_candidates(db, fetch_panel=lambda t: p) == 0
    with get_session(db) as session:
        row = session.execute(
            text(
                "SELECT entry_date, exit_date, excess_pct FROM candidate_outcomes WHERE horizon_days = 21"
            )
        ).one()
    c, s, days = p.close["T0"], p.spy["SPY"], p.close["date"]
    assert row[0] == str(days[101]) and row[1] == str(days[122])
    expected = ((c[122] / c[101]) - (s[122] / s[101])) * 100
    assert row[2] == pytest.approx(expected)


def test_label_backfill_stops_asking_for_tickers_yahoo_never_priced(tmp_path, monkeypatch):
    from stock_analyzer.data import bar_store
    from stock_analyzer.model import labels

    monkeypatch.setenv("YF_BARS_DIR", str(tmp_path / "bars"))
    db = str(tmp_path / "l.db")
    old, recent = date.today() - timedelta(days=300), date.today() - timedelta(days=40)
    for run_day, tickers in ((old, ["TABLE", "LATE", "REAL"]), (recent, ["NEWJUNK"])):
        with get_session(db) as session:
            run_id = insert_run(
                session,
                universe_size=1,
                survivors=1,
                picks=0,
                opus_model="o",
                sonnet_model="s",
                cash_budget=None,
            )
            session.execute(
                text("UPDATE runs SET run_at = :r WHERE id = :i"),
                params={"r": run_day.isoformat(), "i": run_id},
            )
            for t in tickers:
                insert_candidate(
                    session,
                    run_id,
                    t,
                    passed_filter=True,
                    fail_reasons=[],
                    score=None,
                    score_components=None,
                    score_breakdown=None,
                    sources=[],
                    conviction=0,
                    sector=None,
                    price=None,
                )

    def bars(start: date, n: int) -> pl.DataFrame:
        days = [start + timedelta(days=i) for i in range(n)]
        one = np.full(n, 10.0)
        return pl.DataFrame(
            {"date": days, "Open": one, "High": one, "Low": one, "Close": one, "Volume": one}
        )

    asked_from = old - timedelta(days=700)
    bar_store.save("REAL", bars(old - timedelta(days=30), 330), asked_from)
    bar_store.save("LATE", bars(old + timedelta(days=60), 100), asked_from)  # listed after the run
    asked: list[list[str]] = []

    def fake_download(tickers, years=2):
        asked.append(tickers)
        raise RuntimeError("stop here: only the request matters")

    monkeypatch.setattr(labels, "download_panel", fake_download)
    with pytest.raises(RuntimeError):
        labels.label_candidates(db)
    # TABLE (no prices) and LATE (none on the run date) are dropped; REAL is
    # priced, NEWJUNK's 21-day window closed too recently to give up on.
    assert asked == [["NEWJUNK", "REAL"]]


def _candidates(n: int = 6) -> list[dict]:
    return [
        {
            "ticker": f"T{i}",
            "passed_filter": True,
            "score": 50.0,
            "score_components": {"trend": 20.0},
            "score_breakdown": {},
        }
        for i in range(n)
    ]


@pytest.mark.parametrize("accepted", [False, True])
def test_screen_uses_only_an_accepted_model_and_shadows_the_rest(tmp_path, accepted):
    from types import SimpleNamespace

    from stock_analyzer.cli.discover import DiscoverPipeline
    from stock_analyzer.model.ranker_model import ModelResult

    db = str(tmp_path / "s.db")
    save_model(
        db,
        ModelResult(21, "gated", {"rs_1mo": 1.0}, {}, "2020-01-01", "2024-01-01", accepted),
    )
    pipe = DiscoverPipeline.__new__(DiscoverPipeline)
    pipe.settings = SimpleNamespace(discover_db_path=db)
    cands = _candidates()
    tech = {c["ticker"]: {"model_features": {"rs_1mo": i / 10}} for i, c in enumerate(cands)}
    pipe._apply_model_scores(cands, tech)

    best = cands[-1]
    assert best["score_breakdown"]["model"] == {
        "version": 1,
        "percentile": 100.0,
        "accepted": accepted,
    }
    if accepted:
        assert best["score"] == 55.0 and best["score_components"]["model"] == 5.0
    else:
        assert best["score"] == 50.0 and "model" not in best["score_components"]


def test_candidate_snapshot_keeps_numeric_fields_only(tmp_path):
    import json

    from stock_analyzer.db.repository import insert_candidate_snapshot

    db = str(tmp_path / "c.db")
    with get_session(db) as session:
        run_id = insert_run(
            session,
            universe_size=1,
            survivors=1,
            picks=0,
            opus_model="o",
            sonnet_model="s",
            cash_budget=None,
        )
        insert_candidate_snapshot(
            session,
            run_id,
            "AAA",
            {"forward_pe": 23.456789, "sector": "Tech", "fcf_yield": None, "market_cap": 1.5e12},
            {"net_revisions_30d": 3, "direction_30d": "raising"},
        )
        insert_candidate_snapshot(session, run_id, "BBB", {"sector": "Tech"})  # nothing numeric
    with get_session(db) as session:
        rows = session.execute(text("SELECT ticker, data FROM candidate_snapshots")).all()
    assert [r[0] for r in rows] == ["AAA"]
    assert json.loads(rows[0][1]) == {
        "forward_pe": 23.46,
        "market_cap": 1.5e12,
        "net_revisions_30d": 3,
    }


def test_beta_adjusted_label_removes_market_exposure():
    rng = np.random.default_rng(3)
    idx = _bdays(date(2020, 1, 1), 600)
    spy_r = rng.normal(0.001, 0.01, 600)  # a rising market
    spy = pl.DataFrame({"date": idx, "SPY": 100 * np.cumprod(1 + spy_r)})
    # Day-by-day 2x SPY with no stock-specific return: pure market exposure.
    close = pl.DataFrame({"date": idx, "LEV": 50 * np.cumprod(1 + 2 * spy_r)})
    vol = pl.DataFrame({"date": idx, "LEV": rng.integers(1e6, 2e6, 600).astype(float)})
    data = build_dataset(PricePanel(close, close, vol, spy)).filter(pl.col("fwd_21").is_not_nan())
    assert data["beta_252"].mean() == pytest.approx(2.0, abs=1e-6)
    # Plain excess credits the leverage; the beta-neutral label mostly doesn't
    # (what is left is compounding, a small fraction of the excess).
    assert data["fwd_21"].abs().mean() > 5 * data["fwd_21_badj"].abs().mean()


def test_shadow_scores_are_graded_against_realized_outcomes(tmp_path):
    import json

    from stock_analyzer.db.tables import CandidateOutcome
    from stock_analyzer.model.labels import grade_shadow_scores

    db = str(tmp_path / "g.db")
    with get_session(db) as session:
        for run in range(2):
            run_id = insert_run(
                session,
                universe_size=6,
                survivors=6,
                picks=0,
                opus_model="o",
                sonnet_model="s",
                cash_budget=None,
            )
            for i in range(6):
                insert_candidate(
                    session,
                    run_id,
                    f"T{i}",
                    passed_filter=True,
                    fail_reasons=[],
                    score=50.0,
                    score_components={},
                    score_breakdown={"model": {"version": 1, "percentile": i * 20.0}},
                    sources=[],
                    conviction=0,
                    sector=None,
                    price=None,
                )
                # Run 0: outcomes follow the percentile; run 1: reversed.
                excess = float(i if run == 0 else -i)
                session.add(
                    CandidateOutcome(
                        run_id=run_id,
                        ticker=f"T{i}",
                        horizon_days=21,
                        entry_date="2026-01-02",
                        exit_date="2026-02-02",
                        return_pct=excess,
                        spy_return_pct=0.0,
                        excess_pct=excess,
                    )
                )
    g = grade_shadow_scores(db, horizon=21)
    assert (g["runs"], g["names"]) == (2, 12)
    assert g["mean_ic"] == pytest.approx(0.0) and g["hit_rate"] == 0.5
    assert json.dumps(g)  # plain JSON-able numbers for the email
