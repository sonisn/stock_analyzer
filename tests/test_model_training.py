"""Dataset, walk-forward validation, live scoring and label backfill for
the forward-return model — on synthetic prices, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
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
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    spy_r = rng.normal(0.0003, 0.01, n_days)
    rets = np.empty((n_days, n_tickers))
    rets[:21] = rng.normal(0, 0.02, (21, n_tickers))
    drift = rng.normal(0.0005, drift_spread, n_tickers) if drift_spread else np.zeros(n_tickers)
    for t in range(21, n_days):
        past = rets[t - 21 : t].sum(axis=0) - spy_r[t - 21 : t].sum()
        rets[t] = spy_r[t] + drift + signal * past / 21 + rng.normal(0, 0.02, n_tickers)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rets, axis=0)),
        index=idx,
        columns=[f"T{i}" for i in range(n_tickers)],
    )
    spy = pd.Series(100 * np.exp(np.cumsum(spy_r)), index=idx)
    volume = pd.DataFrame(
        rng.integers(500_000, 1_500_000, close.shape), index=idx, columns=close.columns
    ).astype(float)
    return PricePanel(close, close * 1.005, volume, spy)


def test_forward_label_enters_on_the_next_bar():
    p = _panel(3, 100, signal=0)
    fwd = forward_excess(p, 5)
    day = p.close.index[10]
    c, s = p.close["T0"], p.spy
    expected = (c.iloc[16] / c.iloc[11] - 1) - (s.iloc[16] / s.iloc[11] - 1)
    assert fwd.loc[day, "T0"] == pytest.approx(expected)
    assert np.isnan(fwd["T0"].iloc[-3])


def test_walk_forward_finds_a_planted_signal_and_rejects_noise():
    signal = _panel(signal=0.6)
    res = walk_forward(build_dataset(signal), signal.spy.index, horizon=21, population="all")
    assert res.metrics["model"]["ic_mean"] > 0.05
    assert res.accepted
    # Momentum was planted in the last month's return, so its weight leads.
    assert res.coefficients["rs_1mo"] == max(res.coefficients.values())

    noise = _panel(signal=0.0, seed=1, drift_spread=0)
    res = walk_forward(build_dataset(noise), noise.spy.index, horizon=21, population="all")
    assert not res.accepted


def test_training_rows_are_purged_before_each_test_block(monkeypatch):
    from stock_analyzer.model import ranker_model as rm

    seen = []
    real_fit = rm.fit_ridge
    monkeypatch.setattr(rm, "fit_ridge", lambda x, y, **kw: seen.append(len(x)) or real_fit(x, y))
    p = _panel(20, 900, signal=0.3)
    data = build_dataset(p)
    rm.walk_forward(data, p.spy.index, horizon=63, population="all")
    labeled = data["fwd_63"].notna().groupby(level="date").sum()
    first_block = data.index.get_level_values("date").min() + pd.DateOffset(years=2)
    # The purge drops at least 63 trading days (~12 weekly dates) before the block.
    unpurged = int(labeled[labeled.index < first_block].sum())
    assert seen[0] <= unpurged - 12 * 20


def test_live_scoring_and_persistence(tmp_path):
    db = str(tmp_path / "m.db")
    assert load_active_model(db) is None
    p = _panel(signal=0.6)
    res = walk_forward(build_dataset(p), p.spy.index, horizon=21, population="all")
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
    run_day = p.close.index[100]
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
        session.exec(
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
        row = session.exec(
            text(
                "SELECT entry_date, exit_date, excess_pct FROM candidate_outcomes WHERE horizon_days = 21"
            )
        ).one()
    c, s = p.close["T0"], p.spy
    assert row[0] == str(p.close.index[101].date()) and row[1] == str(p.close.index[122].date())
    expected = ((c.iloc[122] / c.iloc[101]) - (s.iloc[122] / s.iloc[101])) * 100
    assert row[2] == pytest.approx(expected)


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
        rows = session.exec(text("SELECT ticker, data FROM candidate_snapshots")).all()
    assert [r[0] for r in rows] == ["AAA"]
    assert json.loads(rows[0][1]) == {
        "forward_pe": 23.46,
        "market_cap": 1.5e12,
        "net_revisions_30d": 3,
    }
