from __future__ import annotations

from datetime import date

import pytest

from stock_analyzer.persistence.db import Database
from stock_analyzer.persistence.repositories import (
    PoliticianRepository,
    PoliticianTradeRepository,
    RunRepository,
    SpyCloseRepository,
)


@pytest.fixture
def db(tmp_path) -> Database:
    d = Database(url=f"sqlite:///{tmp_path / 'r.db'}", ephemeral=False)
    d.create_all()
    return d


def test_upsert_politician(db: Database) -> None:
    repo = PoliticianRepository(db)
    p1 = repo.upsert(
        full_name="A B", party="D", chamber="House", state="CA", capitol_trades_id="X1"
    )
    assert p1.id is not None
    p2 = repo.upsert(
        full_name="A B", party="R", chamber="Senate", state="NY", capitol_trades_id="X1"
    )
    assert p2.id == p1.id
    assert p2.party == "R"  # updated


def test_idempotent_trade_insert(db: Database) -> None:
    pol_repo = PoliticianRepository(db)
    pol = pol_repo.upsert(
        full_name="C D", party="I", chamber="Senate", state="VT", capitol_trades_id="X2"
    )
    trade_repo = PoliticianTradeRepository(db)
    payload = dict(
        politician_id=pol.id,
        ticker="AAPL",
        side="BUY",
        trade_date=date(2026, 5, 4),
        disclosure_date=date(2026, 5, 5),
        amount_min_usd=1001,
        amount_max_usd=15000,
        raw_payload={"k": "v"},
    )
    n1 = trade_repo.bulk_upsert([payload])
    n2 = trade_repo.bulk_upsert([payload])  # same input
    assert n1 == 1
    assert n2 == 0  # already present


def test_spy_close_upsert(db: Database) -> None:
    repo = SpyCloseRepository(db)
    repo.upsert(date(2026, 5, 4), 528.41)
    repo.upsert(date(2026, 5, 4), 530.00)  # overwrite
    series = repo.get_range(date(2026, 5, 1), date(2026, 5, 5))
    assert series[date(2026, 5, 4)] == 530.00


def test_run_idempotency(db: Database) -> None:
    repo = RunRepository(db)
    r = repo.start_run(date(2026, 5, 5))
    assert r.id is not None
    assert repo.find_by_date(date(2026, 5, 5)).id == r.id
    repo.complete_run(
        r.id,
        status="success",
        email_statuses=("sent", "sent", "sent"),
        tokens_in=100,
        tokens_out=200,
        cost_usd=0.05,
        error_log=None,
    )
    final = repo.find_by_date(date(2026, 5, 5))
    assert final.status == "success"
    assert final.completed_at is not None
