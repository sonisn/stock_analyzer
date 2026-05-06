from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from stock_analyzer.persistence.models import (
    Base,
    Politician,
    PoliticianTrade,
    Run,
    SpyDailyClose,
)


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_create_politician_and_trade(session: Session) -> None:
    pol = Politician(full_name="Jane Doe", party="D", chamber="House", state="CA",
                     capitol_trades_id="N00001")
    session.add(pol)
    session.flush()
    trade = PoliticianTrade(
        politician_id=pol.id, ticker="AAPL", side="BUY",
        trade_date=date(2026, 5, 4), disclosure_date=date(2026, 5, 5),
        amount_min_usd=1001, amount_max_usd=15000, raw_payload={"src": "test"},
    )
    session.add(trade)
    session.commit()
    fetched = session.scalar(select(PoliticianTrade))
    assert fetched is not None
    assert fetched.ticker == "AAPL"


def test_unique_disclosure_constraint(session: Session) -> None:
    pol = Politician(full_name="A", party="R", chamber="Senate", state="TX",
                     capitol_trades_id="N00002")
    session.add(pol)
    session.flush()
    common = dict(politician_id=pol.id, ticker="MSFT", side="BUY",
                  trade_date=date(2026, 5, 4), disclosure_date=date(2026, 5, 5),
                  amount_min_usd=1, amount_max_usd=2, raw_payload={})
    session.add(PoliticianTrade(**common))
    session.commit()
    session.add(PoliticianTrade(**common))
    with pytest.raises(IntegrityError):
        session.commit()


def test_run_unique_per_date(session: Session) -> None:
    session.add(Run(run_date=date(2026, 5, 5),
                    started_at=datetime.now(UTC),
                    status="running"))
    session.commit()
    session.add(Run(run_date=date(2026, 5, 5),
                    started_at=datetime.now(UTC),
                    status="running"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_spy_close_round_trip(session: Session) -> None:
    session.add(SpyDailyClose(trade_date=date(2026, 5, 4), close_price=528.41,
                              fetched_at=datetime.now(UTC)))
    session.commit()
    row = session.scalar(select(SpyDailyClose))
    assert row is not None
    assert row.close_price == 528.41
