"""Query helpers — one repository per aggregate root."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from stock_analyzer.persistence.db import Database
from stock_analyzer.persistence.models import (
    Politician,
    PoliticianScore,
    PoliticianTrade,
    Run,
    SpyDailyClose,
)


@dataclass
class PoliticianRepository:
    db: Database

    def upsert(
        self,
        *,
        full_name: str,
        party: str,
        chamber: str,
        state: str | None,
        capitol_trades_id: str | None,
    ) -> Politician:
        with self.db.session() as s:
            existing = s.scalar(select(Politician).where(Politician.full_name == full_name))
            if existing:
                existing.party = party
                existing.chamber = chamber
                existing.state = state
                existing.capitol_trades_id = capitol_trades_id
                existing.updated_at = datetime.now(UTC)
                s.commit()
                s.refresh(existing)
                return existing
            new = Politician(
                full_name=full_name,
                party=party,
                chamber=chamber,
                state=state,
                capitol_trades_id=capitol_trades_id,
            )
            s.add(new)
            s.commit()
            s.refresh(new)
            return new

    def all(self) -> list[Politician]:
        with self.db.session() as s:
            return list(s.scalars(select(Politician)).all())


@dataclass
class PoliticianTradeRepository:
    db: Database

    def bulk_upsert(self, rows: list[dict[str, Any]]) -> int:
        """Returns count of new rows inserted (existing duplicates skipped)."""
        if not rows:
            return 0
        with self.db.session() as s:
            stmt = (
                sqlite_insert(PoliticianTrade)
                .values(rows)
                .on_conflict_do_nothing(
                    index_elements=[
                        "politician_id",
                        "ticker",
                        "side",
                        "trade_date",
                        "disclosure_date",
                    ]
                )
            )
            result = s.execute(stmt)
            s.commit()
            return result.rowcount or 0

    def for_politician(
        self, politician_id: int, since: date | None = None
    ) -> list[PoliticianTrade]:
        with self.db.session() as s:
            stmt = select(PoliticianTrade).where(PoliticianTrade.politician_id == politician_id)
            if since:
                stmt = stmt.where(PoliticianTrade.trade_date >= since)
            return list(s.scalars(stmt).all())

    def disclosed_between(self, start: date, end: date) -> list[PoliticianTrade]:
        with self.db.session() as s:
            stmt = select(PoliticianTrade).where(
                PoliticianTrade.disclosure_date >= start,
                PoliticianTrade.disclosure_date <= end,
            )
            return list(s.scalars(stmt).all())


@dataclass
class PoliticianScoreRepository:
    db: Database

    def upsert(
        self,
        *,
        politician_id: int,
        computed_at: datetime,
        window_start: date,
        window_end: date,
        total_return_pct: float,
        spy_return_pct: float,
        trade_count: int,
    ) -> PoliticianScore:
        alpha = total_return_pct - spy_return_pct
        with self.db.session() as s:
            row = s.get(PoliticianScore, politician_id)
            if row:
                row.computed_at = computed_at
                row.window_start_date = window_start
                row.window_end_date = window_end
                row.total_return_pct = total_return_pct
                row.spy_return_pct = spy_return_pct
                row.alpha_vs_spy_pct = alpha
                row.trade_count = trade_count
            else:
                row = PoliticianScore(
                    politician_id=politician_id,
                    computed_at=computed_at,
                    window_start_date=window_start,
                    window_end_date=window_end,
                    total_return_pct=total_return_pct,
                    spy_return_pct=spy_return_pct,
                    alpha_vs_spy_pct=alpha,
                    trade_count=trade_count,
                )
                s.add(row)
            s.commit()
            s.refresh(row)
            return row

    def beating_spy(self) -> set[int]:
        with self.db.session() as s:
            stmt = select(PoliticianScore.politician_id).where(PoliticianScore.alpha_vs_spy_pct > 0)
            return set(s.scalars(stmt).all())


@dataclass
class SpyCloseRepository:
    db: Database

    def upsert(self, d: date, close_price: float) -> None:
        with self.db.session() as s:
            stmt = (
                sqlite_insert(SpyDailyClose)
                .values(
                    trade_date=d,
                    close_price=close_price,
                    fetched_at=datetime.now(UTC),
                )
                .on_conflict_do_update(
                    index_elements=["trade_date"],
                    set_={
                        "close_price": close_price,
                        "fetched_at": datetime.now(UTC),
                    },
                )
            )
            s.execute(stmt)
            s.commit()

    def get_range(self, start: date, end: date) -> dict[date, float]:
        with self.db.session() as s:
            stmt = select(SpyDailyClose).where(
                SpyDailyClose.trade_date >= start,
                SpyDailyClose.trade_date <= end,
            )
            return {row.trade_date: row.close_price for row in s.scalars(stmt).all()}


@dataclass
class RunRepository:
    db: Database

    def start_run(self, run_date: date) -> Run:
        with self.db.session() as s:
            existing = s.scalar(select(Run).where(Run.run_date == run_date))
            if existing:
                return existing
            r = Run(
                run_date=run_date,
                started_at=datetime.now(UTC),
                status="running",
            )
            s.add(r)
            s.commit()
            s.refresh(r)
            return r

    def find_by_date(self, run_date: date) -> Run | None:
        with self.db.session() as s:
            return s.scalar(select(Run).where(Run.run_date == run_date))

    def complete_run(
        self,
        run_id: int,
        *,
        status: str,
        email_statuses: tuple[str, str, str],
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        error_log: dict[str, Any] | None,
    ) -> None:
        with self.db.session() as s:
            r = s.get(Run, run_id)
            if r is None:
                return
            r.status = status
            r.completed_at = datetime.now(UTC)
            r.email_1_status, r.email_2_status, r.email_3_status = email_statuses
            r.total_tokens_in = tokens_in
            r.total_tokens_out = tokens_out
            r.est_cost_usd = cost_usd
            r.error_log = error_log
            s.commit()

    def history(self, limit: int = 10) -> list[Run]:
        with self.db.session() as s:
            stmt = select(Run).order_by(Run.run_date.desc()).limit(limit)
            return list(s.scalars(stmt).all())
