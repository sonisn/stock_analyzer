"""SQLAlchemy 2.x typed models matching the spec schema."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Politician(Base):
    __tablename__ = "politicians"

    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    party: Mapped[str] = mapped_column(String, nullable=False)
    chamber: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str | None] = mapped_column(String)
    capitol_trades_id: Mapped[str | None] = mapped_column(String, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    trades: Mapped[list[PoliticianTrade]] = relationship(back_populates="politician")
    score: Mapped[PoliticianScore | None] = relationship(back_populates="politician", uselist=False)

    __table_args__ = (
        CheckConstraint("party IN ('D','R','I')", name="party_check"),
        CheckConstraint("chamber IN ('House','Senate')", name="chamber_check"),
    )


class PoliticianTrade(Base):
    __tablename__ = "politician_trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    politician_id: Mapped[int] = mapped_column(ForeignKey("politicians.id"), nullable=False)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    side: Mapped[str] = mapped_column(String, nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    disclosure_date: Mapped[date] = mapped_column(Date, nullable=False)
    amount_min_usd: Mapped[int | None] = mapped_column(Integer)
    amount_max_usd: Mapped[int | None] = mapped_column(Integer)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    politician: Mapped[Politician] = relationship(back_populates="trades")

    __table_args__ = (
        CheckConstraint("side IN ('BUY','SELL')", name="side_check"),
        UniqueConstraint(
            "politician_id", "ticker", "side", "trade_date", "disclosure_date",
            name="uq_disclosure",
        ),
        Index("ix_trades_disclosure_date", "disclosure_date"),
        Index("ix_trades_politician_id", "politician_id"),
    )


class PoliticianScore(Base):
    __tablename__ = "politician_scores"

    politician_id: Mapped[int] = mapped_column(
        ForeignKey("politicians.id"), primary_key=True
    )
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_start_date: Mapped[date] = mapped_column(Date, nullable=False)
    window_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    total_return_pct: Mapped[float] = mapped_column(nullable=False)
    spy_return_pct: Mapped[float] = mapped_column(nullable=False)
    alpha_vs_spy_pct: Mapped[float] = mapped_column(nullable=False)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False)

    politician: Mapped[Politician] = relationship(back_populates="score")


class SpyDailyClose(Base):
    __tablename__ = "spy_daily_close"

    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    close_price: Mapped[float] = mapped_column(nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String, nullable=False)
    email_1_status: Mapped[str | None] = mapped_column(String)
    email_2_status: Mapped[str | None] = mapped_column(String)
    email_3_status: Mapped[str | None] = mapped_column(String)
    error_log: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    total_tokens_in: Mapped[int | None] = mapped_column(Integer)
    total_tokens_out: Mapped[int | None] = mapped_column(Integer)
    est_cost_usd: Mapped[float | None] = mapped_column()

    __table_args__ = (
        CheckConstraint(
            "status IN ('running','success','partial','failed')", name="status_check"
        ),
    )
