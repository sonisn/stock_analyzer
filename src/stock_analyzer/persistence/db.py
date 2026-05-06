"""Database connection management with explicit ephemeral-mode guard."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from stock_analyzer.persistence.models import Base


class EphemeralModeError(RuntimeError):
    """Raised when DB-only code paths are invoked in ephemeral mode."""


class Database:
    """Wraps a SQLAlchemy engine and session factory with an ephemeral-mode guard.

    In ephemeral mode no engine is created and ``session()`` raises
    :class:`EphemeralModeError`; repositories must use in-memory equivalents.
    """

    url: str
    ephemeral: bool

    def __init__(self, url: str, ephemeral: bool) -> None:
        self.url = url
        self.ephemeral = ephemeral
        self._engine: Engine | None = None
        self._sessionmaker: sessionmaker[Session] | None = None
        if not ephemeral:
            self._engine = create_engine(url, future=True)
            self._sessionmaker = sessionmaker(self._engine, expire_on_commit=False)

    def create_all(self) -> None:
        if self.ephemeral:
            return
        assert self._engine is not None  # narrowed by the ephemeral check
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        if self.ephemeral:
            raise EphemeralModeError(
                "Database.session() called in ephemeral mode — repositories "
                "must use the in-memory equivalents."
            )
        assert self._sessionmaker is not None  # narrowed by the ephemeral check
        with self._sessionmaker() as s:
            yield s
