"""SQLAlchemy engine + Session contextmanager for the SQLite analytics DB.

A connect-time event listener turns on PRAGMA foreign_keys=ON for every
sqlite connection (SQLAlchemy disables it by default; the legacy
raw-sqlite code enabled it explicitly).

_apply_legacy_migrations runs the same idempotent ALTER TABLEs that
discover/persistence.py used, so old local DBs created before the kind /
rebalance_text / dashboard_data columns existed still migrate forward.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, SQLModel, create_engine

# Import tables module so SQLModel.metadata is populated before create_all().
from . import tables as _tables  # noqa: F401


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_conn, _connection_record):
    """Mirror the PRAGMA foreign_keys=ON the legacy raw-sqlite code set."""
    try:
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()
    except Exception:
        # If the DBAPI doesn't support PRAGMA (i.e. not sqlite), ignore.
        pass


# Verbatim copy of _MIGRATIONS from discover/persistence.py.
# Order: create_all() is a no-op on existing tables, then these ALTERs
# forward-migrate older local DBs.
_LEGACY_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("runs", "ALTER TABLE runs ADD COLUMN kind TEXT NOT NULL DEFAULT 'discover'"),
    ("run_outputs", "ALTER TABLE run_outputs ADD COLUMN rebalance_text TEXT"),
    ("run_outputs", "ALTER TABLE run_outputs ADD COLUMN dashboard_data TEXT"),
)


def _apply_legacy_migrations(engine: Engine) -> None:
    """Idempotent ALTERs. Swallow 'duplicate column' (already migrated)."""
    with engine.begin() as conn:
        for _table, ddl in _LEGACY_MIGRATIONS:
            try:
                conn.exec_driver_sql(ddl)
            except OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise


def _expanded_path(path: str) -> Path:
    return Path(os.path.expanduser(path))


def _build_engine(db_path: str) -> Engine:
    p = _expanded_path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{p}", echo=False)


@contextmanager
def get_session(db_path: str) -> Iterator[Session]:
    """Open a Session against the SQLite analytics DB.

    create_all() runs first (no-op on existing tables), then the legacy
    ALTER migrations run, then the caller's block executes inside a
    Session that auto-commits on success and rolls back on exception.
    """
    engine = _build_engine(db_path)
    SQLModel.metadata.create_all(engine)
    _apply_legacy_migrations(engine)
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


__all__ = ["get_session"]
