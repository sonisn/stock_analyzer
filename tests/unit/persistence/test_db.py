from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from stock_analyzer.persistence.db import (
    Database,
    EphemeralModeError,
)


def test_production_session_works(tmp_path: Path) -> None:
    db = Database(url=f"sqlite:///{tmp_path / 'x.db'}", ephemeral=False)
    db.create_all()
    with db.session() as s:
        s.execute(text("SELECT 1"))


def test_ephemeral_session_raises() -> None:
    db = Database(url="sqlite:///:memory:", ephemeral=True)
    with pytest.raises(EphemeralModeError), db.session():
        pass


def test_ephemeral_create_all_is_noop() -> None:
    db = Database(url="sqlite:///:memory:", ephemeral=True)
    db.create_all()  # should not raise
