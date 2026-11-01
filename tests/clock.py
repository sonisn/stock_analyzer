"""A `date` whose `today()` is pinned, for tests whose fixtures carry
fixed dates. Patch it over the module under test's `date`:

    monkeypatch.setattr(module, "date", frozen_date(date(2026, 9, 30)))
"""

from __future__ import annotations

from datetime import date
from typing import Self


def frozen_date(today: date) -> type[date]:
    class _FrozenDate(date):
        @classmethod
        def today(cls) -> Self:
            return cls(today.year, today.month, today.day)

    return _FrozenDate
