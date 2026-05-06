"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip ambient stock-analyzer env vars so tests are reproducible."""
    for key in list(os.environ):
        if key.startswith(("ANTHROPIC_", "SNAPTRADE_", "FINNHUB_", "SMTP_", "STOCK_ANALYZER_")):
            monkeypatch.delenv(key, raising=False)
    yield
