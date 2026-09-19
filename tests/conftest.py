"""Suite-wide isolation: no `.env`, no network.

Two separate leaks made the suite untrustworthy before this file existed.

1. `Settings` declares `env_file=".env"`, and pydantic-settings reads that
   file directly rather than going through `os.environ`. So
   `monkeypatch.delenv("TRADIER_API_KEY")` could not make a key look
   unset — `Settings()` still found it on disk. The test that asserted
   "degrade to None when the key is missing" therefore ran *with* the
   developer's real key and made a live API call.

2. Nothing stopped that call. A test that accidentally reaches the
   network is slow, flaky, order-dependent, and spends real credentials
   and real rate limit on every `pytest` run.

`_isolate_settings_env` closes the first by pointing `Settings` at no env
file for the whole suite, so a test sees exactly the vars it sets itself.
`_block_network` closes the second at the socket layer, which catches
httpx, requests, yfinance and every SDK at once. A test that genuinely
needs the network must say so:

    @pytest.mark.allow_network
    def test_hits_a_real_api(): ...
"""

from __future__ import annotations

import os
import socket
import tempfile
from collections.abc import Iterator

# Logging picks its directory when the package is first imported, which
# happens during collection — set it before that so test runs don't fill
# the real ~/.stock_analyzer/logs with hundreds of tiny log files.
os.environ["LOG_DIR"] = tempfile.mkdtemp(prefix="stock-analyzer-test-logs-")

import pytest

_REAL_SOCKET_CONNECT = socket.socket.connect
_REAL_CREATE_CONNECTION = socket.create_connection


class NetworkCallInTestError(RuntimeError):
    """A test tried to open a socket. Mock the provider, or mark the test."""


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `Settings()` read only what the test sets, never the real `.env`.

    `model_config` is a plain dict at runtime, so `setitem` is enough and
    monkeypatch restores it after each test.
    """
    from stock_analyzer.config import Settings

    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture(autouse=True)
def _isolate_db_path(monkeypatch: pytest.MonkeyPatch, tmp_path_factory) -> None:
    """With no `.env`, `Settings()` falls back to the default DB path in the
    user's home directory — building a workflow there left an empty
    `~/.stock_analyzer/discover.db` behind on every run. Point it at a
    throwaway file instead (tests that need a DB still pass their own)."""
    db = tmp_path_factory.mktemp("db") / "stock.db"
    monkeypatch.setenv("DISCOVER_DB_PATH", str(db))


@pytest.fixture(autouse=True)
def _reset_db_engines() -> Iterator[None]:
    """Engines (and their created schema) are cached per database file, so
    a test that recreates a path would otherwise inherit the last one."""
    from stock_analyzer.db.session import reset_engines

    reset_engines()
    yield
    reset_engines()


@pytest.fixture(autouse=True)
def _reset_yf_gateway() -> Iterator[None]:
    """Clear the yfinance gateway's caches between tests.

    It memoizes `yf.Ticker` instances and remembers symbols Yahoo had no
    data for, both of which would otherwise leak a previous test's mock
    (or its "unavailable" verdict) into the next one.
    """
    from stock_analyzer.data import yf_gateway

    yf_gateway.reset()
    yield
    yf_gateway.reset()


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly on any outbound socket unless the test opts in.

    Loopback is left open so a test can still talk to a local fixture
    server or an sqlite-over-TCP style helper if one is ever added.
    """
    if request.node.get_closest_marker("allow_network"):
        return

    def _guard_connect(self: socket.socket, address, *args, **kwargs):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "::1", "localhost"):
            return _REAL_SOCKET_CONNECT(self, address, *args, **kwargs)
        raise NetworkCallInTestError(
            f"Test attempted a network connection to {host!r}. Patch the "
            f"provider/transport instead, or mark the test with "
            f"@pytest.mark.allow_network if it must hit a real endpoint."
        )

    def _guard_create_connection(address, *args, **kwargs):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "::1", "localhost"):
            return _REAL_CREATE_CONNECTION(address, *args, **kwargs)
        raise NetworkCallInTestError(
            f"Test attempted a network connection to {host!r}. Patch the "
            f"provider/transport instead, or mark the test with "
            f"@pytest.mark.allow_network if it must hit a real endpoint."
        )

    monkeypatch.setattr(socket.socket, "connect", _guard_connect)
    monkeypatch.setattr(socket, "create_connection", _guard_create_connection)
