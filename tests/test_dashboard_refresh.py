"""The page and the email must not disagree about what was decided.

The dashboard is a generated file, so it only changes when something
regenerates it. Cron does that after the close — which leaves a run
started by hand showing yesterday's plan on the page while today's sits
in the inbox.
"""

from __future__ import annotations

import inspect

from stock_analyzer.cli.discover import DiscoverPipeline
from stock_analyzer.cli.rebalance import RebalancePipeline
from stock_analyzer.config import Settings


class _Pipe(DiscoverPipeline):
    def __init__(self, settings):  # noqa: D107 - bypass the real pipeline setup
        self.settings = settings
        self.state = {}


def test_both_pipelines_end_by_refreshing_the_page():
    for module in (
        inspect.getsource(RebalancePipeline),
        inspect.getsource(DiscoverPipeline),
    ):
        assert 'Step(name="dashboard"' in module, "a run must leave the page current"


def test_the_refresh_never_fails_the_run(monkeypatch):
    """The plan is emailed and persisted before this runs. A page that did
    not rebuild is cosmetic; taking the run down over it is not."""
    import stock_analyzer.cli.discover as disc

    def boom(*a, **k):
        raise RuntimeError("brokerage is down")

    monkeypatch.setattr(disc, "logger", disc.logger)
    monkeypatch.setattr("stock_analyzer.cli.dashboard.collect", boom)
    out = _Pipe(Settings()).step_refresh_dashboard(None)
    assert "failed" in out.content
    assert "RuntimeError" in out.content


def test_it_can_be_turned_off():
    out = _Pipe(Settings(dashboard_after_run=False)).step_refresh_dashboard(None)
    assert "disabled" in out.content


def test_it_writes_where_the_cli_writes(tmp_path, monkeypatch):
    """One setting owns the path, so the scheduled build and the run-end
    refresh cannot drift to different files."""
    target = tmp_path / "nested" / "dash.html"
    settings = Settings(dashboard_path=str(target))
    monkeypatch.setattr(
        "stock_analyzer.cli.dashboard.collect",
        lambda s, today: {
            "generated": "2026-09-21",
            "latest_run": 36,
            "holdings": [],
            "history": {},
            "views": {},
            "runs": [],
            "suggestions": [],
            "holdings_ok": True,
            "record": {"rows": 0, "tickers": 0, "first": None, "last": None},
        },
    )
    out = _Pipe(settings).step_refresh_dashboard(None)
    assert target.exists(), out.content
    assert "<html" in target.read_text()
