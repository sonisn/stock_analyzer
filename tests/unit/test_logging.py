from __future__ import annotations

import json
import logging

import pytest

from stock_analyzer.logging import configure_logging, get_logger


def test_configure_logging_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="json")
    log = get_logger("test")
    log.info("hello", foo="bar")
    captured = capsys.readouterr().err or capsys.readouterr().out
    payload = json.loads(captured.strip().splitlines()[-1])
    assert payload["event"] == "hello"
    assert payload["foo"] == "bar"
    assert payload["level"] == "info"


def test_configure_logging_pretty(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="pretty")
    log = get_logger("test")
    log.info("readable", foo="bar")
    out = (capsys.readouterr().err or "") + (capsys.readouterr().out or "")
    assert "readable" in out


def test_logger_respects_level() -> None:
    configure_logging(level="WARNING", fmt="json")
    assert logging.getLogger().getEffectiveLevel() == logging.WARNING
