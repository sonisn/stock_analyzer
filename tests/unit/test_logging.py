from __future__ import annotations

import json
import logging

import pytest

from stock_analyzer.logging import configure_logging, get_logger


def test_configure_logging_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="json")
    log = get_logger("test")
    log.info("hello", foo="bar")
    captured = capsys.readouterr()
    out = captured.err + captured.out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["event"] == "hello"
    assert payload["foo"] == "bar"
    assert payload["level"] == "info"


def test_configure_logging_pretty(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="pretty")
    log = get_logger("test")
    log.info("readable", foo="bar")
    captured = capsys.readouterr()
    out = captured.err + captured.out
    assert "readable" in out


def test_logger_respects_level() -> None:
    configure_logging(level="WARNING", fmt="json")
    assert logging.getLogger().getEffectiveLevel() == logging.WARNING


def test_logger_filters_below_level(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="WARNING", fmt="json")
    log = get_logger("test")
    log.info("should-not-appear")
    captured = capsys.readouterr()
    assert "should-not-appear" not in (captured.err + captured.out)


def test_logger_emits_at_or_above_level(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="WARNING", fmt="json")
    log = get_logger("test")
    log.warning("should-appear")
    captured = capsys.readouterr()
    assert "should-appear" in (captured.err + captured.out)


def test_invalid_log_level_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging(level="VERBOSE", fmt="json")
