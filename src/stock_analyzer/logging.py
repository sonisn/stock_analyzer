"""Centralized logger setup. Use `get_logger(__name__)` everywhere.

Writes to BOTH stderr (live) and a timestamped file under
`~/.stock_analyzer/logs/` (durable). The file path is printed at startup
so the user can recover analysis even if email delivery fails.

Override the log file location with `LOG_FILE=/path/to.log` or the
directory with `LOG_DIR=/path/to/dir`. Override level with `LOG_LEVEL`.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_CONFIGURED = False
_LOG_FILE_PATH: str | None = None


def _configure() -> None:
    global _CONFIGURED, _LOG_FILE_PATH
    if _CONFIGURED:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    stream_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(stream_fmt)

    log_file = os.getenv("LOG_FILE")
    if log_file:
        log_file = os.path.expanduser(log_file)
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = Path(
            os.path.expanduser(os.getenv("LOG_DIR", "~/.stock_analyzer/logs"))
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        log_file = str(log_dir / f"stock-analyzer-{ts}.log")

    file_h = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_h.setFormatter(file_fmt)

    root = logging.getLogger("stock_analyzer")
    root.handlers.clear()
    root.addHandler(stream)
    root.addHandler(file_h)
    root.setLevel(level)
    root.propagate = False

    _LOG_FILE_PATH = log_file
    _CONFIGURED = True
    root.info("Log file: %s", log_file)


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)


def current_log_file() -> str | None:
    """Path the file handler is writing to (None until _configure has run)."""
    _configure()
    return _LOG_FILE_PATH
