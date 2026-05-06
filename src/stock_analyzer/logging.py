"""Structured logging — JSON in production, pretty in dev."""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal, cast

import structlog
from structlog.types import FilteringBoundLogger


def configure_logging(level: str = "INFO", fmt: Literal["json", "pretty"] = "json") -> None:
    """Idempotent logger configuration. Call once at startup before spawning tasks.

    Not thread-safe — concurrent calls from multiple coroutines/threads may race.
    """
    try:
        log_level = logging.getLevelNamesMapping()[level.upper()]
    except KeyError as e:
        raise ValueError(f"Unknown log level: {level!r}") from e
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
        force=True,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    if fmt == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a structlog logger. Call after configure_logging."""
    return cast(FilteringBoundLogger, structlog.get_logger(name))
