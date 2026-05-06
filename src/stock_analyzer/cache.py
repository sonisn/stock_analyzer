"""File-based result cache used by both pipelines."""
from __future__ import annotations

from pathlib import Path

from .logging import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"


class FileCache:
    """Simple read-through file cache.

    Returns cached content when present and `enabled=True`. The caller
    decides whether to honor it (e.g., based on a settings flag).
    """

    def __init__(self, filename: str):
        self.path = CACHE_DIR / filename

    def read(self, *, enabled: bool = True) -> str | None:
        if not enabled:
            return None
        if self.path.exists():
            logger.info("Cache hit: %s", self.path)
            return self.path.read_text()
        logger.debug("Cache miss: %s", self.path)
        return None

    def write(self, content: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(content)
        logger.info("Cached result to %s", self.path)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
            logger.info("Cleared cache: %s", self.path)
