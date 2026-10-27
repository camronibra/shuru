"""JSON-lines logging to stdout, with the current run id on every record."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_current_run_id: ContextVar[int | None] = ContextVar("current_run_id", default=None)


class JsonFormatter(logging.Formatter):
    """One JSON object per record: ts (UTC), level, job, run_id, logger, msg, extras."""

    def __init__(self, job: str, max_field_chars: int) -> None:
        raise NotImplementedError

    def format(self, record: logging.LogRecord) -> str:
        """Must never raise. Unserialisable values fall back to a truncated repr."""
        raise NotImplementedError

    def _truncate(self, value: str) -> str:
        raise NotImplementedError


class RunContextFilter(logging.Filter):
    """Adds run_id to each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        raise NotImplementedError


def configure_logging(job: str, level: str, max_field_chars: int) -> None:
    """Replace the root logger's handlers with the JSON handler. Safe to call twice."""
    raise NotImplementedError


@contextmanager
def bind_run_id(run_id: int) -> Iterator[None]:
    """Set the run id for the block, then restore the previous one."""
    raise NotImplementedError


def get_logger(name: str) -> logging.Logger:
    raise NotImplementedError
