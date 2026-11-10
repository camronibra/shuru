"""Logging to stderr, as JSON lines or plain text, with the current run id on every record.

stderr keeps stdout free for command output (reports, exports). Log records can carry
scraped text, so every string field is length-capped and control characters are escaped:
a hostile page cannot forge extra log lines or flood the journal.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

_current_run_id: ContextVar[int | None] = ContextVar("current_run_id", default=None)

# Attributes every LogRecord has. Anything else was passed with ``extra=`` and is logged.
_STANDARD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime", "run_id"}

# Control characters -> visible escapes, for the text format.
_ESCAPE_CONTROL = {code: f"\\x{code:02x}" for code in (*range(0x20), 0x7F)}
_ESCAPE_CONTROL[0x2028] = "\\u2028"
_ESCAPE_CONTROL[0x2029] = "\\u2029"


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {key: value for key, value in vars(record).items() if key not in _STANDARD_ATTRS}


class JsonFormatter(logging.Formatter):
    """One JSON object per record: ts (UTC), level, job, run_id, logger, msg, extras."""

    def __init__(self, job: str, max_field_chars: int) -> None:
        super().__init__()
        self._job = job
        self._max = max_field_chars

    def format(self, record: logging.LogRecord) -> str:
        """Must never raise. Unserialisable values fall back to a truncated repr."""
        try:
            entry: dict[str, Any] = {
                "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="seconds"),
                "level": record.levelname,
                "job": self._job,
                "run_id": getattr(record, "run_id", None),
                "logger": record.name,
                "msg": self._truncate(record.getMessage()),
            }
            for key, value in _extras(record).items():
                entry[key] = self._truncate(value) if isinstance(value, str) else value
            if record.exc_info:
                entry["exc"] = self._truncate(self.formatException(record.exc_info))
            return json.dumps(entry, ensure_ascii=False, default=self._fallback)
        except Exception:
            return json.dumps({"level": "ERROR", "job": self._job, "msg": "unloggable record"})

    def _fallback(self, value: object) -> str:
        return self._truncate(repr(value))

    def _truncate(self, value: str) -> str:
        if len(value) <= self._max:
            return value
        return value[: self._max] + f"…[+{len(value) - self._max} chars]"


class TextFormatter(logging.Formatter):
    """Readable single line for a terminal: time, level, job, message, key=value extras."""

    def __init__(self, job: str, max_field_chars: int) -> None:
        super().__init__()
        self._job = job
        self._max = max_field_chars

    def format(self, record: logging.LogRecord) -> str:
        try:
            stamp = datetime.fromtimestamp(record.created, UTC).strftime("%H:%M:%S")
            run_id = getattr(record, "run_id", None)
            where = self._job if run_id is None else f"{self._job}#{run_id}"
            parts = [f"{stamp} {record.levelname:<7} {where}: {self._clean(record.getMessage())}"]
            parts += [f"{key}={self._clean(str(value))}" for key, value in _extras(record).items()]
            line = " ".join(parts)
            if record.exc_info:
                line += "\n" + self.formatException(record.exc_info)
            return line
        except Exception:
            return f"ERROR {self._job}: unloggable record"

    def _clean(self, value: str) -> str:
        return value[: self._max].translate(_ESCAPE_CONTROL)


class RunContextFilter(logging.Filter):
    """Adds run_id to each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _current_run_id.get()
        return True


def configure_logging(job: str, level: str, max_field_chars: int, fmt: str = "text") -> None:
    """Replace the root logger's handlers with one stderr handler. Safe to call twice."""
    handler = logging.StreamHandler(sys.stderr)
    formatter: logging.Formatter
    if fmt == "json":
        formatter = JsonFormatter(job, max_field_chars)
    else:
        formatter = TextFormatter(job, max_field_chars)
    handler.setFormatter(formatter)
    handler.addFilter(RunContextFilter())
    root = logging.getLogger()
    for old in list(root.handlers):
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(level)
    # httpx logs every request at INFO, including full URLs. The fetcher logs its own.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@contextmanager
def bind_run_id(run_id: int) -> Iterator[None]:
    """Set the run id for the block, then restore the previous one."""
    token = _current_run_id.set(run_id)
    try:
        yield
    finally:
        _current_run_id.reset(token)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
