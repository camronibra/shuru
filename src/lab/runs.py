"""The runs table: one row per job execution, with status, duration and metrics.

Usage::

    with job_lock(settings.paths.lock_file), run(conn, "ingest", notifier) as r:
        with r.step("fetch"):
            ...
        r.count("stored", 3)

If the block raises, the run is marked failed, the error is recorded and an alert
is sent. The exception is then re-raised so the process exits non-zero.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from lab.logs import bind_run_id, get_logger
from lab.notify import Notifier
from lab.timeutil import utc_now

log = get_logger(__name__)

MAX_ERROR_CHARS = 500

try:  # POSIX only. Without it (Windows) the lock is skipped with a warning.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


class JobBusyError(Exception):
    """Another heavy job holds the lock and did not release it in time."""


class Timings:
    """Seconds spent per named step. Worker threads keep their own and merge at the end.

    Two clocks, because the two kinds of step need different ones:

    - Steps that wait (network, the polite delay, the database) use the wall clock.
    - Steps that compute (parsing, normalising, hashing) pass ``cpu=True`` and are
      measured in CPU time of the current thread. With several crawl threads only
      one runs Python code at any moment; the wall clock would also count the time
      a thread spent waiting for its turn and make every computation look slower
      than it is.
    """

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def step(self, name: str, *, cpu: bool = False) -> Iterator[None]:
        clock = time.thread_time if cpu else time.perf_counter
        started = clock()
        try:
            yield
        finally:
            self.add(name, clock() - started)

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds

    def merge(self, other: Timings) -> None:
        for name, seconds in other.seconds.items():
            self.add(name, seconds)


@dataclass
class Run:
    """Handle for the run in progress. Whatever lands in ``metrics`` is stored as JSON."""

    id: int
    job: str
    metrics: dict[str, int | float | str] = field(default_factory=dict)
    timings: Timings = field(default_factory=Timings)

    def count(self, key: str, n: int = 1) -> None:
        """Add n to a counter."""
        current = self.metrics.get(key, 0)
        self.metrics[key] = (current if isinstance(current, int) else 0) + n

    def set(self, key: str, value: int | float | str) -> None:
        self.metrics[key] = value

    def update(self, values: Mapping[str, int | float | str]) -> None:
        self.metrics.update(values)

    def step(self, name: str, *, cpu: bool = False) -> AbstractContextManager[None]:
        """Time a block: ``with run.step("extract"): ...`` adds to ``t_extract_s``."""
        return self.timings.step(name, cpu=cpu)

    def metrics_json(self) -> str:
        merged: dict[str, int | float | str] = dict(self.metrics)
        for name, seconds in sorted(self.timings.seconds.items()):
            merged[f"t_{name}_s"] = round(seconds, 3)
        return json.dumps(merged, ensure_ascii=False, sort_keys=True)


@contextmanager
def run(conn: sqlite3.Connection, job: str, notifier: Notifier | None = None) -> Iterator[Run]:
    """Record a job execution around the block. See the module docstring."""
    _close_interrupted(conn, job)
    cursor = conn.execute(
        "INSERT INTO runs (job, status, started_at) VALUES (?, 'running', ?)", (job, utc_now())
    )
    assert cursor.lastrowid is not None  # noqa: S101 - narrowing for the type checker
    current = Run(id=cursor.lastrowid, job=job)
    started = time.monotonic()

    def finish(status: str, error: str | None) -> None:
        if conn.in_transaction:  # the block died mid-transaction
            conn.execute("ROLLBACK")
        conn.execute(
            "UPDATE runs SET status = ?, finished_at = ?, duration_s = ?, error = ?, metrics = ? "
            "WHERE id = ?",
            (
                status,
                utc_now(),
                round(time.monotonic() - started, 3),
                error,
                current.metrics_json(),
                current.id,
            ),
        )

    with bind_run_id(current.id):
        log.info("run started")
        try:
            yield current
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
            finish("failed", error)
            log.error("run failed: %s", error)
            if notifier is not None and isinstance(exc, Exception):
                notifier.send(f"shuru: {job} failed", f"run {current.id}: {error}", priority=4)
            raise
        else:
            finish("ok", None)
            # The numbers are in the runs table (`shuru runs --id N`); keep the log line short.
            log.info("run finished in %.1fs", time.monotonic() - started)
            log.debug("run metrics: %s", current.metrics_json())


def _close_interrupted(conn: sqlite3.Connection, job: str) -> None:
    """Mark leftover 'running' rows of this job as failed.

    A run that was killed (power cut, out of memory) never reaches its own cleanup.
    The row would say 'running' forever, so the next run of the same job closes it.
    """
    conn.execute(
        "UPDATE runs SET status = 'failed', finished_at = ?, "
        "error = 'interrupted: the process ended without finishing the run' "
        "WHERE job = ? AND status = 'running'",
        (utc_now(), job),
    )


@contextmanager
def job_lock(path: Path, timeout_s: float = 0.0) -> Iterator[None]:
    """Hold an exclusive lock file so two heavy jobs never overlap.

    The lock is tied to the open file, so it is released by the kernel even if the
    process is killed. Waits up to timeout_s, then raises JobBusyError.
    """
    if fcntl is None:  # pragma: no cover
        log.warning("file locking is not available on this platform; running without the job lock")
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    with path.open("a") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise JobBusyError(f"another job holds {path}") from None
                time.sleep(1.0)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def recent_runs(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, job, status, started_at, duration_s, error, metrics "
        "FROM runs ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
