from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lab.config import REPO_ROOT
from lab.db import DatabaseError, connect, dump_schema, migrate, transaction
from lab.notify import Notifier
from lab.runs import JobBusyError, Timings, job_lock, run


class RecordingNotifier(Notifier):
    def __init__(self) -> None:
        super().__init__("https://ntfy.invalid", "topic")
        self.sent: list[tuple[str, str]] = []

    def send(self, title: str, message: str, *, priority: int = 3) -> bool:
        self.sent.append((title, message))
        return True


def test_successful_run_is_recorded_with_metrics_and_timings(conn: sqlite3.Connection) -> None:
    with run(conn, "ingest") as current:
        current.count("stored", 2)
        current.count("stored")
        current.set("note", "fine")
        with current.step("fetch"):
            pass
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (current.id,)).fetchone()
    assert (row["job"], row["status"], row["error"]) == ("ingest", "ok", None)
    assert row["finished_at"] >= row["started_at"]
    metrics = json.loads(row["metrics"])
    assert metrics["stored"] == 3 and metrics["note"] == "fine"
    assert metrics["t_fetch_s"] >= 0


def test_failed_run_is_recorded_alerted_and_reraised(conn: sqlite3.Connection) -> None:
    notifier = RecordingNotifier()
    with pytest.raises(RuntimeError, match="boom"), run(conn, "news", notifier) as current:
        current.count("fetched", 5)
        raise RuntimeError("boom")
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (current.id,)).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "RuntimeError: boom"
    assert json.loads(row["metrics"])["fetched"] == 5  # what was counted before the crash
    assert notifier.sent == [("shuru: news failed", f"run {current.id}: RuntimeError: boom")]


def test_failure_inside_a_transaction_rolls_back_but_still_records(
    conn: sqlite3.Connection,
) -> None:
    with pytest.raises(sqlite3.IntegrityError), run(conn, "ingest") as current:
        conn.execute("BEGIN")
        conn.execute("INSERT INTO runs (job, status, started_at) VALUES ('x', 'bogus', 'now')")
    assert not conn.in_transaction
    statuses = [row["status"] for row in conn.execute("SELECT status FROM runs")]
    assert statuses == ["failed"]
    assert current.id == 1


def test_run_left_running_by_a_killed_process_is_closed_by_the_next(
    conn: sqlite3.Connection,
) -> None:
    conn.execute("INSERT INTO runs (job, status, started_at) VALUES ('ingest', 'running', 'x')")
    conn.execute("INSERT INTO runs (job, status, started_at) VALUES ('news', 'running', 'x')")
    with run(conn, "ingest"):
        pass
    rows = conn.execute("SELECT job, status, error FROM runs ORDER BY id").fetchall()
    assert (rows[0]["status"], rows[0]["error"][:11]) == ("failed", "interrupted")
    assert rows[1]["status"] == "running"  # another job's row is not touched
    assert rows[2]["status"] == "ok"


def test_job_lock_keeps_a_second_job_out(tmp_path: Path) -> None:
    lock = tmp_path / "db" / "jobs.lock"
    with job_lock(lock), pytest.raises(JobBusyError), job_lock(lock, timeout_s=0):
        pass
    with job_lock(lock):  # released again
        pass


def test_timings_accumulate_and_merge() -> None:
    first, second = Timings(), Timings()
    first.add("fetch", 1.5)
    second.add("fetch", 0.5)
    second.add("extract", 0.25)
    with second.step("extract", cpu=True):
        sum(range(1000))
    first.merge(second)
    assert first.seconds["fetch"] == 2.0
    assert first.seconds["extract"] >= 0.25


def test_migrations_apply_once_and_transaction_rolls_back(tmp_path: Path) -> None:
    connection = connect(tmp_path / "x.db")
    assert len(migrate(connection)) == 4
    assert migrate(connection) == []
    with pytest.raises(ZeroDivisionError), transaction(connection):
        connection.execute("INSERT INTO runs (job, status, started_at) VALUES ('a', 'ok', 'x')")
        _ = 1 / 0
    assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_broken_migration_leaves_nothing_behind(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_ok.sql").write_text("CREATE TABLE a (x INTEGER);")
    (migrations / "002_bad.sql").write_text("CREATE TABLE b (x INTEGER); CREATE TABLE a (y);")
    connection = connect(tmp_path / "x.db")
    with pytest.raises(DatabaseError, match=r"002_bad\.sql"):
        migrate(connection, migrations)
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
    assert "a" in tables and "b" not in tables  # 001 stays, 002 is rolled back entirely


def test_schema_file_is_in_sync_with_the_migrations(conn: sqlite3.Connection) -> None:
    """db/schema.sql is generated (`make schema`). This fails if someone forgot to."""
    committed = (REPO_ROOT / "db" / "schema.sql").read_text(encoding="utf-8")
    assert committed.split("\n\n", 1)[1] == dump_schema(conn)


def test_read_only_connection_cannot_write(conn: sqlite3.Connection, tmp_path: Path) -> None:
    path = Path(conn.execute("PRAGMA database_list").fetchone()["file"])
    reader = connect(path, read_only=True)
    with pytest.raises(sqlite3.OperationalError):
        reader.execute("INSERT INTO runs (job, status, started_at) VALUES ('a', 'ok', 'x')")
    with pytest.raises(DatabaseError, match="not found"):
        connect(tmp_path / "missing.db", read_only=True)
