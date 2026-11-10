"""SQLite access: connections, transactions and schema migrations.

There is no database server and no writer daemon. SQLite allows many readers and
exactly one writer at a time; in WAL mode readers are not blocked by the writer.
A second writer waits (``busy_timeout``) instead of failing. Heavy jobs also take a
lock file (see ``runs.job_lock``) so they never run side by side.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from lab.config import MIGRATIONS_DIR
from lab.timeutil import utc_now

# FTS5 and STRICT tables need a reasonably recent SQLite.
MIN_SQLITE = (3, 38, 0)
BUSY_TIMEOUT_MS = 30_000


class DatabaseError(Exception):
    """The database cannot be opened or migrated."""


def connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the database with the pragmas every connection needs.

    The connection is in autocommit mode: each statement commits on its own unless
    it runs inside ``transaction()``. Rows come back as ``sqlite3.Row`` (index by name).
    """
    if sqlite3.sqlite_version_info < MIN_SQLITE:
        wanted = ".".join(map(str, MIN_SQLITE))
        raise DatabaseError(f"SQLite {wanted}+ is required, found {sqlite3.sqlite_version}")
    if read_only:
        if not path.exists():
            raise DatabaseError(f"database not found: {path} (run `shuru init`)")
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, isolation_level=None)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    # Small fixed page cache (KiB when negative): memory use does not grow with the corpus.
    conn.execute("PRAGMA cache_size = -8000")
    if not read_only:
        conn.execute("PRAGMA journal_mode = WAL")
        # NORMAL is safe with WAL: a power cut can lose the last commits, not corrupt the file.
        conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Group statements into one atomic write: commit on success, roll back on error.

    BEGIN IMMEDIATE takes the write lock up front, so a busy database is noticed
    (and waited for) at the start instead of halfway through.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def applied_migrations(conn: sqlite3.Connection) -> set[str]:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "name TEXT PRIMARY KEY, applied_at TEXT NOT NULL) STRICT"
    )
    return {row["name"] for row in conn.execute("SELECT name FROM schema_migrations")}


def migrate(conn: sqlite3.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every migration file not yet recorded, in file-name order.

    Each file runs in its own transaction together with the row that records it, so
    a migration is applied completely or not at all. Returns the names applied.
    """
    files = sorted(migrations_dir.glob("*.sql"))
    if not files:
        raise DatabaseError(f"no migrations found in {migrations_dir}")
    done = applied_migrations(conn)
    applied: list[str] = []
    for file in files:
        if file.name in done:
            continue
        sql = file.read_text(encoding="utf-8")
        script = (
            "BEGIN IMMEDIATE;\n"
            f"{sql}\n"
            "INSERT INTO schema_migrations (name, applied_at) "
            f"VALUES ('{file.name}', '{utc_now()}');\n"
            "COMMIT;"
        )
        try:
            conn.executescript(script)
        except sqlite3.Error as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise DatabaseError(f"migration {file.name} failed: {exc}") from exc
        applied.append(file.name)
    return applied


def open_database(path: Path) -> sqlite3.Connection:
    """Connect for reading and writing, and bring the schema up to date."""
    conn = connect(path)
    migrate(conn)
    return conn


def dump_schema(conn: sqlite3.Connection) -> str:
    """The current schema as SQL text, for db/schema.sql. FTS5 shadow tables are left out."""
    rows = conn.execute(
        "SELECT sql FROM sqlite_schema "
        "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
        "AND name NOT LIKE 'documents_fts_%' AND name != 'schema_migrations' "
        "ORDER BY rowid"
    )
    return "".join(f"{row['sql']};\n\n" for row in rows).rstrip("\n") + "\n"


def file_size_mb(path: Path) -> float:
    """Size of a file in MiB, 0 if it does not exist."""
    try:
        return round(path.stat().st_size / (1024 * 1024), 2)
    except FileNotFoundError:
        return 0.0
