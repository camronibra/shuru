"""Database snapshots.

``backup_database`` uses SQLite's online backup API: it copies a consistent image
of the database while other jobs may still be reading or writing, which a plain
file copy cannot guarantee. The copy is checked, compressed and the oldest
snapshots beyond ``keep`` are removed.

A snapshot is also the hand-off to the training machine: copy the newest file
there, unpack it as that machine's database and run ``shuru train``.
"""

from __future__ import annotations

import gzip
import shutil
import sqlite3
from pathlib import Path

from lab.logs import get_logger
from lab.timeutil import utc_now

log = get_logger(__name__)

PREFIX = "lab-"
SUFFIX = ".db.gz"


class BackupError(Exception):
    """The snapshot failed its integrity check."""


def backup_database(conn: sqlite3.Connection, backups_dir: Path, keep: int) -> Path:
    """Write backups/lab-<timestamp>.db.gz and prune old snapshots. Returns the new file."""
    backups_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().replace("-", "").replace(":", "")
    plain = backups_dir / f"{PREFIX}{stamp}.db"
    target = backups_dir / f"{PREFIX}{stamp}{SUFFIX}"
    try:
        copy = sqlite3.connect(plain)
        try:
            # 256 pages at a time, so writers are only briefly held up.
            conn.backup(copy, pages=256)
            verdict = copy.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            copy.close()
        if verdict != "ok":
            raise BackupError(f"snapshot failed its integrity check: {verdict}")
        partial = target.with_name(target.name + ".tmp")
        with plain.open("rb") as source, gzip.open(partial, "wb", compresslevel=6) as packed:
            shutil.copyfileobj(source, packed, length=1 << 20)
        partial.replace(target)
    finally:
        plain.unlink(missing_ok=True)

    snapshots = sorted(backups_dir.glob(f"{PREFIX}*{SUFFIX}"))
    for old in snapshots[:-keep]:
        old.unlink()
        log.info("removed old snapshot %s", old.name)
    return target
