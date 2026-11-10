"""The model registry: which model files exist, how they scored, which one is active.

A model is trained on one machine and used on another, so the registry treats the
file as the source of truth: it carries its own labels, parameters and metrics.
``install`` registers a file that was trained elsewhere. The registry stores the
file's SHA-256, and every load checks it, so a damaged or swapped file is refused.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from lab.db import transaction
from lab.model.predict import Model, ModelError, sha256_file


def register(conn: sqlite3.Connection, path: Path, model: Model, note: str = "") -> int:
    """Record a model file that already sits in the models folder. Returns its id."""
    digest = sha256_file(path)
    existing = conn.execute("SELECT id FROM models WHERE sha256 = ?", (digest,)).fetchone()
    if existing:
        return int(existing["id"])
    params = {
        "hash_bits": model.hash_bits,
        "ngram_min": model.ngram_min,
        "ngram_max": model.ngram_max,
    }
    cursor = conn.execute(
        "INSERT INTO models (created_at, file_name, sha256, labels, params, metrics, note) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            model.created_at,
            path.name,
            digest,
            json.dumps(list(model.labels)),
            json.dumps(params),
            json.dumps(model.metrics, ensure_ascii=False),
            note,
        ),
    )
    assert cursor.lastrowid is not None  # noqa: S101 - narrowing for the type checker
    return cursor.lastrowid


def install(conn: sqlite3.Connection, file: Path, models_dir: Path, note: str = "") -> int:
    """Copy a model file trained on another machine into the models folder and register it."""
    model = Model.load(file)  # validates the file before anything is copied
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / file.name
    if file.resolve() != target.resolve():
        if target.exists() and sha256_file(target) != sha256_file(file):
            raise ModelError(f"a different model named {file.name} is already installed")
        shutil.copyfile(file, target)
    return register(conn, target, model, note)


def activate(conn: sqlite3.Connection, model_id: int) -> None:
    """Make one model the active one. The others stay registered."""
    if not conn.execute("SELECT 1 FROM models WHERE id = ?", (model_id,)).fetchone():
        raise ModelError(f"no model with id {model_id}")
    with transaction(conn):
        conn.execute("UPDATE models SET is_active = 0 WHERE is_active = 1")
        conn.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))


def active_model_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT id FROM models WHERE is_active = 1").fetchone()
    return int(row["id"]) if row else None


def load_model(conn: sqlite3.Connection, model_id: int, models_dir: Path) -> Model:
    """Load a registered model from disk, verifying the recorded hash."""
    row = conn.execute("SELECT file_name, sha256 FROM models WHERE id = ?", (model_id,)).fetchone()
    if row is None:
        raise ModelError(f"no model with id {model_id}")
    return Model.load(models_dir / row["file_name"], expected_sha256=row["sha256"])


def load_active(conn: sqlite3.Connection, models_dir: Path) -> tuple[int, Model] | None:
    """The active model, or None if no model has been activated yet."""
    model_id = active_model_id(conn)
    if model_id is None:
        return None
    return model_id, load_model(conn, model_id, models_dir)


def list_models(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, created_at, file_name, labels, metrics, is_active, note FROM models ORDER BY id"
    ).fetchall()
