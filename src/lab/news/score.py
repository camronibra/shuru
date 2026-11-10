"""Score stored headlines with a model."""

from __future__ import annotations

import sqlite3

from lab.db import transaction
from lab.model.predict import Model
from lab.runs import Run
from lab.timeutil import utc_now

BATCH = 500


def score_headlines(conn: sqlite3.Connection, model_id: int, model: Model, run: Run) -> int:
    """Score every headline this model has not scored yet. Returns how many.

    Scores are keyed by (headline, model), so activating a new model scores the
    stored headlines again without losing the old model's results.
    """
    scored = 0
    while True:
        rows = conn.execute(
            "SELECT i.id, i.title FROM news_items i WHERE NOT EXISTS ("
            "  SELECT 1 FROM news_scores s WHERE s.item_id = i.id AND s.model_id = ?) "
            "ORDER BY i.id LIMIT ?",
            (model_id, BATCH),
        ).fetchall()
        if not rows:
            break
        now = utc_now()
        with run.step("score", cpu=True):
            batch = []
            for row in rows:
                prediction = model.predict(row["title"])
                batch.append(
                    (
                        row["id"],
                        model_id,
                        prediction.label,
                        prediction.confidence,
                        prediction.margin,
                        now,
                    )
                )
        with run.step("db_write"), transaction(conn):
            conn.executemany(
                "INSERT INTO news_scores (item_id, model_id, label, confidence, margin, scored_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                batch,
            )
        scored += len(batch)
    run.count("headlines_scored", scored)
    return scored
