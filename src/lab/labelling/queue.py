"""The hand-labelling queue: which headlines a person should label next.

Labelling is the expensive step, so the queue asks for the labels that teach the
most. Those are the headlines the model is least sure about: where the top two
probabilities are closest (the smallest *margin*). A headline the model already
calls with 99% confidence would tell us little. This is uncertainty sampling, the
simplest form of active learning.

Work happens on a sheet: ``export_sheet`` writes the pending items to a
tab-separated file, a person fills in the ``label`` column in any editor or
spreadsheet, and ``apply.apply_sheet`` reads it back.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from lab.config import LabellingConfig
from lab.db import transaction
from lab.timeutil import utc_now

SHEET_COLUMNS = ("item_id", "feed", "model_label", "confidence", "label", "note", "title")
# A cell starting with one of these is run as a formula by spreadsheet programs.
_FORMULA_STARTS = ("=", "+", "-", "@")


def refresh_queue(conn: sqlite3.Connection, model_id: int, config: LabellingConfig) -> int:
    """Top the queue up to ``queue_size`` pending items. Returns how many were added.

    Pending items queued by an older model are dropped first: what was uncertain
    for the old model may be easy for the new one. No feed gets more than
    ``max_per_feed`` pending items, so one busy outlet cannot fill the queue.
    """
    added = 0
    with transaction(conn):
        conn.execute(
            "DELETE FROM label_queue WHERE status = 'pending' AND model_id != ?", (model_id,)
        )
        per_feed = {
            row["feed_id"]: row["n"]
            for row in conn.execute(
                "SELECT i.feed_id, COUNT(*) AS n FROM label_queue q "
                "JOIN news_items i ON i.id = q.item_id WHERE q.status = 'pending' "
                "GROUP BY i.feed_id"
            )
        }
        room = config.queue_size - sum(per_feed.values())
        if room <= 0:
            return 0
        candidates = conn.execute(
            "SELECT s.item_id, s.margin, i.feed_id FROM news_scores s "
            "JOIN news_items i ON i.id = s.item_id "
            "WHERE s.model_id = ? "
            "AND s.item_id NOT IN (SELECT item_id FROM label_queue) "
            "AND s.item_id NOT IN (SELECT item_id FROM human_labels) "
            "ORDER BY s.margin ASC, s.item_id LIMIT ?",
            (model_id, config.queue_size * 10),
        ).fetchall()
        now = utc_now()
        for row in candidates:
            if added >= room:
                break
            if per_feed.get(row["feed_id"], 0) >= config.max_per_feed:
                continue
            conn.execute(
                "INSERT INTO label_queue (item_id, model_id, margin, queued_at) "
                "VALUES (?, ?, ?, ?)",
                (row["item_id"], model_id, row["margin"], now),
            )
            per_feed[row["feed_id"]] = per_feed.get(row["feed_id"], 0) + 1
            added += 1
    return added


def pending_items(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    """Pending queue items, least certain first, with the model's guess."""
    return conn.execute(
        "SELECT q.item_id, f.name AS feed, i.title, s.label AS model_label, s.confidence, "
        "q.margin FROM label_queue q "
        "JOIN news_items i ON i.id = q.item_id JOIN news_feeds f ON f.id = i.feed_id "
        "LEFT JOIN news_scores s ON s.item_id = q.item_id AND s.model_id = q.model_id "
        "WHERE q.status = 'pending' ORDER BY q.margin ASC, q.item_id LIMIT ?",
        (limit,),
    ).fetchall()


def _safe_cell(text: str) -> str:
    """Make untrusted text safe for a spreadsheet cell: one line, never a formula."""
    text = " ".join(text.split())
    return "'" + text if text.startswith(_FORMULA_STARTS) else text


def export_sheet(conn: sqlite3.Connection, path: Path, limit: int) -> int:
    """Write the pending items to a TSV file for labelling. Returns the number of rows.

    The ``label`` and ``note`` columns are left empty for the person to fill in.
    """
    rows = pending_items(conn, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(SHEET_COLUMNS)
        for row in rows:
            confidence = "" if row["confidence"] is None else f"{row['confidence']:.3f}"
            writer.writerow(
                (
                    row["item_id"],
                    row["feed"],
                    row["model_label"] or "",
                    confidence,
                    "",
                    "",
                    _safe_cell(row["title"]),
                )
            )
    return len(rows)
