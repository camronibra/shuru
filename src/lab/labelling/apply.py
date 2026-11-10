"""Apply hand labels from a filled-in sheet, and report what they show.

Hand labels live in their own table, apart from the distant labels that come from
"which outlet published this". Keeping them apart is what makes two measurements
possible:

- **How noisy is distant supervision?** Compare each hand label with the label
  its feed was expected to have.
- **How good is the model really?** Compare each hand label with the model's
  prediction. This is the only accuracy figure that does not rest on the
  assumption being tested.
"""

from __future__ import annotations

import csv
import sqlite3
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab.db import transaction
from lab.model.evaluate import score_predictions
from lab.timeutil import utc_now

SKIP = "skip"
MAX_NOTE_CHARS = 500
MAX_ERRORS_REPORTED = 20


@dataclass(slots=True)
class ApplyResult:
    applied: int = 0
    skipped: int = 0
    blank: int = 0
    errors: list[str] = field(default_factory=list)


def known_labels(conn: sqlite3.Connection) -> set[str]:
    """Labels a person may assign: every source label and expected feed label."""
    labels = {row["label"] for row in conn.execute("SELECT DISTINCT label FROM sources")}
    labels |= {
        row["expected_label"]
        for row in conn.execute(
            "SELECT DISTINCT expected_label FROM news_feeds WHERE expected_label != ''"
        )
    }
    return labels


def apply_sheet(
    conn: sqlite3.Connection, path: Path, allowed: Collection[str], annotator: str = ""
) -> ApplyResult:
    """Read a labelling sheet and store its labels.

    Per row: an empty label leaves the item pending, ``skip`` removes it from the
    queue without a label (not Persian-script, unreadable, mixed), and anything else
    must be one of ``allowed``. Bad rows are reported and the rest are still applied.
    Applying the same sheet twice changes nothing.
    """
    result = ApplyResult()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"item_id", "label"} <= set(reader.fieldnames):
            result.errors.append("the sheet needs 'item_id' and 'label' columns")
            return result
        rows = list(reader)

    now = utc_now()
    with transaction(conn):
        for line_no, row in enumerate(rows, start=2):
            label = (row.get("label") or "").strip()
            if not label:
                result.blank += 1
                continue
            raw_id = (row.get("item_id") or "").strip()
            if not raw_id.isdigit():
                result.errors.append(f"line {line_no}: item_id {raw_id!r} is not a number")
                continue
            item_id = int(raw_id)
            if not conn.execute("SELECT 1 FROM news_items WHERE id = ?", (item_id,)).fetchone():
                result.errors.append(f"line {line_no}: no headline with id {item_id}")
                continue
            if label.lower() == SKIP:
                conn.execute(
                    "UPDATE label_queue SET status = 'skipped' WHERE item_id = ?", (item_id,)
                )
                result.skipped += 1
                continue
            if label not in allowed:
                result.errors.append(
                    f"line {line_no}: unknown label {label!r} (use one of {sorted(allowed)}"
                    f" or '{SKIP}')"
                )
                continue
            note = (row.get("note") or "").strip()[:MAX_NOTE_CHARS]
            conn.execute(
                "INSERT INTO human_labels (item_id, label, annotator, note, labelled_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (item_id) DO UPDATE SET "
                "label = excluded.label, annotator = excluded.annotator, note = excluded.note, "
                "labelled_at = excluded.labelled_at",
                (item_id, label, annotator, note, now),
            )
            conn.execute("UPDATE label_queue SET status = 'labelled' WHERE item_id = ?", (item_id,))
            result.applied += 1
    del result.errors[MAX_ERRORS_REPORTED:]
    return result


def label_report(conn: sqlite3.Connection, model_id: int | None) -> dict[str, Any]:
    """What the hand labels say about the distant labels and about the model.

    ``distant_vs_human``: the feed's expected label scored against the hand label.
    Its accuracy is an estimate of how often "outlet country = variety" holds.
    ``model_vs_human``: the model's prediction scored against the hand label.

    Read ``model_vs_human`` with care: the queue picks the headlines the model was
    least sure of, so this figure is measured on its hardest cases and understates
    its accuracy on headlines in general.
    """
    report: dict[str, Any] = {
        "n_labelled": conn.execute("SELECT COUNT(*) FROM human_labels").fetchone()[0],
        "pending": conn.execute(
            "SELECT COUNT(*) FROM label_queue WHERE status = 'pending'"
        ).fetchone()[0],
        "by_label": {
            row["label"]: row["n"]
            for row in conn.execute(
                "SELECT label, COUNT(*) AS n FROM human_labels GROUP BY label ORDER BY label"
            )
        },
    }
    distant = conn.execute(
        "SELECT h.label, f.expected_label, f.name FROM human_labels h "
        "JOIN news_items i ON i.id = h.item_id JOIN news_feeds f ON f.id = i.feed_id "
        "WHERE f.expected_label != ''"
    )
    report["distant_vs_human"] = score_predictions(
        (row["label"], row["expected_label"], row["name"]) for row in distant
    )
    if model_id is not None:
        predicted = conn.execute(
            "SELECT h.label, s.label AS predicted, f.name FROM human_labels h "
            "JOIN news_scores s ON s.item_id = h.item_id AND s.model_id = ? "
            "JOIN news_items i ON i.id = h.item_id JOIN news_feeds f ON f.id = i.feed_id",
            (model_id,),
        )
        report["model_vs_human"] = score_predictions(
            (row["label"], row["predicted"], row["name"]) for row in predicted
        )
    return report
