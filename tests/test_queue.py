"""The labelling queue, applying hand labels, and the label report."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from lab.config import LabellingConfig
from lab.labelling.apply import apply_sheet, known_labels, label_report
from lab.labelling.queue import SHEET_COLUMNS, export_sheet, pending_items, refresh_queue


@pytest.fixture
def scored(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Two feeds with 10 scored headlines each; margins grow with the item id."""
    conn.execute(
        "INSERT INTO sources (name, kind, label, created_at) VALUES ('s', 'rss', 'prs', 'x')"
    )
    conn.execute(
        "INSERT INTO models (id, created_at, file_name, sha256, labels, params, metrics) "
        "VALUES (1, 'x', 'm1', 'h1', '[]', '{}', '{}'), (2, 'x', 'm2', 'h2', '[]', '{}', '{}')"
    )
    for feed_id, (name, label) in enumerate((("af", "prs"), ("ir", "pes")), start=1):
        conn.execute(
            "INSERT INTO news_feeds (id, name, url, expected_label, created_at) "
            "VALUES (?, ?, 'u', ?, 'x')",
            (feed_id, name, label),
        )
    for item in range(1, 21):
        feed_id = 1 if item <= 10 else 2
        title = '=HYPERLINK("http://evil")' if item == 1 else f"سرخط شماره {item}\tبا\nخط"
        conn.execute(
            "INSERT INTO news_items (id, feed_id, guid, title, fetched_at) "
            "VALUES (?, ?, ?, ?, 'x')",
            (item, feed_id, f"g{item}", title),
        )
        predicted = "prs" if item % 3 else "pes"
        conn.execute(
            "INSERT INTO news_scores (item_id, model_id, label, confidence, margin, scored_at) "
            "VALUES (?, 1, ?, ?, ?, 'x')",
            (item, predicted, 0.5 + item / 50, item / 25),
        )
    return conn


def test_queue_takes_the_least_certain_with_a_per_feed_cap(scored: sqlite3.Connection) -> None:
    assert refresh_queue(scored, 1, LabellingConfig(queue_size=8, max_per_feed=5)) == 8
    queued = [row["item_id"] for row in pending_items(scored, 100)]
    # Feed 1 (items 1-10) has the lowest margins but is capped at 5.
    assert queued == [1, 2, 3, 4, 5, 11, 12, 13]
    assert refresh_queue(scored, 1, LabellingConfig(queue_size=8, max_per_feed=5)) == 0  # full


def test_a_new_model_replaces_pending_items(scored: sqlite3.Connection) -> None:
    refresh_queue(scored, 1, LabellingConfig(queue_size=4, max_per_feed=4))
    scored.execute(
        "INSERT INTO news_scores SELECT item_id, 2, label, confidence, 1 - margin, scored_at "
        "FROM news_scores WHERE model_id = 1"
    )
    refresh_queue(scored, 2, LabellingConfig(queue_size=4, max_per_feed=4))
    rows = scored.execute("SELECT item_id, model_id FROM label_queue ORDER BY item_id").fetchall()
    assert {row["model_id"] for row in rows} == {2}
    assert [row["item_id"] for row in rows] == [17, 18, 19, 20]  # least certain for model 2


def test_sheet_roundtrip(scored: sqlite3.Connection, tmp_path: Path) -> None:
    refresh_queue(scored, 1, LabellingConfig(queue_size=20, max_per_feed=10))
    sheet = tmp_path / "sheet.tsv"
    assert export_sheet(scored, sheet, limit=20) == 20
    with sheet.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert tuple(rows[0]) == SHEET_COLUMNS
    assert rows[0]["title"].startswith("'=")  # a formula would run in a spreadsheet
    assert all("\t" not in row["title"] and "\n" not in row["title"] for row in rows)

    rows[0]["label"] = "prs"
    rows[1]["label"] = "skip"
    rows[2]["label"] = "english"
    rows[3]["label"] = "pes"
    rows[3]["note"] = "Iranian wire copy"
    rows[4]["item_id"] = "999"
    rows[4]["label"] = "prs"
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    allowed = known_labels(scored)
    assert allowed == {"prs", "pes"}
    result = apply_sheet(scored, sheet, allowed, "tester")
    assert (result.applied, result.skipped, result.blank) == (2, 1, 15)
    assert "unknown label 'english'" in result.errors[0] and "id 999" in result.errors[1]
    again = apply_sheet(scored, sheet, allowed, "tester")  # applying twice changes nothing
    assert again.applied == 2
    assert scored.execute("SELECT COUNT(*) FROM human_labels").fetchone()[0] == 2
    statuses = dict(scored.execute("SELECT item_id, status FROM label_queue WHERE item_id <= 2"))
    assert statuses == {1: "labelled", 2: "skipped"}

    # Labelled and skipped items never come back into the queue.
    scored.execute("DELETE FROM label_queue WHERE status = 'pending'")
    refresh_queue(scored, 1, LabellingConfig(queue_size=20, max_per_feed=20))
    assert 1 not in [row["item_id"] for row in pending_items(scored, 100)]


def test_label_report_compares_hand_labels_with_distant_labels_and_the_model(
    scored: sqlite3.Connection,
) -> None:
    for item, label in ((2, "prs"), (3, "pes"), (12, "pes"), (13, "prs")):
        scored.execute(
            "INSERT INTO human_labels (item_id, label, labelled_at) VALUES (?, ?, 'x')",
            (item, label),
        )
    report = label_report(scored, model_id=1)
    assert report["n_labelled"] == 4
    # Feed af expects prs (items 2, 3), feed ir expects pes (items 12, 13): two of four agree.
    assert report["distant_vs_human"]["accuracy"] == 0.5
    # The model said prs for 2, 13 and pes for 3, 12: all four right.
    assert report["model_vs_human"]["accuracy"] == 1.0


def test_sheet_without_the_right_columns(conn: sqlite3.Connection, tmp_path: Path) -> None:
    sheet = tmp_path / "bad.tsv"
    sheet.write_text("a\tb\n1\t2\n")
    assert apply_sheet(conn, sheet, {"prs"}).errors == [
        "the sheet needs 'item_id' and 'label' columns"
    ]
