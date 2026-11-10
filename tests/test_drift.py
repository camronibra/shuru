"""News polling, headline scoring and drift."""

from __future__ import annotations

import json
import random
import sqlite3
from array import array
from pathlib import Path

import pytest

from lab import synth
from lab.config import ConfigError, DriftConfig, Settings
from lab.ingest.fetch import Fetcher
from lab.ingest.models import Source
from lab.model.predict import Model
from lab.news.drift import drift_report, psi, update_daily, window
from lab.news.poll import Feed, load_feeds, overlapping_feeds, poll_feeds
from lab.news.score import score_headlines
from lab.runs import Run


def tiny_model(conn: sqlite3.Connection) -> tuple[int, Model]:
    """A model that leans towards 'prs' a little, registered without a file."""
    model = Model(("pes", "prs"), 10, 1, 2, [array("f", [0.0] * 1024)] * 2, [0.0, 0.4])
    cursor = conn.execute(
        "INSERT INTO models (created_at, file_name, sha256, labels, params, metrics, is_active) "
        "VALUES ('x', 'm', 'h', '[]', '{}', '{}', 1)"
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid, model


def test_psi() -> None:
    assert psi([10, 20, 30], [10, 20, 30]) == pytest.approx(0.0)
    # Same shape, twice the items. Not exactly 0 because of the half-count smoothing.
    assert psi([10, 20, 30], [20, 40, 60]) == pytest.approx(0.0, abs=1e-3)
    assert psi([100, 0, 0], [0, 0, 100]) > 1.0  # everything moved
    assert 0 < psi([50, 50], [55, 45]) < 0.1  # a small wobble
    with pytest.raises(ValueError):
        psi([1, 2], [1, 2, 3])


def test_poll_score_and_daily_aggregates(
    settings: Settings, conn: sqlite3.Connection, run: Run
) -> None:
    rng = random.Random(3)
    site = synth.build_headline_feed(
        rng, "af", synth.make_style(rng, "prs"), 30, today=synth.fixed_today()
    )
    with synth.serve([site]) as urls:
        feeds = [Feed("af", f"{urls['af']}/feed.xml", "prs")]
        fetcher = Fetcher(settings.http)
        assert poll_feeds(settings, feeds, conn, run, fetcher) == 30
        assert poll_feeds(settings, feeds, conn, run, fetcher) == 0  # 304, nothing new
        fetcher.close()
    model_id, model = tiny_model(conn)
    assert score_headlines(conn, model_id, model, run) == 30
    assert score_headlines(conn, model_id, model, run) == 0
    assert update_daily(conn, model_id, "2000-01-01") >= 1
    today = conn.execute("SELECT MAX(day) FROM drift_daily").fetchone()[0]
    stats = window(conn, model_id, "2000-01-01", today)
    assert stats.n == 30 and sum(stats.hist) == 30
    assert stats.label_counts == {"prs": 30}  # the bias decides when no weights are set
    assert stats.agreement == 1.0


def test_drift_report_needs_two_full_windows_then_compares_them(conn: sqlite3.Connection) -> None:
    model_id, _ = tiny_model(conn)
    feed_id = conn.execute(
        "INSERT INTO news_feeds (name, url, created_at) VALUES ('f', 'u', 'x')"
    ).lastrowid

    def day_row(day: str, hist: list[int]) -> None:
        conn.execute(
            "INSERT INTO drift_daily "
            "(day, feed_id, model_id, n, conf_sum, conf_hist, label_counts) "
            "VALUES (?, ?, ?, ?, ?, ?, '{}')",
            (day, feed_id, model_id, sum(hist), 0.0, json.dumps(hist)),
        )

    config = DriftConfig(window_days=3, min_items=10, psi_alert=0.25)
    confident = [0, 0, 0, 0, 0, 1, 1, 2, 4, 12]
    for day in range(1, 4):
        day_row(f"2026-03-0{day}", confident)
    report = drift_report(conn, model_id, config)
    assert report.psi is None and "not enough history" in report.reason

    for day in range(4, 7):
        day_row(f"2026-03-0{day}", confident)
    report = drift_report(conn, model_id, config)
    assert report.psi == pytest.approx(0.0) and not report.alert
    assert report.current is not None
    assert (report.current.first_day, report.current.last_day) == ("2026-03-04", "2026-03-06")

    unsure = [0, 0, 0, 0, 12, 6, 2, 0, 0, 0]  # the model has become much less certain
    for day in range(7, 10):
        day_row(f"2026-03-0{day}", unsure)
    report = drift_report(conn, model_id, config)
    assert report.psi is not None and report.psi > 0.25 and report.alert


def test_feed_config_and_overlap_warning(tmp_path: Path) -> None:
    (tmp_path / "news_feeds.toml").write_text(
        '[[feed]]\nname = "a"\nurl = "https://news.test/rss"\nlabel = "prs"\n'
    )
    feeds = load_feeds(tmp_path)
    assert feeds == [Feed("a", "https://news.test/rss", "prs")]
    training = [
        Source("t", "rss", "prs", ("https://news.test/feed",), allowed_hosts=("news.test",))
    ]
    assert overlapping_feeds(feeds, training) == ["a"]
    (tmp_path / "news_feeds.local.toml").write_text('[[feed]]\nname = "B"\nurl = "ftp://x"\n')
    with pytest.raises(ConfigError, match="name must be"):
        load_feeds(tmp_path)
