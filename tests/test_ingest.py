"""The ingest job end to end, against fake sites served on localhost."""

from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from conftest import SettingsFactory
from lab import synth
from lab.config import ConfigError, Settings
from lab.corpus import store
from lab.ingest.job import run_ingest
from lab.ingest.models import Source, load_sources, parse_sources
from lab.runs import run


@pytest.fixture
def sites() -> Iterator[tuple[list[synth.Site], dict[str, str]]]:
    rng = random.Random(1)
    today = synth.fixed_today()
    wire = [(synth.make_title(rng, synth.make_style(rng, "prs")), synth.make_paragraphs(rng,
             synth.make_style(rng, "prs"), 5))]  # fmt: skip
    built = [
        synth.build_news_site(
            rng, "one", synth.make_style(rng, "prs"), 8, today=today, shared=wire
        ),
        synth.build_news_site(
            rng, "two", synth.make_style(rng, "pes"), 8, today=today, shared=wire
        ),
    ]
    with synth.serve(built) as urls:
        yield built, urls


def rss_source(name: str, base: str, label: str, **extra: object) -> Source:
    host = base.split("//", 1)[1]
    return Source(name, "rss", label, (f"{base}/feed.xml",), allowed_hosts=(host,),
                  content_selector="div.body", **extra)  # type: ignore[arg-type]  # fmt: skip


def status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM urls GROUP BY status")
    return {row["status"]: row["n"] for row in rows}


def test_crawl_stores_rejects_dedupes_and_respects_robots(
    settings: Settings, conn: sqlite3.Connection, sites: tuple[list[synth.Site], dict[str, str]]
) -> None:
    built, urls = sites
    sources = [rss_source("one", urls["one"], "prs"), rss_source("two", urls["two"], "pes")]
    with run(conn, "ingest") as current:
        reports = run_ingest(settings, sources, conn, current)

    # Per site: 8 articles + 1 wire story stored, 1 too short, 1 in English, 1 robots-blocked.
    # The second site's copy of the wire story is an exact duplicate of the first.
    assert status_counts(conn) == {"stored": 17, "duplicate": 1, "rejected": 4, "blocked": 2}
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 17
    assert {r.name: r.discovered for r in reports} == {"one": 12, "two": 12}
    for site in built:
        assert "/private/hidden.html" not in site.hits  # never requested
    labels = dict(
        conn.execute(
            "SELECT s.name, MIN(d.label) FROM documents d JOIN sources s ON s.id = d.source_id "
            "GROUP BY s.name"
        ).fetchall()
    )
    assert labels == {"one": "prs", "two": "pes"}
    metrics = json.loads(conn.execute("SELECT metrics FROM runs").fetchone()[0])
    assert metrics["stored"] == 17 and metrics["t_fetch_s"] > 0 and metrics["t_minhash_s"] > 0


def test_second_run_fetches_nothing_new(
    settings: Settings, conn: sqlite3.Connection, sites: tuple[list[synth.Site], dict[str, str]]
) -> None:
    built, urls = sites
    sources = [rss_source("one", urls["one"], "prs")]
    with run(conn, "ingest") as current:
        run_ingest(settings, sources, conn, current)
    before = dict(built[0].hits)
    with run(conn, "ingest") as current:
        run_ingest(settings, sources, conn, current)
    after = built[0].hits
    new_requests = {
        path: count - before.get(path, 0)
        for path, count in after.items()
        if count != before.get(path, 0)
    }
    # robots.txt is read once per run (rules may change), then the feed answers
    # 304 Not Modified, and no article is requested at all.
    assert new_requests == {"/robots.txt": 1, "/feed.xml": 1}
    assert current.metrics["feeds_not_modified"] == 1


def test_page_limit_and_quota_stop_a_source_early(
    settings: Settings, conn: sqlite3.Connection, sites: tuple[list[synth.Site], dict[str, str]]
) -> None:
    _built, urls = sites
    with run(conn, "ingest") as current:
        reports = run_ingest(settings, [rss_source("one", urls["one"], "prs")], conn, current,
                             max_pages=3)  # fmt: skip
    assert reports[0].fetched == 3 and reports[0].truncated
    # Validators are not saved for a truncated feed, so the next run sees the rest.
    assert store.get_http_cache(conn, f"{urls['one']}/feed.xml") == (None, None)

    with run(conn, "ingest") as current:
        run_ingest(settings, [rss_source("two", urls["two"], "pes", quota_chars=1)], conn, current)
    stored_two = conn.execute(
        "SELECT COUNT(*) FROM documents d JOIN sources s ON s.id = d.source_id WHERE s.name = 'two'"
    ).fetchone()[0]
    assert stored_two == 1  # the quota is checked before each page


def test_near_duplicates_are_detected(
    settings: Settings, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    rng = random.Random(2)
    style = synth.make_style(rng, "prs")
    paragraphs = synth.make_paragraphs(rng, style, 8)
    edited = [*paragraphs, synth.make_sentence(rng, style)]
    data = tmp_path / "data.jsonl"
    lines = [
        {"text": "\n".join(paragraphs), "url": "https://x.test/1"},
        {"text": "\n".join(edited), "url": "https://y.test/1"},
        {"text": synth.make_document(rng, style), "url": "https://z.test/1"},
    ]
    data.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n")
    source = Source("dump", "file", "prs", (str(data),))
    with run(conn, "ingest") as current:
        run_ingest(settings, [source], conn, current)
    assert status_counts(conn) == {"stored": 2, "near_duplicate": 1}
    duplicate_of = conn.execute(
        "SELECT document_id FROM urls WHERE status = 'near_duplicate'"
    ).fetchone()[0]
    assert duplicate_of == 1  # points at the original


def test_file_source_skips_bad_lines_and_known_entries(
    settings: Settings, conn: sqlite3.Connection, tmp_path: Path
) -> None:
    rng = random.Random(4)
    style = synth.make_style(rng, "pes")
    data = tmp_path / "data.jsonl"
    good = json.dumps({"text": synth.make_document(rng, style), "date": "2026-01-05"})
    data.write_text(f'not json\n{good}\n[1, 2]\n\n{{"text": 5}}\n', encoding="utf-8")
    source = Source("dump", "file", "pes", (str(data), str(tmp_path / "missing.jsonl")))
    with run(conn, "ingest") as current:
        reports = run_ingest(settings, [source], conn, current)
    assert current.metrics["invalid_lines"] == 3
    assert "missing.jsonl" in reports[0].error
    row = conn.execute("SELECT url, doc_date FROM documents").fetchone()
    assert (row["url"], row["doc_date"]) == ("file://dump/data.jsonl#2", "2026-01-05")
    with run(conn, "ingest") as current:
        run_ingest(settings, [source], conn, current)
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


def test_source_config_validation(tmp_path: Path, make_settings: SettingsFactory) -> None:
    good = {"name": "a", "kind": "rss", "label": "prs", "urls": ["https://a.test/feed"]}
    (source,) = parse_sources({"source": [good]}, "t", tmp_path)
    assert source.allowed_hosts == ("a.test",) and source.site == "a.test"

    bad = {
        "source": [
            {**good, "name": "Bad Name", "colour": "red"},
            {**good, "kind": "listing"},  # needs article_pattern
            {**good, "content_selector": "p[["},
            {**good, "urls": ["file:///etc/passwd"]},
            {**good, "article_pattern": "("},
        ]
    }
    with pytest.raises(ConfigError) as error:
        parse_sources(bad, "t", tmp_path)
    message = str(error.value)
    expected = [
        "name must be",
        "unknown key 'colour'",
        "needs article_pattern",
        "not a valid CSS selector",
        "not an http(s) URL",
        "not a valid regex",
    ]
    for fragment in expected:
        assert fragment in message

    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    (config / "sources.toml").write_text(
        '[[source]]\nname="a"\nkind="rss"\nlabel="x"\nurls=["https://a.test/"]\n'
    )
    (config / "sources.local.toml").write_text(
        '[[source]]\nname="a"\nkind="rss"\nlabel="y"\nurls=["https://a.test/"]\n'
    )
    assert [s.label for s in load_sources(config)] == ["y"]  # local file wins


def test_unknown_source_name_is_an_error(settings: Settings, conn: sqlite3.Connection) -> None:
    with pytest.raises(ConfigError, match="unknown source"), run(conn, "ingest") as current:
        run_ingest(settings, [], conn, current, only={"nope"})
