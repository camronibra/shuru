"""Poll news feeds and store their headlines.

News feeds are configured apart from the training sources (config/news_feeds.toml)
and should be outlets the classifier was *not* trained on. Their headlines are
then genuinely unseen text: short, fresh and from unfamiliar outlets, which is the
hardest and most honest test the model gets.

Only headlines are stored, not articles.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from lab.config import ConfigError, Settings, read_toml
from lab.corpus import store
from lab.db import transaction
from lab.ingest.discover import DiscoverError, host_of, parse_feed
from lab.ingest.fetch import Fetcher, FetchError
from lab.ingest.models import Source
from lab.logs import get_logger
from lab.runs import Run
from lab.text.normalise import normalise
from lab.text.tokenise import SCRIPT_RANGES, script_ratio
from lab.timeutil import utc_now

log = get_logger(__name__)

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
MAX_TITLE_CHARS = 500
MIN_TITLE_CHARS = 10


@dataclass(frozen=True, slots=True)
class Feed:
    name: str
    url: str
    # What the outlet is assumed to publish, if known. Used to track how often the
    # model agrees; never used for training.
    label: str = ""
    enabled: bool = True


def load_feeds(config_dir: Path) -> list[Feed]:
    """Read news_feeds.toml, then news_feeds.local.toml (gitignored) on top of it."""
    by_name: dict[str, Feed] = {}
    for file_name in ("news_feeds.toml", "news_feeds.local.toml"):
        path = config_dir / file_name
        if not path.exists():
            continue
        entries = read_toml(path).get("feed", [])
        if not isinstance(entries, list):
            raise ConfigError(f"{path}: expected [[feed]] tables")
        problems: list[str] = []
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                problems.append(f"feed #{index}: must be a table")
                continue
            name, url = entry.get("name"), entry.get("url")
            label, enabled = entry.get("label", ""), entry.get("enabled", True)
            where = f"feed {name!r}" if isinstance(name, str) else f"feed #{index}"
            unknown = set(entry) - {"name", "url", "label", "enabled"}
            if unknown:
                problems.append(f"{where}: unknown key(s) {sorted(unknown)}")
            if not isinstance(name, str) or not _NAME.match(name):
                problems.append(f"{where}: name must be lower-case letters, digits, '-' or '_'")
            elif not isinstance(url, str) or urlsplit(url).scheme not in ("http", "https"):
                problems.append(f"{where}: url must be an http(s) URL")
            elif not isinstance(label, str) or (label and not _LABEL.match(label)):
                problems.append(f"{where}: label must be a short code or empty")
            elif not isinstance(enabled, bool):
                problems.append(f"{where}: enabled must be true or false")
            else:
                by_name[name] = Feed(name, url, label, enabled)
        if problems:
            raise ConfigError(f"{path}: " + "; ".join(problems))
    return list(by_name.values())


def overlapping_feeds(feeds: Iterable[Feed], sources: Iterable[Source]) -> list[str]:
    """Names of feeds that share a host with a training source (so are not unseen data)."""
    training_hosts = {host for s in sources if not s.holdout for host in s.allowed_hosts}
    return [feed.name for feed in feeds if host_of(feed.url) in training_hosts]


def sync_feeds(conn: sqlite3.Connection, feeds: Iterable[Feed]) -> dict[str, int]:
    for feed in feeds:
        conn.execute(
            "INSERT INTO news_feeds (name, url, expected_label, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (name) DO UPDATE SET url = excluded.url, "
            "expected_label = excluded.expected_label",
            (feed.name, feed.url, feed.label, utc_now()),
        )
    return {row["name"]: row["id"] for row in conn.execute("SELECT id, name FROM news_feeds")}


def poll_feeds(
    settings: Settings,
    feeds: list[Feed],
    conn: sqlite3.Connection,
    run: Run,
    fetcher: Fetcher,
) -> int:
    """Fetch every enabled feed and store headlines not seen before. Returns how many."""
    feed_ids = sync_feeds(conn, feeds)
    clean = settings.clean
    check_script = clean.min_script_ratio > 0 and clean.script in SCRIPT_RANGES
    new_items = 0
    for feed in feeds:
        if not feed.enabled:
            continue
        etag, last_modified = store.get_http_cache(conn, feed.url)
        try:
            result = fetcher.get(
                feed.url,
                allowed_hosts={host_of(feed.url)},
                etag=etag,
                last_modified=last_modified,
                timings=run.timings,
            )
        except FetchError as exc:
            run.count("feed_errors")
            log.warning("%s: could not fetch feed: %s", feed.name, exc)
            continue
        if result.not_modified:
            run.count("feeds_not_modified")
            continue
        try:
            with run.step("discover", cpu=True):
                entries = parse_feed(result.body, result.url)
        except DiscoverError as exc:
            run.count("feed_errors")
            log.warning("%s: could not parse feed: %s", feed.name, exc)
            continue

        now = utc_now()
        added = 0
        with run.step("db_write"), transaction(conn):
            for entry in entries[: settings.news.max_items_per_feed]:
                title = " ".join(normalise(entry.title).split())[:MAX_TITLE_CHARS]
                if len(title) < MIN_TITLE_CHARS:
                    continue
                if check_script and script_ratio(title, clean.script) < clean.min_script_ratio:
                    continue
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO news_items "
                    "(feed_id, guid, url, title, published_at, fetched_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (feed_ids[feed.name], entry.guid, entry.url, title, entry.published_at, now),
                )
                added += cursor.rowcount
            store.put_http_cache(conn, feed.url, result.etag, result.last_modified)
        new_items += added
        log.info("%s: %d entries, %d new", feed.name, len(entries), added)
    run.count("feeds_polled", sum(1 for feed in feeds if feed.enabled))
    run.count("headlines_new", new_items)
    return new_items
