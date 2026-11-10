"""The ingest job: discover article URLs, fetch the new ones, clean and store them.

How the work is split between threads::

    crawl thread per site  --+
    crawl thread per site  --+--> bounded queue --> main thread: dedupe + write
    crawl thread per site  --+

- **One crawl thread per site.** Most of a crawl is waiting: for the network and
  for the polite delay between two requests to the same site. Threads let ten
  sites wait at the same time, while each site still sees one slow visitor.
- **The main thread is the only writer.** It takes finished pages off the queue,
  checks them against the corpus for duplicates and writes them in batches.
  Crawl threads only read the database (their own read-only connection).
- **The queue is bounded.** If writing falls behind, crawl threads block instead
  of piling pages up in memory.

Every step is timed. The totals are saved with the run, so "where does the time
go?" is answered by data: see ``shuru runs``.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from lab.config import ConfigError, Settings
from lab.corpus import store
from lab.corpus.clean import Rejected, prepare
from lab.corpus.store import NewDocument
from lab.db import connect, transaction
from lab.ingest.discover import (
    Discovered,
    DiscoverError,
    gunzip_if_needed,
    host_of,
    parse_feed,
    parse_listing,
    parse_sitemap,
)
from lab.ingest.extract import extract
from lab.ingest.fetch import Fetcher, FetchError, decode_body
from lab.ingest.models import Source
from lab.logs import bind_run_id, get_logger
from lab.notify import Notifier
from lab.runs import Run, Timings
from lab.text.dedup import MinHasher
from lab.text.normalise import normalise
from lab.timeutil import parse_timestamp, utc_now

log = get_logger(__name__)

QUEUE_SIZE = 200
MAX_CHILD_SITEMAPS = 10
MAX_TITLE_CHARS = 500
# A source that fetched at least this many pages and kept none probably has a
# broken selector or changed its layout.
MIN_PAGES_FOR_QUALITY_CHECK = 5


@dataclass(frozen=True, slots=True)
class PageOutcome:
    """What happened to one article URL. ``status`` is a value of urls.status."""

    source: str
    url: str
    status: str
    detail: str = ""
    http_status: int | None = None
    document: NewDocument | None = None
    n_bytes: int = 0


@dataclass(frozen=True, slots=True)
class CacheUpdate:
    """New ETag / Last-Modified for a feed, to send back on the next run."""

    url: str
    etag: str | None
    last_modified: str | None


@dataclass(slots=True)
class SourceReport:
    name: str
    discovered: int = 0
    new: int = 0
    fetched: int = 0
    not_modified: int = 0
    truncated: bool = False
    error: str = ""
    outcomes: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SiteDone:
    """Last message from a crawl thread."""

    site: str
    timings: Timings
    reports: list[SourceReport]


class _SiteDown(Exception):
    """The site failed too often; stop asking it for the rest of this run."""


def make_hasher(settings: Settings) -> MinHasher:
    dedup = settings.dedup
    return MinHasher(dedup.num_perm, dedup.bands, dedup.shingle_size)


class _SiteCrawler:
    """Runs in a crawl thread: everything for the sources of one site, in order."""

    def __init__(
        self,
        settings: Settings,
        fetcher: Fetcher,
        hasher: MinHasher,
        out: queue.Queue[PageOutcome | CacheUpdate | SiteDone],
        stop: threading.Event,
        run_id: int,
        max_pages: int,
    ) -> None:
        self._settings = settings
        self._fetcher = fetcher
        self._hasher = hasher
        self._out = out
        self._stop = stop
        self._run_id = run_id
        self._max_pages = max_pages

    def crawl(self, site: str, sources: list[Source], budgets: dict[str, int | None]) -> None:
        timings = Timings()
        reports: list[SourceReport] = []
        seen: set[str] = set()
        try:
            with bind_run_id(self._run_id):
                reader = connect(self._settings.paths.db_file, read_only=True)
                try:
                    for source in sources:
                        if self._stop.is_set():
                            break
                        report = SourceReport(source.name)
                        reports.append(report)
                        try:
                            self._crawl_source(source, reader, timings, report, seen, budgets)
                        except _SiteDown:
                            report.error = "site failed too often; skipped for this run"
                            report.truncated = True
                            log.warning("%s: %s", source.name, report.error)
                        except Exception as exc:
                            report.error = f"{type(exc).__name__}: {exc}"[:200]
                            log.exception("%s: crawl failed", source.name)
                finally:
                    reader.close()
        finally:
            self._out.put(SiteDone(site, timings, reports))

    def _crawl_source(
        self,
        source: Source,
        reader: sqlite3.Connection,
        timings: Timings,
        report: SourceReport,
        seen: set[str],
        budgets: dict[str, int | None],
    ) -> None:
        ingest = self._settings.ingest
        deadline = time.monotonic() + ingest.max_seconds_per_source
        candidates, cache_updates = self._discover(source, reader, timings, report)
        report.discovered = len(candidates)

        with timings.step("known_urls"):
            fresh = set(
                store.filter_new_urls(
                    reader, [c.url for c in candidates], ingest.retry_after_hours, ingest.max_tries
                )
            )
        to_fetch = [c for c in candidates if c.url in fresh and c.url not in seen]
        report.new = len(to_fetch)
        if len(to_fetch) > self._max_pages:
            to_fetch = to_fetch[: self._max_pages]
            report.truncated = True

        budget = budgets.get(source.name)
        for candidate in to_fetch:
            out_of_budget = budget is not None and budget <= 0
            if self._stop.is_set() or time.monotonic() > deadline or out_of_budget:
                report.truncated = True
                break
            seen.add(candidate.url)
            outcome = self._fetch_page(source, candidate, timings)
            report.fetched += 1
            if outcome.document is not None and budget is not None:
                budget -= outcome.document.prepared.n_chars
            self._out.put(outcome)

        # Only remember the feed's validators once everything it listed was handled.
        # Otherwise a "304 Not Modified" next time would hide the pages we skipped.
        if not report.truncated:
            for update in cache_updates:
                self._out.put(update)

    def _discover(
        self,
        source: Source,
        reader: sqlite3.Connection,
        timings: Timings,
        report: SourceReport,
    ) -> tuple[list[Discovered], list[CacheUpdate]]:
        """Read the source's feeds, sitemaps or listing pages. Returns the article URLs."""
        found: dict[str, Discovered] = {}
        updates: list[CacheUpdate] = []
        pending = [(url, 0) for url in source.urls]
        children_followed = 0
        while pending and not self._stop.is_set():
            url, depth = pending.pop(0)
            etag, last_modified = store.get_http_cache(reader, url)
            try:
                result = self._fetcher.get(
                    url,
                    allowed_hosts=source.allowed_hosts,
                    etag=etag,
                    last_modified=last_modified,
                    min_delay_s=source.delay_s,
                    timings=timings,
                )
            except FetchError as exc:
                if exc.kind == "host_down":
                    raise _SiteDown from None
                report.error = f"{url}: {exc}"[:200]
                log.warning("%s: could not read %s: %s", source.name, url, exc)
                continue
            if result.not_modified:
                report.not_modified += 1
                continue

            items: list[Discovered] = []
            try:
                with timings.step("discover", cpu=True):
                    if source.kind == "rss":
                        items = parse_feed(result.body, result.url)
                    elif source.kind == "sitemap":
                        body = gunzip_if_needed(result.body, self._settings.http.max_bytes)
                        items, children = parse_sitemap(body, result.url)
                        if depth == 0:
                            # Newest child sitemaps first; undated ones keep their order.
                            children.sort(key=lambda c: c.published_at or "", reverse=True)
                            for child in children[: MAX_CHILD_SITEMAPS - children_followed]:
                                pending.append((child.url, 1))
                                children_followed += 1
                    elif source.article_pattern is not None:
                        html = decode_body(result.body, result.content_type)
                        items = parse_listing(html, result.url, source.article_pattern)
            except DiscoverError as exc:
                report.error = f"{url}: {exc}"[:200]
                log.warning("%s: could not parse %s: %s", source.name, url, exc)
                continue

            updates.append(CacheUpdate(url, result.etag, result.last_modified))
            for item in items:
                if host_of(item.url) not in source.allowed_hosts:
                    continue
                if source.article_pattern is not None and not source.article_pattern.search(
                    item.url
                ):
                    continue
                found.setdefault(item.url, item)
        return list(found.values()), updates

    def _fetch_page(self, source: Source, candidate: Discovered, timings: Timings) -> PageOutcome:
        """Fetch one article and run it through extraction and cleaning."""
        try:
            result = self._fetcher.get(
                candidate.url,
                allowed_hosts=source.allowed_hosts,
                min_delay_s=source.delay_s,
                timings=timings,
            )
        except FetchError as exc:
            if exc.kind == "host_down":
                raise _SiteDown from None
            status = "blocked" if exc.blocked else "error"
            return PageOutcome(source.name, candidate.url, status, exc.kind, exc.status)

        fetched_at = utc_now()
        with timings.step("decode", cpu=True):
            html = decode_body(result.body, result.content_type)
        with timings.step("extract", cpu=True):
            extracted = extract(html, source.content_selector)
        if not extracted.text:
            return PageOutcome(source.name, candidate.url, "rejected", "no_text", result.status)

        prepared = prepare(extracted.text, self._settings.clean, self._hasher, timings)
        if isinstance(prepared, Rejected):
            return PageOutcome(
                source.name, candidate.url, "rejected", prepared.reason, result.status
            )
        document = NewDocument(
            url=candidate.url,
            title=normalise(extracted.title or candidate.title)[:MAX_TITLE_CHARS],
            published_at=extracted.published_at or candidate.published_at,
            fetched_at=fetched_at,
            prepared=prepared,
        )
        return PageOutcome(
            source.name, candidate.url, "stored", "", result.status, document, len(result.body)
        )


class _Writer:
    """Runs in the main thread: the only code that writes during an ingest run."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        settings: Settings,
        sources: dict[str, Source],
        source_ids: dict[str, int],
        run: Run,
    ) -> None:
        self._conn = conn
        self._threshold = settings.dedup.threshold
        self._batch_size = settings.ingest.batch_size
        self._sources = sources
        self._source_ids = source_ids
        self._run = run
        self._batch: list[PageOutcome | CacheUpdate] = []
        self.per_source: dict[str, dict[str, int]] = {}

    def add(self, message: PageOutcome | CacheUpdate) -> None:
        self._batch.append(message)
        if len(self._batch) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        """Write the pending batch in one transaction."""
        if not self._batch:
            return
        started = time.perf_counter()
        lookups_before = self._run.timings.seconds.get("dedup_lookup", 0.0)
        with transaction(self._conn):
            for message in self._batch:
                if isinstance(message, CacheUpdate):
                    store.put_http_cache(
                        self._conn, message.url, message.etag, message.last_modified
                    )
                else:
                    self._apply(message)
        self._batch.clear()
        lookups = self._run.timings.seconds.get("dedup_lookup", 0.0) - lookups_before
        self._run.timings.add("db_write", time.perf_counter() - started - lookups)

    def _apply(self, outcome: PageOutcome) -> None:
        source = self._sources[outcome.source]
        source_id = self._source_ids[outcome.source]
        status, document_id = outcome.status, None
        if outcome.document is not None:
            with self._run.step("dedup_lookup"):
                duplicate = store.find_duplicate(
                    self._conn, outcome.document.url, outcome.document.prepared, self._threshold
                )
            if duplicate is not None:
                status, document_id = duplicate
            else:
                document_id = store.insert_document(
                    self._conn, source_id, source.label, outcome.document
                )
                self._run.count("chars_stored", outcome.document.prepared.n_chars)
        store.record_url(
            self._conn,
            source_id,
            outcome.url,
            status,
            detail=outcome.detail,
            http_status=outcome.http_status,
            document_id=document_id,
        )
        self._run.count(status)
        self._run.count("bytes_fetched", outcome.n_bytes)
        counts = self.per_source.setdefault(outcome.source, {})
        counts[status] = counts.get(status, 0) + 1


def run_ingest(
    settings: Settings,
    sources: list[Source],
    conn: sqlite3.Connection,
    run: Run,
    *,
    notifier: Notifier | None = None,
    client: httpx.Client | None = None,
    only: set[str] | None = None,
    max_pages: int | None = None,
) -> list[SourceReport]:
    """Run one ingest pass over the configured sources. Returns a report per source."""
    active = [s for s in sources if s.enabled and (only is None or s.name in only)]
    if only:
        unknown = only - {s.name for s in sources}
        if unknown:
            raise ConfigError(f"unknown source(s): {', '.join(sorted(unknown))}")
    if not active:
        log.warning("no sources to ingest: add some to config/sources.local.toml")
        return []

    source_ids = store.sync_sources(conn, sources)
    used = store.chars_by_source(conn)
    budgets: dict[str, int | None] = {
        s.name: (s.quota_chars - used.get(s.name, 0)) if s.quota_chars else None for s in active
    }
    hasher = make_hasher(settings)
    writer = _Writer(conn, settings, {s.name: s for s in active}, source_ids, run)
    page_limit = max_pages if max_pages is not None else settings.ingest.max_pages_per_source
    reports: list[SourceReport] = []

    # Local files need no politeness and no threads.
    for source in active:
        if source.kind == "file":
            reports.append(
                _ingest_file(source, settings, hasher, conn, writer, run, budgets, page_limit)
            )
    writer.flush()

    # Web sources, grouped so that each site is crawled by exactly one thread.
    sites: dict[str, list[Source]] = {}
    for source in active:
        if source.kind != "file":
            sites.setdefault(source.site, []).append(source)

    if sites:
        fetcher = Fetcher(settings.http, client)
        out: queue.Queue[PageOutcome | CacheUpdate | SiteDone] = queue.Queue(maxsize=QUEUE_SIZE)
        stop = threading.Event()
        crawler = _SiteCrawler(settings, fetcher, hasher, out, stop, run.id, page_limit)
        workers = min(settings.ingest.max_workers, len(sites))
        pending = len(sites)
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="crawl") as pool:
                for site, group in sites.items():
                    pool.submit(crawler.crawl, site, group, budgets)
                try:
                    while pending:
                        try:
                            message = out.get(timeout=1.0)
                        except queue.Empty:
                            writer.flush()  # nothing arriving: make what we have durable
                            continue
                        if isinstance(message, SiteDone):
                            pending -= 1
                            run.timings.merge(message.timings)
                            reports.extend(message.reports)
                        else:
                            writer.add(message)
                    writer.flush()
                except BaseException:
                    # Tell the crawl threads to stop, then keep emptying the queue so
                    # none of them stays blocked on a full queue while we shut down.
                    stop.set()
                    while pending:
                        if isinstance(out.get(), SiteDone):
                            pending -= 1
                    raise
        finally:
            fetcher.close()

    _summarise(run, reports, writer, notifier, len(sites))
    return reports


def _ingest_file(
    source: Source,
    settings: Settings,
    hasher: MinHasher,
    conn: sqlite3.Connection,
    writer: _Writer,
    run: Run,
    budgets: dict[str, int | None],
    page_limit: int,
) -> SourceReport:
    """Read a local JSONL dataset line by line: {"text": ..., "url"?, "title"?, "date"?}."""
    report = SourceReport(source.name)
    budget = budgets.get(source.name)
    max_line_chars = settings.clean.max_chars * 8
    for path in source.urls:
        try:
            with Path(path).open(encoding="utf-8", errors="replace") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if report.fetched >= page_limit or (budget is not None and budget <= 0):
                        report.truncated = True
                        break
                    if not line.strip() or len(line) > max_line_chars:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        run.count("invalid_lines")
                        continue
                    if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                        run.count("invalid_lines")
                        continue
                    report.discovered += 1
                    key = record.get("url")
                    if not isinstance(key, str) or not key:
                        key = f"file://{source.name}/{Path(path).name}#{line_no}"
                    if not store.filter_new_urls(conn, [key], 0, 0):
                        continue
                    report.new += 1
                    report.fetched += 1
                    prepared = prepare(record["text"], settings.clean, hasher, run.timings)
                    if isinstance(prepared, Rejected):
                        writer.add(PageOutcome(source.name, key, "rejected", prepared.reason))
                        continue
                    title = record.get("title")
                    date = record.get("date")
                    document = NewDocument(
                        url=key,
                        title=normalise(title)[:MAX_TITLE_CHARS] if isinstance(title, str) else "",
                        published_at=parse_timestamp(date) if isinstance(date, str) else None,
                        fetched_at=utc_now(),
                        prepared=prepared,
                    )
                    if budget is not None:
                        budget -= prepared.n_chars
                    writer.add(PageOutcome(source.name, key, "stored", document=document))
        except OSError as exc:
            report.error = f"{path}: {type(exc).__name__}"
            log.warning("%s: cannot read %s", source.name, path)
    return report


def _summarise(
    run: Run,
    reports: list[SourceReport],
    writer: _Writer,
    notifier: Notifier | None,
    n_sites: int,
) -> None:
    """Fill the run metrics and flag sources that fetched pages but kept nothing."""
    run.set("sources", len(reports))
    run.set("sites", n_sites)
    run.set("discovered", sum(r.discovered for r in reports))
    run.set("new_urls", sum(r.new for r in reports))
    run.set("fetched", sum(r.fetched for r in reports))
    run.set("feeds_not_modified", sum(r.not_modified for r in reports))
    run.set("sources_with_errors", sum(1 for r in reports if r.error))

    silent: list[str] = []
    for report in reports:
        report.outcomes = writer.per_source.get(report.name, {})
        kept = sum(report.outcomes.get(s, 0) for s in ("stored", "duplicate", "near_duplicate"))
        if report.fetched >= MIN_PAGES_FOR_QUALITY_CHECK and kept == 0:
            silent.append(report.name)
        log.info(
            "%s: discovered=%d new=%d fetched=%d %s%s",
            report.name,
            report.discovered,
            report.new,
            report.fetched,
            " ".join(f"{k}={v}" for k, v in sorted(report.outcomes.items())),
            f" error={report.error}" if report.error else "",
        )
    run.set("sources_without_text", len(silent))
    if silent and notifier is not None:
        notifier.send(
            "shuru: sources returning no text",
            "Pages were fetched but none produced usable text (check content_selector): "
            + ", ".join(silent),
        )
