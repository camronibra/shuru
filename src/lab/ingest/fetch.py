"""Polite, bounded HTTP fetching.

Every request made by the pipeline goes through ``Fetcher.get``. In order, it:

1. refuses anything that is not http(s) or not on the source's allowed hosts,
2. checks the site's robots.txt,
3. waits until the per-site delay has passed since the last request to that site,
4. downloads with a hard cap on size and total time,
5. retries temporary failures a few times, waiting longer each time,
6. follows redirects itself, so every hop goes through steps 1 to 4 again.

A page is untrusted input from the first byte: the cap applies to the decompressed
size (a small gzip file can expand to gigabytes), only text content types are
accepted, and a server that trickles bytes is cut off.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from lab.config import HttpConfig
from lab.ingest.robots import MAX_ROBOTS_BYTES, RobotsCache
from lab.logs import get_logger
from lab.runs import Timings

log = get_logger(__name__)

_REDIRECTS = (301, 302, 303, 307, 308)
_RETRY_STATUSES = (429, 500, 502, 503, 504)
# Statuses that mean "the site is unhappy with us", counted towards giving up on it.
_PUSHBACK_STATUSES = (401, 403, 429)
_TEXT_TYPES = (
    "text/",
    "application/xhtml",
    "application/xml",
    "application/rss",
    "application/atom",
)
_GZIP_TYPES = ("application/gzip", "application/x-gzip", "application/octet-stream")
MAX_RETRY_AFTER_S = 120.0
_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_\-]+)""", re.IGNORECASE)


class FetchError(Exception):
    """A URL could not be fetched. ``kind`` says why in one word.

    Kinds: scheme, off_site, robots, host_down, redirects, too_large, too_slow,
    content_type, http, network.
    """

    def __init__(self, kind: str, detail: str = "", status: int | None = None) -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail
        self.status = status

    @property
    def blocked(self) -> bool:
        """True when we chose not to fetch (rules), as opposed to trying and failing."""
        return self.kind in ("scheme", "off_site", "robots")


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str  # final URL, after redirects
    status: int  # 200, or 304 when the conditional request matched
    body: bytes
    content_type: str
    etag: str | None
    last_modified: str | None

    @property
    def not_modified(self) -> bool:
        return self.status == 304


@dataclass(frozen=True, slots=True)
class _Raw:
    status: int
    headers: httpx.Headers
    body: bytes


class Throttle:
    """Keeps requests to one site at least ``delay`` seconds apart. Thread-safe."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._next_allowed: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str, delay: float) -> float:
        """Block until host may be contacted again. Returns the seconds slept."""
        with self._lock:
            now = self._clock()
            start = max(now, self._next_allowed.get(host, 0.0))
            self._next_allowed[host] = start + delay
        pause = start - now
        if pause > 0:
            self._sleep(pause)
        return max(pause, 0.0)


def make_client(config: HttpConfig) -> httpx.Client:
    """The HTTP client: identifies itself, never follows redirects on its own."""
    return httpx.Client(
        headers={
            "User-Agent": config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.5",
        },
        timeout=httpx.Timeout(config.timeout_s),
        follow_redirects=False,
    )


class Fetcher:
    """Shared by all crawl threads. See the module docstring for what ``get`` does."""

    def __init__(
        self,
        config: HttpConfig,
        client: httpx.Client | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._client = client if client is not None else make_client(config)
        self._owns_client = client is None
        self._clock = clock
        self._sleep = sleep
        self._throttle = Throttle(clock, sleep)
        self._failures: dict[str, int] = {}
        self._lock = threading.Lock()
        self.robots = RobotsCache(config.user_agent, self._fetch_robots)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def get(
        self,
        url: str,
        *,
        allowed_hosts: Collection[str],
        etag: str | None = None,
        last_modified: str | None = None,
        min_delay_s: float = 0.0,
        timings: Timings | None = None,
    ) -> FetchResult:
        """Fetch url politely. Raises FetchError; never returns a partial body.

        etag and last_modified are the validators saved from an earlier fetch of the
        same URL. Sending them back lets the server reply 304 with no body.
        """
        timings = timings if timings is not None else Timings()
        headers: dict[str, str] = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        for _hop in range(self._config.max_redirects + 1):
            parts = urlsplit(url)
            host = parts.netloc.lower()
            if parts.scheme not in ("http", "https") or not host:
                raise FetchError("scheme", parts.scheme)
            if host not in allowed_hosts:
                raise FetchError("off_site", host)
            if self._given_up(host):
                raise FetchError("host_down", host)
            with timings.step("robots"):
                rules = self.robots.rules_for(url)
            if not rules.allowed(url):
                raise FetchError("robots", url)

            delay = max(self._config.default_delay_s, min_delay_s, rules.crawl_delay or 0.0)
            raw = self._request(url, delay, headers, timings)
            if raw.status in _REDIRECTS:
                location = raw.headers.get("location")
                if not location:
                    raise FetchError("http", "redirect without a Location header", raw.status)
                url = urljoin(url, location)
                headers = {}  # the validators belonged to the original URL
                continue
            if raw.status == 304:
                return FetchResult(url, 304, b"", "", etag, last_modified)
            return FetchResult(
                url=url,
                status=raw.status,
                body=raw.body,
                content_type=raw.headers.get("content-type", ""),
                etag=raw.headers.get("etag"),
                last_modified=raw.headers.get("last-modified"),
            )
        raise FetchError("redirects", f"more than {self._config.max_redirects}")

    def _given_up(self, host: str) -> bool:
        with self._lock:
            return self._failures.get(host, 0) >= self._config.max_host_failures

    def _note(self, host: str, *, failed: bool) -> None:
        with self._lock:
            self._failures[host] = self._failures.get(host, 0) + 1 if failed else 0

    def _request(
        self,
        url: str,
        delay: float,
        headers: dict[str, str],
        timings: Timings,
        *,
        max_bytes: int | None = None,
        any_content_type: bool = False,
    ) -> _Raw:
        """One URL, no redirects: throttle, download, and retry temporary failures."""
        host = urlsplit(url).netloc.lower()
        limit = self._config.max_bytes if max_bytes is None else max_bytes
        attempts = self._config.max_retries + 1
        error = FetchError("network", "no attempt made")
        for attempt in range(attempts):
            timings.add("wait", self._throttle.wait(host, delay))
            retry_after = 0.0
            try:
                with timings.step("fetch"):
                    raw = self._download(url, headers, limit, any_content_type)
            except httpx.TransportError as exc:
                error = FetchError("network", type(exc).__name__)
            else:
                if raw.status in _RETRY_STATUSES:
                    error = FetchError("http", f"status {raw.status}", raw.status)
                    retry_after = _retry_after(raw.headers)
                elif raw.status >= 400:
                    self._note(host, failed=raw.status in _PUSHBACK_STATUSES)
                    raise FetchError("http", f"status {raw.status}", raw.status)
                else:
                    self._note(host, failed=False)
                    return raw
            if attempt + 1 < attempts:
                # Back off: twice as long each time, or as long as the server asked.
                pause = max(retry_after, max(delay, 1.0) * 2**attempt)
                log.debug("retrying %s in %.1fs (%s)", url, pause, error)
                timings.add("wait", pause)
                self._sleep(pause)
        self._note(host, failed=True)
        raise error

    def _download(
        self, url: str, headers: dict[str, str], limit: int, any_content_type: bool
    ) -> _Raw:
        deadline = self._clock() + self._config.timeout_s * 3
        with self._client.stream("GET", url, headers=headers) as response:
            status = response.status_code
            if status != 200:
                return _Raw(status, response.headers, b"")
            content_type = response.headers.get("content-type", "").lower()
            if content_type and not any_content_type and not _acceptable(content_type):
                raise FetchError("content_type", content_type[:80])
            declared = response.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > limit:
                raise FetchError("too_large", f"{declared} bytes declared")
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > limit:
                    raise FetchError("too_large", f"more than {limit} bytes")
                if self._clock() > deadline:
                    raise FetchError("too_slow", "download exceeded the time limit")
                chunks.append(chunk)
            return _Raw(status, response.headers, b"".join(chunks))

    def _fetch_robots(self, robots_url: str) -> tuple[int, bytes]:
        """Download a robots.txt, following redirects (the RFC allows up to five)."""
        timings = Timings()
        url = robots_url
        for _hop in range(6):
            if urlsplit(url).scheme not in ("http", "https"):
                break
            try:
                raw = self._request(
                    url,
                    self._config.default_delay_s,
                    {},
                    timings,
                    max_bytes=MAX_ROBOTS_BYTES * 4,
                    any_content_type=True,
                )
            except FetchError as exc:
                # An HTTP error status is an answer (404 = no rules, so all allowed),
                # not a failure. RobotsCache decides what each status means.
                if exc.status is not None:
                    return exc.status, b""
                raise
            if raw.status in _REDIRECTS and raw.headers.get("location"):
                url = urljoin(url, raw.headers["location"])
                continue
            return raw.status, raw.body
        raise FetchError("redirects", "robots.txt")


def _acceptable(content_type: str) -> bool:
    return content_type.startswith(_TEXT_TYPES) or content_type.startswith(_GZIP_TYPES)


def _retry_after(headers: httpx.Headers) -> float:
    """Seconds the server asked us to wait (Retry-After), capped. 0 if absent or a date."""
    value = headers.get("retry-after", "").strip()
    if value.isdigit():
        return min(float(value), MAX_RETRY_AFTER_S)
    return 0.0


def decode_body(body: bytes, content_type: str) -> str:
    """Bytes to text. Charset from the header, else a <meta charset> near the top, else UTF-8.

    Undecodable bytes become U+FFFD instead of raising: one bad byte should not lose a page.
    """
    charset = ""
    match = re.search(r"charset\s*=\s*[\"']?([A-Za-z0-9_\-]+)", content_type, re.IGNORECASE)
    if match:
        charset = match.group(1)
    else:
        meta = _META_CHARSET.search(body[:4096])
        if meta:
            charset = meta.group(1).decode("ascii", errors="replace")
    try:
        return body.decode(charset or "utf-8", errors="replace")
    except LookupError:  # a charset name Python does not know
        return body.decode("utf-8", errors="replace")
