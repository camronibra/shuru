"""The Source model and the loader for config/sources.toml.

A source is one place text comes from. Nothing about any outlet is written in code:
which URLs to read, how to recognise an article link, where the article text sits in
the page and which label its text gets are all configuration.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from selectolax.lexbor import LexborHTMLParser, SelectolaxError

from lab.config import ConfigError, read_toml

SOURCE_KINDS = ("rss", "sitemap", "listing", "file")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_KEYS = {
    "name",
    "kind",
    "label",
    "urls",
    "licence",
    "allowed_hosts",
    "article_pattern",
    "content_selector",
    "holdout",
    "quota_chars",
    "delay_s",
    "enabled",
}


@dataclass(frozen=True, slots=True)
class Source:
    """One configured source.

    kind decides how article URLs are found:

    - ``rss``: ``urls`` are RSS or Atom feeds; every entry link is an article.
    - ``sitemap``: ``urls`` are XML sitemaps (or sitemap indexes).
    - ``listing``: ``urls`` are HTML pages (a front page, a category page); links that
      match ``article_pattern`` are articles.
    - ``file``: ``urls`` are local JSONL files, one ``{"text": ...}`` object per line.
      For datasets you downloaded yourself; record their terms in ``licence``.
    """

    name: str
    kind: str
    label: str
    urls: tuple[str, ...]
    licence: str = ""
    # Hosts this source may fetch from. Links and redirects elsewhere are dropped.
    allowed_hosts: tuple[str, ...] = ()
    # Regex searched in the absolute URL. Required for listings, an optional filter otherwise.
    article_pattern: re.Pattern[str] | None = None
    # CSS selector for the article body. Empty = generic extraction (<article>, <main>, ...).
    content_selector: str = ""
    # True = never trained on; used to test the classifier on a source it has not seen.
    holdout: bool = False
    # Stop collecting from this source at this many stored characters. 0 = no limit.
    quota_chars: int = 0
    # Extra-polite delay for this source. The larger of this, the global default and
    # the site's robots.txt Crawl-delay is used.
    delay_s: float = 0.0
    enabled: bool = True

    @property
    def site(self) -> str:
        """The host that sets the pace for this source. One crawl thread per site."""
        return self.allowed_hosts[0] if self.allowed_hosts else ""


def load_sources(config_dir: Path) -> list[Source]:
    """Read sources.toml, then sources.local.toml (gitignored) on top of it.

    A local source with the same name replaces the committed one.
    """
    by_name: dict[str, Source] = {}
    for file_name in ("sources.toml", "sources.local.toml"):
        path = config_dir / file_name
        if not path.exists():
            continue
        for source in parse_sources(read_toml(path), str(path), config_dir):
            by_name[source.name] = source
    return list(by_name.values())


def parse_sources(raw: Mapping[str, Any], origin: str, config_dir: Path) -> list[Source]:
    """Validate the ``[[source]]`` tables of one file. Reports every problem at once."""
    problems: list[str] = []
    entries = raw.get("source", [])
    if not isinstance(entries, list):
        raise ConfigError(f"{origin}: expected [[source]] tables")
    for key in raw:
        if key != "source":
            problems.append(f"unknown top-level key {key!r}")

    sources: list[Source] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        where = f"source #{index}"
        if not isinstance(entry, dict):
            problems.append(f"{where}: must be a table")
            continue
        name = entry.get("name")
        if isinstance(name, str):
            where = f"source {name!r}"
        source = _parse_one(entry, where, config_dir, problems)
        if source is None:
            continue
        if source.name in seen:
            problems.append(f"{where}: name is used twice")
        seen.add(source.name)
        sources.append(source)
    if problems:
        raise ConfigError(f"{origin}: " + "; ".join(problems))
    return sources


def _parse_one(
    entry: Mapping[str, Any], where: str, config_dir: Path, problems: list[str]
) -> Source | None:
    before = len(problems)

    def bad(message: str) -> None:
        problems.append(f"{where}: {message}")

    for key in entry:
        if key not in _KEYS:
            bad(f"unknown key {key!r}")

    name = entry.get("name")
    if not isinstance(name, str) or not _NAME.match(name):
        bad("name must be lower-case letters, digits, '-' or '_'")
    kind = entry.get("kind")
    if kind not in SOURCE_KINDS:
        bad(f"kind must be one of {SOURCE_KINDS}")
    label = entry.get("label")
    if not isinstance(label, str) or not _LABEL.match(label):
        bad("label must be a short code such as 'prs' (letters, digits, '-' or '_')")

    urls = entry.get("urls")
    if not isinstance(urls, list) or not urls or not all(isinstance(u, str) and u for u in urls):
        bad("urls must be a non-empty list of strings")
        urls = []

    hosts: list[str] = []
    if kind == "file":
        urls = [str((config_dir / u).expanduser().resolve()) for u in urls]
    else:
        for url in urls:
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.netloc:
                bad(f"{url!r} is not an http(s) URL")
            elif parts.netloc.lower() not in hosts:
                hosts.append(parts.netloc.lower())
        extra_hosts = entry.get("allowed_hosts", [])
        if not isinstance(extra_hosts, list) or not all(isinstance(h, str) for h in extra_hosts):
            bad("allowed_hosts must be a list of host names")
        else:
            hosts += [h.lower() for h in extra_hosts if h.lower() not in hosts]

    pattern: re.Pattern[str] | None = None
    raw_pattern = entry.get("article_pattern", "")
    if not isinstance(raw_pattern, str):
        bad("article_pattern must be a string")
    elif raw_pattern:
        try:
            pattern = re.compile(raw_pattern)
        except re.error as exc:
            bad(f"article_pattern is not a valid regex: {exc}")
    elif kind == "listing":
        bad("a listing source needs article_pattern to tell article links from the rest")

    selector = entry.get("content_selector", "")
    if not isinstance(selector, str):
        bad("content_selector must be a string")
        selector = ""
    elif selector:
        try:
            LexborHTMLParser("<p></p>").css(selector)
        except SelectolaxError:
            bad(f"content_selector {selector!r} is not a valid CSS selector")

    licence = entry.get("licence", "")
    if not isinstance(licence, str):
        bad("licence must be a string")
    holdout = entry.get("holdout", False)
    enabled = entry.get("enabled", True)
    if not isinstance(holdout, bool) or not isinstance(enabled, bool):
        bad("holdout and enabled must be true or false")
    quota = entry.get("quota_chars", 0)
    if not isinstance(quota, int) or isinstance(quota, bool) or quota < 0:
        bad("quota_chars must be a whole number >= 0")
    delay = entry.get("delay_s", 0.0)
    if not isinstance(delay, int | float) or isinstance(delay, bool) or delay < 0:
        bad("delay_s must be a number >= 0")

    if len(problems) > before:
        return None
    return Source(
        name=str(name),
        kind=str(kind),
        label=str(label),
        urls=tuple(urls),
        licence=str(licence),
        allowed_hosts=tuple(hosts),
        article_pattern=pattern,
        content_selector=selector,
        holdout=bool(holdout),
        quota_chars=int(quota),
        delay_s=float(delay),
        enabled=bool(enabled),
    )
