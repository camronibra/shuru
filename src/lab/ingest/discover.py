"""Find article URLs: in RSS/Atom feeds, XML sitemaps and HTML listing pages.

Feeds and sitemaps are the cheap way to learn what a site has published: one small
request lists dozens of articles, often with titles and dates, and the site meant
them to be read by machines.

XML is parsed with defusedxml, which refuses the entity tricks ("billion laughs")
that make a tiny file expand without limit.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit
from xml.etree.ElementTree import Element, ParseError

from defusedxml import DefusedXmlException
from defusedxml.ElementTree import fromstring
from selectolax.lexbor import LexborHTMLParser

from lab.timeutil import parse_timestamp

MAX_URL_CHARS = 2000
MAX_TITLE_CHARS = 500
# Query parameters that only track where a click came from. Removing them stops the
# same article from being fetched once per campaign.
_TRACKING = re.compile(r"^(utm_\w+|fbclid|gclid|mc_cid|mc_eid|igshid)$", re.IGNORECASE)


class DiscoverError(Exception):
    """A feed or sitemap could not be parsed."""


@dataclass(frozen=True, slots=True)
class Discovered:
    """An article URL, with whatever the feed already told us about it."""

    url: str
    title: str = ""
    published_at: str | None = None
    guid: str = ""


def canonical_url(url: str, base: str | None = None) -> str | None:
    """One spelling per page: absolute, lower-case host, no fragment, no tracking parameters.

    Returns None for anything that is not a usable http(s) URL.
    """
    url = url.strip()
    if base:
        url = urljoin(base, url)
    if not url or len(url) > MAX_URL_CHARS:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    query = "&".join(
        pair for pair in parts.query.split("&") if pair and not _TRACKING.match(pair.split("=")[0])
    )
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", query, ""))


def _local(tag: object) -> str:
    """Tag name without its XML namespace: '{http://www.w3.org/2005/Atom}entry' -> 'entry'."""
    return tag.rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _text(element: Element) -> str:
    return " ".join("".join(element.itertext()).split())


def _parse_xml(body: bytes) -> Element:
    try:
        root: Element = fromstring(body)
    except (ParseError, DefusedXmlException, ValueError) as exc:
        raise DiscoverError(f"not valid XML: {type(exc).__name__}") from None
    return root


def gunzip_if_needed(body: bytes, max_bytes: int) -> bytes:
    """Sitemaps are often served as .xml.gz. Decompress, but never beyond max_bytes."""
    if not body.startswith(b"\x1f\x8b"):
        return body
    decompressor = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
    try:
        data = decompressor.decompress(body, max_bytes + 1)
    except zlib.error:
        raise DiscoverError("corrupt gzip data") from None
    if len(data) > max_bytes:
        raise DiscoverError("decompressed sitemap is too large")
    return data


def parse_feed(body: bytes, base_url: str) -> list[Discovered]:
    """Entries of an RSS 2.0, RSS 1.0 or Atom feed, in feed order."""
    root = _parse_xml(body)
    found: list[Discovered] = []
    for element in root.iter():
        if _local(element.tag) not in ("item", "entry"):
            continue
        link = title = guid = ""
        date: str | None = None
        for child in element:
            name = _local(child.tag)
            if name == "link":
                # Atom: <link href="..." rel="alternate">. RSS: <link>...</link>.
                href = child.get("href")
                if href is not None:
                    if child.get("rel", "alternate") == "alternate" and not link:
                        link = href
                elif not link:
                    link = _text(child)
            elif name == "title" and not title:
                title = _text(child)[:MAX_TITLE_CHARS]
            elif name in ("guid", "id") and not guid:
                guid = _text(child)[:MAX_URL_CHARS]
            elif name in ("pubdate", "published", "updated", "date") and date is None:
                date = parse_timestamp(_text(child))
        url = canonical_url(link or guid, base_url)
        if url:
            found.append(Discovered(url, title, date, guid or url))
    return found


def parse_sitemap(body: bytes, base_url: str) -> tuple[list[Discovered], list[Discovered]]:
    """URLs in a sitemap, plus the child sitemaps if this is a sitemap index.

    A child's ``published_at`` is its <lastmod>, so the caller can read the newest first.
    """
    root = _parse_xml(body)
    pages: list[Discovered] = []
    children: list[Discovered] = []
    for element in root.iter():
        kind = _local(element.tag)
        if kind not in ("url", "sitemap"):
            continue
        loc = title = ""
        date: str | None = None
        for child in element.iter():
            name = _local(child.tag)
            if name == "loc" and not loc:
                loc = _text(child)
            elif name == "title" and not title:  # <news:title> in news sitemaps
                title = _text(child)[:MAX_TITLE_CHARS]
            elif name in ("publication_date", "lastmod") and date is None:
                date = parse_timestamp(_text(child))
        url = canonical_url(loc, base_url)
        if not url:
            continue
        entry = Discovered(url, title, date, url)
        if kind == "sitemap":
            children.append(entry)
        else:
            pages.append(entry)
    return pages, children


def parse_listing(html: str, base_url: str, pattern: re.Pattern[str]) -> list[Discovered]:
    """Links on an HTML page whose absolute URL matches pattern, in page order."""
    seen: dict[str, Discovered] = {}
    for anchor in LexborHTMLParser(html).css("a[href]"):
        url = canonical_url(anchor.attributes.get("href") or "", base_url)
        if url and url not in seen and pattern.search(url):
            seen[url] = Discovered(url, guid=url)
    return list(seen.values())


def host_of(url: str) -> str:
    return urlsplit(url).netloc.lower()
