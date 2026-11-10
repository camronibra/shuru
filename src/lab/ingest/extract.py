"""Pull the article out of an HTML page: title, date and body text.

The page is parsed, never executed or rendered. Scripts, styles, navigation and
other page furniture are removed first. Then the body is taken from the source's
``content_selector`` if it has one, or from the page's own ``<article>`` or
``<main>`` element. The result is plain text with one paragraph per line.

A selector per source is the precise option. The generic fallback is good enough
for well-structured sites and keeps a new source working before it has one.
"""

from __future__ import annotations

from dataclasses import dataclass

from selectolax.lexbor import LexborHTMLParser, LexborNode

from lab.timeutil import parse_timestamp

# Elements that never hold article text.
_JUNK_TAGS = [
    "script",
    "style",
    "noscript",
    "template",
    "iframe",
    "svg",
    "form",
    "nav",
    "header",
    "footer",
    "aside",
    "button",
    "figure",
]
_BLOCKS = "p, h2, h3, h4, li, blockquote, pre"
_GENERIC_CONTAINERS = ("article", "main", "[role=main]")
_DATE_METAS = (
    'meta[property="article:published_time"]',
    'meta[name="article:published_time"]',
    'meta[itemprop="datePublished"]',
    'meta[name="date"]',
    'meta[name="pubdate"]',
    'meta[name="DC.date.issued"]',
)
MAX_TITLE_CHARS = 500


@dataclass(frozen=True, slots=True)
class Extracted:
    title: str
    published_at: str | None
    text: str  # paragraphs separated by "\n"; empty if no body was found


def _squash(text: str) -> str:
    """Collapse every run of whitespace to one space."""
    return " ".join(text.split())


def _block_texts(container: LexborNode) -> list[str]:
    """Text of each paragraph-like element inside container, outermost wrappers skipped."""
    texts: list[str] = []
    for block in container.css(_BLOCKS):
        # <li><p>..</p></li>: the inner <p> is also in the list, so skip the wrapper.
        if block.tag in ("li", "blockquote") and block.css_first("p") is not None:
            continue
        text = _squash(block.text(deep=True, separator="", strip=False))
        if text:
            texts.append(text)
    if not texts:  # no paragraph markup at all: take the container's text as it is
        raw = container.text(deep=True, separator="\n", strip=False)
        texts = [line for line in (_squash(line) for line in raw.split("\n")) if line]
    return texts


def _title(tree: LexborHTMLParser) -> str:
    meta = tree.css_first('meta[property="og:title"]')
    if meta is not None and meta.attributes.get("content"):
        return _squash(meta.attributes["content"] or "")[:MAX_TITLE_CHARS]
    for selector in ("h1", "title"):
        node = tree.css_first(selector)
        if node is not None:
            text = _squash(node.text(deep=True, separator=" ", strip=False))
            if text:
                return text[:MAX_TITLE_CHARS]
    return ""


def _published(tree: LexborHTMLParser) -> str | None:
    for selector in _DATE_METAS:
        meta = tree.css_first(selector)
        if meta is not None:
            stamp = parse_timestamp(meta.attributes.get("content"))
            if stamp:
                return stamp
    time_node = tree.css_first("time[datetime]")
    if time_node is not None:
        return parse_timestamp(time_node.attributes.get("datetime"))
    return None


def extract(html: str, content_selector: str = "") -> Extracted:
    """Parse one page. Never raises on bad HTML: the parser repairs what it is given."""
    tree = LexborHTMLParser(html)
    # Metadata lives in <head> and <header>, so read it before stripping anything.
    title = _title(tree)
    published_at = _published(tree)

    tree.strip_tags(_JUNK_TAGS)
    for line_break in tree.css("br"):
        line_break.replace_with("\n")

    containers: list[LexborNode] = []
    if content_selector:
        containers = tree.css(content_selector)
    else:
        for selector in _GENERIC_CONTAINERS:
            candidates = tree.css(selector)
            if candidates:
                # A page can hold several <article> teasers; the real one has the most text.
                containers = [max(candidates, key=lambda node: len(node.text(deep=True)))]
                break
        if not containers and tree.body is not None:
            containers = [tree.body]

    paragraphs: list[str] = []
    seen: set[str] = set()
    for container in containers:
        for text in _block_texts(container):
            if text not in seen:  # repeated blocks are almost always page furniture
                seen.add(text)
                paragraphs.append(text)
    return Extracted(title=title, published_at=published_at, text="\n".join(paragraphs))
