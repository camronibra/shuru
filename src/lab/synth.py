"""Synthetic Persian-script text and websites, for tests, benchmarks and the demo.

**This is not real data.** Sentences are random draws from a small word list. They
have realistic word lengths, character sets and spelling noise, which is what the
tests and benchmarks need, but they mean nothing, and no claim about real Dari or
Iranian Persian follows from any score measured on them.

Two made-up "varieties", A and B, differ in three ways the classifier can pick up:

- a few marker words (the real Dari / Iranian Persian pairs from config/markers.toml),
- how often they use certain style words,
- nothing else. Most words are shared, as with real close varieties.

The module can also build small fake news sites (HTML pages, RSS feed, robots.txt)
and serve them from localhost, so the whole pipeline can run end to end with real
HTTP requests and without contacting anyone's server.
"""

from __future__ import annotations

import hashlib
import random
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VARIETY_A, VARIETY_B = "prs", "pes"

# The word lists are wrapped by hand; the formatter would put one word per line.
# fmt: off

# Shared vocabulary: common words of written Persian, used by both varieties.
COMMON_WORDS = [
    "و", "در", "به", "از", "که", "این", "را", "با", "است", "برای", "آن", "یک", "خود", "تا", "بر",
    "هم", "نیز", "گفت", "وی", "شد", "ما", "اما", "یا", "شده", "باید", "هر", "آنها", "بود", "او",
    "دیگر", "دو", "مورد", "شود", "کند", "وجود", "بین", "پیش", "سال", "روز", "کشور", "مردم",
    "دولت", "کار", "زمان", "جهان", "شهر", "خانه", "آب", "راه", "دست", "زندگی", "تاریخ", "زبان",
    "کتاب", "خبر", "گزارش", "مسئولان", "مقام\u200cها", "اعلام", "امروز", "دیروز", "هفته", "ماه",
    "گذشته", "آینده", "بزرگ", "کوچک", "جدید", "مهم", "اقتصادی", "سیاسی", "فرهنگی", "اجتماعی",
    "بین\u200cالمللی", "سازمان", "ملل", "امنیت", "صلح", "جنگ", "انتخابات", "نمایندگان", "مجلس",
    "وزیر", "رئیس", "جمهور", "برنامه", "پروژه", "توسعه", "آموزش", "بازار", "قیمت", "تجارت",
    "صادرات", "واردات", "کشاورزی", "باران", "زمستان", "تابستان", "کودکان", "زنان", "جوانان",
    "خانواده", "ورزش", "تیم", "بازی", "مسابقه", "پیروزی", "می\u200cشود", "می\u200cکند",
    "می\u200cگوید", "خواهد", "کرده\u200cاند", "شده\u200cاست", "بیش", "کمتر", "بسیار", "همه",
    "چند", "پس", "حال", "اگر", "چون", "زیرا", "هنوز", "تنها", "نخست", "دوم", "سوم", "میلیون",
    "هزار", "درباره", "براساس", "توسط", "همچنین", "منطقه", "مرکز", "بخش", "نقش", "حضور", "ادامه",
    "آغاز", "پایان", "نتیجه", "دلیل", "موضوع", "مسئله", "افزایش", "کاهش", "تغییر", "شرایط",
    "وضعیت", "امکان", "نیاز", "استفاده", "تولید", "خدمات", "شرکت", "بانک", "سرمایه", "کارگران",
    "کارمندان", "دانش", "علم", "هنر", "فیلم", "موسیقی", "شعر", "نویسنده", "روزنامه", "رسانه",
    "تلویزیون", "اینترنت",
]

# Words used by one variety and not the other (same pairs as config/markers.toml).
MARKER_WORDS = {
    VARIETY_A: [
        "پوهنتون", "شفاخانه", "ولسوالی", "شاروالی", "ولایت", "طیاره", "موتر", "پولیس", "داکتر",
        "فیصد",
    ],
    VARIETY_B: [
        "دانشگاه", "بیمارستان", "شهرستان", "شهرداری", "استان", "هواپیما", "خودرو", "پلیس",
        "دکتر", "درصد",
    ],
}
# Words both varieties use, but one noticeably more often.
STYLE_WORDS = {
    VARIETY_A: ["می\u200cباشد", "نموده", "نمودند", "گردید", "ایشان", "جریان", "هکذا"],
    VARIETY_B: ["می\u200cباشند", "کرده", "کردند", "گشت", "آنان", "روند", "همین\u200cطور"],
}

# fmt: on

_ZIPF_WEIGHTS = [1.0 / (rank + 1) ** 0.9 for rank in range(len(COMMON_WORDS))]


@dataclass(frozen=True, slots=True)
class Style:
    """How a text source writes. ``favourites`` gives an outlet its own vocabulary bias."""

    variety: str
    marker_rate: float = 0.25  # chance that a sentence contains a marker word
    style_rate: float = 0.5  # chance that a sentence contains a style word
    cross_rate: float = 0.05  # chance of using the *other* variety's word (label noise)
    favourites: tuple[str, ...] = ()


def make_style(rng: random.Random, variety: str) -> Style:
    return Style(variety, favourites=tuple(rng.sample(COMMON_WORDS, 12)))


def _other(variety: str) -> str:
    return VARIETY_B if variety == VARIETY_A else VARIETY_A


def make_sentence(rng: random.Random, style: Style) -> str:
    """One sentence of 8 to 24 words ending in a full stop."""
    words = rng.choices(COMMON_WORDS, weights=_ZIPF_WEIGHTS, k=rng.randint(8, 24))
    for position in range(len(words)):
        if style.favourites and rng.random() < 0.08:
            words[position] = rng.choice(style.favourites)
    for table, rate in ((MARKER_WORDS, style.marker_rate), (STYLE_WORDS, style.style_rate)):
        if rng.random() < rate:
            variety = _other(style.variety) if rng.random() < style.cross_rate else style.variety
            words[rng.randrange(len(words))] = rng.choice(table[variety])
    if rng.random() < 0.15:
        words.insert(rng.randrange(len(words)), str(rng.randint(2, 2500)))
    return " ".join(words) + "."


def make_paragraphs(rng: random.Random, style: Style, n_paragraphs: int = 0) -> list[str]:
    count = n_paragraphs or rng.randint(4, 9)
    return [
        " ".join(make_sentence(rng, style) for _ in range(rng.randint(2, 5))) for _ in range(count)
    ]


def make_title(rng: random.Random, style: Style) -> str:
    return make_sentence(rng, style).rstrip(".")


def add_noise(rng: random.Random, text: str, rate: float = 0.3) -> str:
    """Spelling variation as found on real sites: Arabic yeh and kaf, Persian or
    Arabic digits, a stray vowel mark, tatweel, and invisible characters."""
    if rng.random() < rate:
        text = text.replace("ی", "ي")
    if rng.random() < rate:
        text = text.replace("ک", "ك")
    if rng.random() < rate:
        text = text.translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))
    elif rng.random() < rate:
        text = text.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
    if rng.random() < rate:
        text = text.replace("ا ", "ا\u064b ", 2).replace("ت", "تـ", 1)
    if rng.random() < rate:
        text = text.replace(" ", "\u00a0", 3).replace("م", "م\u200d", 1) + "\u200f"
    return text


def make_document(rng: random.Random, style: Style, noisy: bool = True) -> str:
    """Plain article text, paragraphs separated by newlines."""
    text = "\n".join(make_paragraphs(rng, style))
    return add_noise(rng, text) if noisy else text


def article_html(title: str, paragraphs: list[str], published: datetime | None) -> str:
    """A news page with the usual furniture around the article."""
    meta = (
        f'<meta property="article:published_time" content="{published.isoformat()}">'
        if published
        else ""
    )
    body = "\n".join(f"<p>{escape(paragraph)}</p>" for paragraph in paragraphs)
    return f"""<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8"><title>{escape(title)} | خبرگزاری</title>
<meta property="og:title" content="{escape(title)}">{meta}
<script>window.analytics = [];</script><style>body {{ margin: 0 }}</style></head>
<body>
<header><nav><a href="/">خانه</a> <a href="/politics">سیاست</a></nav></header>
<main><article class="story"><h1>{escape(title)}</h1>
<div class="body">
{body}
</div>
<aside>مطالب مرتبط: <a href="/a/1.html">بیشتر بخوانید</a></aside></article></main>
<footer><p>تمام حقوق محفوظ است.</p><form><input name="q"></form></footer>
<script>track();</script></body></html>"""


def rss_xml(site_title: str, items: list[tuple[str, str, datetime | None]]) -> str:
    """An RSS 2.0 feed. items are (absolute url, title, published)."""
    entries = []
    for url, title, published in items:
        date = f"<pubDate>{format_datetime(published)}</pubDate>" if published else ""
        entries.append(
            f"<item><title>{escape(title)}</title><link>{escape(url)}</link>"
            f"<guid>{escape(url)}</guid>{date}</item>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0"><channel>'
        f"<title>{escape(site_title)}</title>{''.join(entries)}</channel></rss>"
    )


@dataclass(slots=True)
class Page:
    body: bytes
    content_type: str = "text/html; charset=utf-8"


@dataclass(slots=True)
class Site:
    """A fake site: path -> page. ``hits`` counts requests per path while it is served."""

    name: str
    pages: dict[str, Page] = field(default_factory=dict)
    hits: dict[str, int] = field(default_factory=dict)
    # (path, title, published) of everything the RSS feed at /feed.xml lists. The feed
    # is rendered when the site is served, because entries need the server's address.
    feed_items: list[tuple[str, str, datetime | None]] = field(default_factory=list)

    def add(self, path: str, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        self.pages[path] = Page(text.encode("utf-8"), content_type)


def build_news_site(
    rng: random.Random,
    name: str,
    style: Style,
    n_articles: int,
    *,
    today: datetime,
    days: int = 30,
    shared: list[tuple[str, list[str]]] | None = None,
) -> Site:
    """A site with an RSS feed at /feed.xml, articles at /news/<n>.html and a robots.txt.

    Besides normal articles it contains the awkward cases a real crawl meets:
    one page that is too short, one in the wrong script, one that robots.txt
    forbids, and any ``shared`` articles (wire copy that other sites carry too).
    """
    site = Site(name)
    site.add("/robots.txt", "User-agent: *\nDisallow: /private/\n", "text/plain")

    def publish(path: str, title: str, paragraphs: list[str], age_days: float | None) -> None:
        published = today - timedelta(days=age_days) if age_days is not None else None
        noisy = [add_noise(rng, paragraph) for paragraph in paragraphs]
        site.add(path, article_html(title, noisy, published))
        site.feed_items.append((path, title, published))

    for number in range(1, n_articles + 1):
        publish(
            f"/news/{number}.html",
            make_title(rng, style),
            make_paragraphs(rng, style),
            rng.uniform(0, days),
        )
    for number, (title, paragraphs) in enumerate(shared or [], start=1):
        publish(f"/news/wire-{number}.html", title, paragraphs, rng.uniform(0, days))
    publish("/news/short.html", make_title(rng, style), ["خبر کوتاه."], 1.0)
    publish(
        "/news/english.html",
        "English page",
        ["This page is written in English and should be rejected by the script check. " * 6],
        1.0,
    )
    publish("/private/hidden.html", make_title(rng, style), make_paragraphs(rng, style), 1.0)
    return site


def build_headline_feed(
    rng: random.Random, name: str, style: Style, n_items: int, *, today: datetime
) -> Site:
    """A site that only offers an RSS feed of headlines (for the news job)."""
    site = Site(name)
    site.add("/robots.txt", "User-agent: *\nAllow: /\n", "text/plain")
    site.feed_items = [
        (f"/item/{number}", make_title(rng, style), today - timedelta(hours=number))
        for number in range(1, n_items + 1)
    ]
    return site


def _handler_for(site: Site) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            site.hits[path] = site.hits.get(path, 0) + 1
            page = site.pages.get(path)
            if page is None:
                self.send_error(404)
                return
            # A strong ETag from the content, so conditional requests can be exercised.
            etag = '"' + hashlib.sha256(page.body).hexdigest()[:16] + '"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", page.content_type)
            self.send_header("Content-Length", str(len(page.body)))
            self.send_header("ETag", etag)
            self.end_headers()
            self.wfile.write(page.body)

        def log_message(self, format: str, *args: object) -> None:
            pass  # keep test and demo output quiet

    return Handler


@contextmanager
def serve(sites: list[Site]) -> Iterator[dict[str, str]]:
    """Serve each site on its own localhost port. Yields {site name: base URL}.

    Separate ports make the sites separate hosts as far as the crawler is
    concerned: each gets its own robots.txt, delay and crawl thread.
    """
    servers: list[ThreadingHTTPServer] = []
    urls: dict[str, str] = {}
    try:
        for site in sites:
            server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(site))
            base = f"http://127.0.0.1:{server.server_address[1]}"
            if site.feed_items:
                absolute = [(base + path, title, when) for path, title, when in site.feed_items]
                site.add("/feed.xml", rss_xml(site.name, absolute), "application/rss+xml")
            threading.Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)
            urls[site.name] = base
        yield urls
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def fixed_today() -> datetime:
    """Midnight UTC today: a stable "now" for building dated demo content."""
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
