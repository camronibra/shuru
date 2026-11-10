from __future__ import annotations

import gzip
import re

import pytest

from lab.config import CleanConfig
from lab.corpus.clean import Prepared, Rejected, prepare, truncate
from lab.ingest.discover import (
    DiscoverError,
    canonical_url,
    gunzip_if_needed,
    parse_feed,
    parse_listing,
    parse_sitemap,
)
from lab.ingest.extract import extract
from lab.text.dedup import MinHasher

PERSIAN = "این یک جمله آزمایشی به زبان فارسی است که برای آزمودن نوشته شده است. "

PAGE = """<!doctype html><html><head>
<title>عنوان صفحه | سایت</title>
<meta property="og:title" content="عنوان خبر">
<meta property="article:published_time" content="2026-03-01T10:00:00+04:30">
<script>document.write("<p>injected</p>")</script><style>p { color: red }</style>
</head><body>
<header><nav><a href="/">خانه</a></nav></header>
<article class="teaser"><p>پیش‌نمایش کوتاه</p></article>
<article class="story">
  <h1>عنوان خبر</h1>
  <p>پاراگراف <b>اول</b> با   فاصله‌های  زیاد.</p>
  <script>alert("x")</script>
  <figure><img src="x.jpg"><figcaption>زیرنویس عکس</figcaption></figure>
  <ul><li><p>مورد یک</p></li><li>مورد دو</li></ul>
  <p>خط اول<br>خط دوم</p>
  <aside>مطالب مرتبط</aside>
  <p>پاراگراف <b>اول</b> با   فاصله‌های  زیاد.</p>
</article>
<footer><p>تمام حقوق محفوظ است.</p></footer>
</body></html>"""


# --- extract ----------------------------------------------------------------------------


def test_generic_extraction_takes_the_largest_article_and_drops_furniture() -> None:
    result = extract(PAGE)
    assert result.title == "عنوان خبر"
    assert result.published_at == "2026-03-01T05:30:00Z"  # converted to UTC
    assert result.text.split("\n") == [
        "پاراگراف اول با فاصله‌های زیاد.",  # inline tags merged, spaces squashed
        "مورد یک",
        "مورد دو",
        "خط اول خط دوم",
    ]  # the duplicate paragraph appears once; teaser, aside, figure, footer are gone
    for junk in ("injected", "alert", "زیرنویس", "مطالب مرتبط", "حقوق", "پیش‌نمایش"):
        assert junk not in result.text


def test_selector_extraction() -> None:
    assert extract(PAGE, "article.teaser").text == "پیش‌نمایش کوتاه"
    assert extract(PAGE, "div.no-such-thing").text == ""


def test_broken_html_does_not_raise() -> None:
    result = extract("<html><body><p>بسته نشده<div><p>دوم")
    assert "بسته نشده" in result.text and "دوم" in result.text
    assert extract("").text == ""
    assert extract("\x00\xff<<<>>>").title == ""


def test_title_falls_back_to_h1_then_title_tag() -> None:
    assert extract("<h1>سرخط</h1><p>متن</p>").title == "سرخط"
    assert extract("<title>فقط عنوان</title>").title == "فقط عنوان"


def test_unreadable_dates_are_dropped() -> None:
    # 1403 is a Solar Hijri year written in an ISO field: refused rather than guessed.
    page = '<meta property="article:published_time" content="1403-05-01"><p>x</p>'
    assert extract(page).published_at is None
    page = '<time datetime="2026-02-03">۳ فوریه</time><p>x</p>'
    assert extract(page).published_at == "2026-02-03T00:00:00Z"


# --- discover ---------------------------------------------------------------------------


def test_canonical_url() -> None:
    assert (
        canonical_url("/news/1?utm_source=x&id=5&fbclid=y#comments", "https://Site.TEST/a/")
        == "https://site.test/news/1?id=5"
    )
    assert canonical_url("https://site.test") == "https://site.test/"
    for bad in ("javascript:alert(1)", "mailto:a@b.c", "", "https://" + "a" * 3000):
        assert canonical_url(bad) is None


RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>  \xd8\xae\xd8\xa8\xd8\xb1  </title><link>https://site.test/n/1?utm_medium=rss</link>
<guid>n-1</guid><pubDate>Sun, 01 Mar 2026 10:00:00 +0430</pubDate></item>
<item><title>no link</title><guid>https://site.test/n/2</guid></item>
<item><title>bad</title><link>javascript:void(0)</link></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0" encoding="utf-8"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>a</title><link rel="enclosure" href="/a.mp3"/><link href="/a"/>
<id>tag:a</id><updated>2026-03-02T00:00:00Z</updated></entry></feed>"""


def test_rss_and_atom_feeds() -> None:
    items = parse_feed(RSS, "https://site.test/feed")
    assert [(i.url, i.title, i.guid, i.published_at) for i in items] == [
        ("https://site.test/n/1", "خبر", "n-1", "2026-03-01T05:30:00Z"),
        ("https://site.test/n/2", "no link", "https://site.test/n/2", None),
    ]
    (entry,) = parse_feed(ATOM, "https://site.test/feed.atom")
    assert (entry.url, entry.published_at) == ("https://site.test/a", "2026-03-02T00:00:00Z")


def test_hostile_xml_is_refused() -> None:
    billion_laughs = b"""<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">
    <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]>
    <rss><channel><item><title>&lol2;</title></item></channel></rss>"""
    with pytest.raises(DiscoverError):
        parse_feed(billion_laughs, "https://site.test/")
    with pytest.raises(DiscoverError):
        parse_feed(b"<html>not a feed", "https://site.test/")


def test_sitemap_and_sitemap_index() -> None:
    urlset = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
      xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
      <url><loc>https://site.test/n/1</loc><lastmod>2026-03-01</lastmod></url>
      <url><loc>https://site.test/n/2</loc><news:news><news:publication_date>2026-03-02T08:00:00Z
      </news:publication_date><news:title>title two</news:title></news:news></url>
    </urlset>"""
    pages, children = parse_sitemap(urlset, "https://site.test/sitemap.xml")
    assert [(p.url, p.title, p.published_at) for p in pages] == [
        ("https://site.test/n/1", "", "2026-03-01T00:00:00Z"),
        ("https://site.test/n/2", "title two", "2026-03-02T08:00:00Z"),
    ]
    assert children == []
    index = b"""<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://site.test/s1.xml</loc><lastmod>2026-01-01</lastmod></sitemap>
    </sitemapindex>"""
    pages, children = parse_sitemap(index, "https://site.test/sitemap.xml")
    assert pages == [] and [c.url for c in children] == ["https://site.test/s1.xml"]


def test_gzipped_sitemaps_are_capped_after_decompression() -> None:
    assert gunzip_if_needed(gzip.compress(b"<urlset/>"), 1000) == b"<urlset/>"
    assert gunzip_if_needed(b"<urlset/>", 1000) == b"<urlset/>"
    with pytest.raises(DiscoverError, match="too large"):
        gunzip_if_needed(gzip.compress(b"x" * 10_000), 1000)


def test_listing_keeps_matching_links_once_in_page_order() -> None:
    html = """<a href="/news/2">b</a><a href="/about">x</a><a href="/news/1#c">a</a>
              <a href="/news/2?utm_source=tw">b again</a><a>no href</a>"""
    found = parse_listing(html, "https://site.test/", re.compile(r"/news/\d+$"))
    assert [item.url for item in found] == ["https://site.test/news/2", "https://site.test/news/1"]


# --- clean ------------------------------------------------------------------------------


CLEAN = CleanConfig(min_chars=100, max_chars=1000, min_script_ratio=0.6)
HASHER = MinHasher()


def test_prepare_accepts_persian_text_and_hashes_folded_form() -> None:
    first = prepare(PERSIAN * 3, CLEAN, HASHER)
    # Same text typed on an Arabic keyboard, with Persian digits elsewhere: same hash.
    second = prepare((PERSIAN * 3).replace("ی", "ي").replace("ک", "ك"), CLEAN, HASHER)
    assert isinstance(first, Prepared) and isinstance(second, Prepared)
    assert first.text_hash == second.text_hash
    assert first.signature == second.signature
    assert first.n_tokens == 3 * len(PERSIAN.split())
    assert len(first.bands) == HASHER.bands


def test_prepare_rejects_short_and_wrong_script_text() -> None:
    assert prepare("کوتاه", CLEAN, HASHER) == Rejected("too_short")
    english = "This is plainly written in English and nothing else. " * 5
    assert prepare(english, CLEAN, HASHER) == Rejected("wrong_script")
    mixed = PERSIAN * 3 + "Some English quoted in the article. "
    assert isinstance(prepare(mixed, CLEAN, HASHER), Prepared)


def test_long_text_is_cut_at_a_paragraph_boundary() -> None:
    text = "\n".join(f"پاراگراف شماره {n} " * 10 for n in range(50))
    cut = truncate(text, 500)
    assert len(cut) <= 500 and "\n" in cut and text.startswith(cut)
    assert truncate("کوتاه", 500) == "کوتاه"
