from __future__ import annotations

from pathlib import Path

import pytest

from lab.ingest.robots import RobotsCache, RobotsRules, product_token

FIXTURES = Path(__file__).parent / "fixtures" / "robots"
UA = "shuru/0.1 (+https://example.org/about)"


def rules(name: str, user_agent: str = UA) -> RobotsRules:
    return RobotsRules.parse((FIXTURES / name).read_text(encoding="utf-8"), user_agent)


def test_product_token() -> None:
    assert product_token(UA) == "shuru"
    assert product_token("Mozilla 5.0") == "mozilla"


@pytest.mark.parametrize(
    ("path", "allowed"),
    [
        ("/", True),
        ("/news/1.html", True),
        ("/search", False),
        ("/search/results?q=x", False),  # a rule is a prefix
        ("/news/1.html?print=1", False),  # * matches any characters
        ("/files/report.pdf", False),  # $ anchors the end
        ("/files/report.pdf?download=1", True),  # ... so this is not matched
        ("/private/notes.html", False),
        ("/private/press/release.html", True),  # the longer (more specific) rule wins
        ("/robots.txt", True),  # always fetchable
    ],
)
def test_wildcards_and_longest_match(path: str, allowed: bool) -> None:
    parsed = rules("wildcards.txt")
    assert parsed.allowed(f"https://example.org{path}") is allowed
    assert parsed.crawl_delay == 7


def test_our_own_group_replaces_the_wildcard_group() -> None:
    ours = rules("groups.txt")
    assert not ours.allowed("https://example.org/drafts/x")
    assert ours.allowed("https://example.org/admin/")  # the * group no longer applies to us
    assert ours.crawl_delay == 20

    someone_else = rules("groups.txt", "curl/8.0")
    assert not someone_else.allowed("https://example.org/admin/")
    assert someone_else.allowed("https://example.org/drafts/x")
    assert someone_else.crawl_delay is None

    for bot in ("otherbot/1.0", "ThirdBot"):  # two user-agent lines share one rule block
        assert not rules("groups.txt", bot).allowed("https://example.org/anything")


def test_edge_cases() -> None:
    assert RobotsRules.parse("", UA).allowed("https://example.org/x")
    assert RobotsRules.parse("User-agent: *\nDisallow:\n", UA).allowed("https://example.org/x")
    assert not RobotsRules.parse("User-agent: *\nDisallow: /\n", UA).allowed("https://e.org/x")
    # Allow wins a tie in length.
    tie = RobotsRules.parse("User-agent: *\nDisallow: /a\nAllow: /a\n", UA)
    assert tie.allowed("https://example.org/a")
    # Garbage lines and bad delays are skipped, not fatal.
    messy = RobotsRules.parse("<html>\nUser-agent: *\nCrawl-delay: soon\nDisallow /x\n", UA)
    assert messy.allowed("https://example.org/x") and messy.crawl_delay is None


@pytest.mark.parametrize(
    ("status", "allowed"),
    [
        (200, False),
        (404, True),
        (410, True),
        (401, False),
        (403, False),
        (500, False),
        (503, False),
    ],
)
def test_what_each_http_status_means(status: int, allowed: bool) -> None:
    cache = RobotsCache(UA, lambda url: (status, b"User-agent: *\nDisallow: /\n"))
    assert (
        cache.rules_for("https://example.org/page").allowed("https://example.org/page") is allowed
    )


def test_network_failure_means_do_not_fetch_and_result_is_cached() -> None:
    calls: list[str] = []

    def failing(url: str) -> tuple[int, bytes]:
        calls.append(url)
        raise OSError("connection refused")

    cache = RobotsCache(UA, failing)
    for _ in range(3):
        assert not cache.rules_for("https://Example.org/a").allowed("https://example.org/a")
    assert calls == ["https://example.org/robots.txt"]  # asked once per site, not per page
