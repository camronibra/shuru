from __future__ import annotations

import gzip
from collections.abc import Callable

import httpx
import pytest

from lab.config import HttpConfig
from lab.ingest.fetch import Fetcher, FetchError, Throttle, decode_body
from lab.runs import Timings

HOST = "site.test"
HOSTS = {HOST}
PAGE = b"<html><body><p>hello</p></body></html>"
Handler = Callable[[httpx.Request], httpx.Response]


class FakeTime:
    """A clock that only moves when something sleeps, so tests take no real time."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 3))
        self.now += seconds


def html(body: bytes = PAGE, **headers: str) -> httpx.Response:
    return httpx.Response(200, content=body, headers={"content-type": "text/html", **headers})


def make_fetcher(
    handler: Handler, robots: str = "", time: FakeTime | None = None, **config: float | int
) -> tuple[Fetcher, list[str]]:
    """A Fetcher wired to a fake site. Returns it with the list of paths requested."""
    requested: list[str] = []

    def route(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=robots) if robots else httpx.Response(404)
        return handler(request)

    time = time or FakeTime()
    settings = HttpConfig(**{"default_delay_s": 0.0, **config})  # type: ignore[arg-type]
    client = httpx.Client(transport=httpx.MockTransport(route))
    return Fetcher(settings, client, clock=time.clock, sleep=time.sleep), requested


def test_plain_fetch_returns_body_and_validators() -> None:
    fetcher, requested = make_fetcher(lambda r: html(etag='"v1"'))
    result = fetcher.get(f"https://{HOST}/a", allowed_hosts=HOSTS)
    assert (result.status, result.body, result.etag) == (200, PAGE, '"v1"')
    assert requested == ["/robots.txt", "/a"]


def test_robots_disallow_means_no_request_is_made() -> None:
    fetcher, requested = make_fetcher(lambda r: html(), robots="User-agent: *\nDisallow: /no\n")
    with pytest.raises(FetchError) as error:
        fetcher.get(f"https://{HOST}/no/page", allowed_hosts=HOSTS)
    assert error.value.kind == "robots" and error.value.blocked
    assert requested == ["/robots.txt"]


def test_other_hosts_and_schemes_are_refused_without_a_request() -> None:
    fetcher, requested = make_fetcher(lambda r: html())
    for url, kind in (("https://elsewhere.test/a", "off_site"), ("ftp://site.test/a", "scheme")):
        with pytest.raises(FetchError) as error:
            fetcher.get(url, allowed_hosts=HOSTS)
        assert error.value.kind == kind
    assert requested == []


def test_redirect_within_the_site_is_followed_and_checked_again() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return html()

    fetcher, requested = make_fetcher(handler)
    assert fetcher.get(f"https://{HOST}/old", allowed_hosts=HOSTS).url == f"https://{HOST}/new"
    assert requested == ["/robots.txt", "/old", "/new"]


def test_redirect_to_another_site_or_a_forbidden_path_is_not_followed() -> None:
    targets = {"/away": "https://tracker.test/x", "/sneaky": "/no/secret"}
    fetcher, requested = make_fetcher(
        lambda r: httpx.Response(302, headers={"location": targets[r.url.path]}),
        robots="User-agent: *\nDisallow: /no\n",
    )
    with pytest.raises(FetchError) as error:
        fetcher.get(f"https://{HOST}/away", allowed_hosts=HOSTS)
    assert error.value.kind == "off_site"
    with pytest.raises(FetchError) as error:
        fetcher.get(f"https://{HOST}/sneaky", allowed_hosts=HOSTS)
    assert error.value.kind == "robots"
    assert "/no/secret" not in requested


def test_redirect_loops_end() -> None:
    fetcher, _ = make_fetcher(lambda r: httpx.Response(302, headers={"location": "/loop"}))
    with pytest.raises(FetchError) as error:
        fetcher.get(f"https://{HOST}/loop", allowed_hosts=HOSTS)
    assert error.value.kind == "redirects"


def test_oversized_bodies_are_cut_off() -> None:
    big = b"x" * 5000
    fetcher, _ = make_fetcher(lambda r: html(big), max_bytes=2048)
    with pytest.raises(FetchError) as error:
        fetcher.get(f"https://{HOST}/big", allowed_hosts=HOSTS)
    assert error.value.kind == "too_large"


def test_size_limit_applies_after_decompression() -> None:
    bomb = gzip.compress(b"0" * 200_000)  # about 200 bytes on the wire
    assert len(bomb) < 1024
    fetcher, _ = make_fetcher(lambda r: html(bomb, **{"content-encoding": "gzip"}), max_bytes=4096)
    with pytest.raises(FetchError) as error:
        fetcher.get(f"https://{HOST}/bomb", allowed_hosts=HOSTS)
    assert error.value.kind == "too_large"


def test_non_text_content_is_refused() -> None:
    fetcher, _ = make_fetcher(
        lambda r: httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})
    )
    with pytest.raises(FetchError) as error:
        fetcher.get(f"https://{HOST}/logo.png", allowed_hosts=HOSTS)
    assert error.value.kind == "content_type"


def test_conditional_request_sends_validators_and_understands_304() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(304)

    fetcher, _ = make_fetcher(handler)
    result = fetcher.get(
        f"https://{HOST}/feed", allowed_hosts=HOSTS, etag='"v1"', last_modified="Mon, 01 Jan"
    )
    assert result.not_modified and result.body == b""
    assert seen["if-none-match"] == '"v1"' and seen["if-modified-since"] == "Mon, 01 Jan"


def test_temporary_errors_are_retried_with_growing_pauses() -> None:
    answers = iter([httpx.Response(503), httpx.Response(502), html()])
    time = FakeTime()
    fetcher, _ = make_fetcher(lambda r: next(answers), time=time, default_delay_s=2.0)
    timings = Timings()
    assert fetcher.get(f"https://{HOST}/a", allowed_hosts=HOSTS, timings=timings).status == 200
    # The usual 2 s gap after robots.txt, then back-off pauses of 2 s and 4 s. A back-off
    # is never shorter than the site delay, so no extra gap is added on top of it.
    assert time.sleeps == [2.0, 2.0, 4.0]
    assert timings.seconds["wait"] == pytest.approx(8.0)


def test_retry_after_header_is_honoured_and_gives_up_eventually() -> None:
    time = FakeTime()
    fetcher, requested = make_fetcher(
        lambda r: httpx.Response(429, headers={"retry-after": "30"}), time=time, max_retries=1
    )
    with pytest.raises(FetchError) as error:
        fetcher.get(f"https://{HOST}/a", allowed_hosts=HOSTS)
    assert (error.value.kind, error.value.status) == ("http", 429)
    assert 30.0 in time.sleeps
    assert requested.count("/a") == 2


def test_not_found_is_not_retried() -> None:
    fetcher, requested = make_fetcher(lambda r: httpx.Response(404))
    with pytest.raises(FetchError) as error:
        fetcher.get(f"https://{HOST}/gone", allowed_hosts=HOSTS)
    assert error.value.status == 404
    assert requested.count("/gone") == 1


def test_network_errors_are_retried_then_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    fetcher, requested = make_fetcher(handler, max_retries=2)
    with pytest.raises(FetchError) as error:
        fetcher.get(f"https://{HOST}/a", allowed_hosts=HOSTS)
    assert (error.value.kind, error.value.detail) == ("network", "ConnectError")
    assert requested.count("/a") == 3


def test_a_site_that_keeps_failing_is_left_alone() -> None:
    fetcher, requested = make_fetcher(
        lambda r: httpx.Response(403), max_host_failures=3, max_retries=0
    )
    kinds = []
    for number in range(5):
        with pytest.raises(FetchError) as error:
            fetcher.get(f"https://{HOST}/p{number}", allowed_hosts=HOSTS)
        kinds.append(error.value.kind)
    assert kinds == ["http", "http", "http", "host_down", "host_down"]
    assert len(requested) == 1 + 3  # robots.txt, then three refusals; nothing after that


def test_requests_to_one_site_are_spaced_by_the_largest_delay() -> None:
    time = FakeTime()
    fetcher, _ = make_fetcher(
        lambda r: html(), robots="User-agent: *\nCrawl-delay: 5\n", time=time, default_delay_s=2.0
    )
    for number in range(3):
        fetcher.get(f"https://{HOST}/p{number}", allowed_hosts=HOSTS)
    # robots.txt itself, then: 2 s (default) before the first page, 5 s (robots) after that.
    assert time.sleeps == [2.0, 5.0, 5.0]
    fetcher.get(f"https://{HOST}/slow", allowed_hosts=HOSTS, min_delay_s=9.0)
    assert time.sleeps[-1] == 5.0  # the pause already owed; the next gap would be 9 s


def test_throttle_tracks_each_host_separately() -> None:
    time = FakeTime()
    throttle = Throttle(time.clock, time.sleep)
    assert throttle.wait("a", 3.0) == 0.0
    assert throttle.wait("b", 3.0) == 0.0  # another site does not wait for the first
    assert throttle.wait("a", 3.0) == 3.0
    time.now += 10
    assert throttle.wait("a", 3.0) == 0.0  # enough time has passed by itself


def test_decode_body_finds_the_charset() -> None:
    # windows-1256 is the old Windows Arabic code page. It has no Persian yeh,
    # which is one reason old Iranian sites used Arabic yeh (see normalise.py).
    text = "سلام كتاب"
    assert decode_body(text.encode("utf-8"), "text/html; charset=UTF-8") == text
    assert decode_body(text.encode("windows-1256"), "text/html; charset=windows-1256") == text
    meta = b'<html><head><meta charset="windows-1256"></head>' + text.encode("windows-1256")
    assert decode_body(meta, "text/html").endswith(text)
    assert decode_body(b"caf\xe9", "text/html; charset=no-such-charset") == "caf�"
