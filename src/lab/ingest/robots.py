"""robots.txt: parse the rules and decide whether a URL may be fetched.

Follows RFC 9309 (the Robots Exclusion Protocol):

- Rules are grouped under ``User-agent`` lines. The group naming our product token
  ("shuru") applies; if there is none, the ``*`` group applies.
- A URL is matched against every Allow and Disallow pattern. The longest matching
  pattern wins, and Allow wins a tie. ``*`` matches any characters, ``$`` anchors
  the end.
- No matching rule means allowed.

``Crawl-delay`` is not part of the RFC but many sites set it, so it is honoured.

The standard library's ``urllib.robotparser`` is not used because it does not
understand ``*`` and ``$`` in paths, so it would fetch URLs a site meant to exclude.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

# Only the start of an oversized robots.txt is parsed (the RFC asks for at least 500 KiB).
MAX_ROBOTS_BYTES = 512 * 1024
MAX_CRAWL_DELAY_S = 3600.0


@dataclass(frozen=True, slots=True)
class _Rule:
    allow: bool
    pattern: str
    regex: re.Pattern[str]


def _compile(pattern: str) -> re.Pattern[str]:
    """Turn a robots path pattern into a regex anchored at the start of the path."""
    anchored = pattern.endswith("$")
    if anchored:
        pattern = pattern[:-1]
    body = ".*".join(re.escape(part) for part in pattern.split("*"))
    return re.compile(body + ("$" if anchored else ""))


def product_token(user_agent: str) -> str:
    """The name robots.txt files address us by: "shuru/0.1 (+url)" -> "shuru"."""
    return user_agent.split("/", 1)[0].split(" ", 1)[0].strip().lower()


class RobotsRules:
    """The rules of one site, as they apply to one user agent."""

    def __init__(self, rules: list[_Rule], crawl_delay: float | None = None) -> None:
        self._rules = rules
        self.crawl_delay = crawl_delay

    @classmethod
    def allow_all(cls) -> RobotsRules:
        return cls([])

    @classmethod
    def disallow_all(cls) -> RobotsRules:
        return cls([_Rule(False, "/", _compile("/"))])

    @classmethod
    def parse(cls, text: str, user_agent: str) -> RobotsRules:
        """Parse robots.txt and keep the rules meant for user_agent."""
        token = product_token(user_agent)
        # group name -> (rules, crawl delay). Several groups may share one rule block.
        groups: dict[str, tuple[list[_Rule], list[float]]] = {}
        current: list[str] = []
        reading_agents = False
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "user-agent":
                if not reading_agents:
                    current = []
                reading_agents = True
                name = value.lower()
                current.append(name)
                groups.setdefault(name, ([], []))
                continue
            if key not in ("allow", "disallow", "crawl-delay"):
                continue  # Sitemap: and unknown lines do not end a group
            reading_agents = False
            if key in ("allow", "disallow"):
                if not value:  # "Disallow:" with nothing after it restricts nothing
                    continue
                if not value.startswith(("/", "*")):
                    value = "/" + value
                rule = _Rule(key == "allow", value, _compile(value))
                for name in current:
                    groups[name][0].append(rule)
            else:
                try:
                    delay = float(value)
                except ValueError:
                    continue
                if delay >= 0:
                    for name in current:
                        groups[name][1].append(min(delay, MAX_CRAWL_DELAY_S))
        rules, delays = groups.get(token) or groups.get("*") or ([], [])
        return cls(rules, max(delays) if delays else None)

    def allowed(self, url: str) -> bool:
        """True if the rules let us fetch url."""
        parts = urlsplit(url)
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        if path == "/robots.txt":
            return True
        best_length = -1
        verdict = True
        for rule in self._rules:
            if rule.regex.match(path):
                length = len(rule.pattern)
                if length > best_length or (length == best_length and rule.allow):
                    best_length, verdict = length, rule.allow
        return verdict


class RobotsCache:
    """Fetches each site's robots.txt once per run and remembers the verdict.

    ``fetch`` takes the robots.txt URL and returns (HTTP status, body). It raises
    on network failure. What the status means:

    - 2xx: parse the rules.
    - 4xx (the file does not exist): everything is allowed. 401 and 403 are the
      exception: the site is refusing us, so nothing is fetched.
    - 5xx or a network error: nothing is fetched this run. When a site is in
      trouble, the polite assumption is "not now".
    """

    def __init__(self, user_agent: str, fetch: Callable[[str], tuple[int, bytes]]) -> None:
        self._user_agent = user_agent
        self._fetch = fetch
        self._rules: dict[str, RobotsRules] = {}
        self._lock = threading.Lock()

    def rules_for(self, url: str) -> RobotsRules:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc.lower()}"
        with self._lock:
            cached = self._rules.get(origin)
        if cached is not None:
            return cached
        rules = self._load(origin + "/robots.txt")
        with self._lock:
            self._rules[origin] = rules
        return rules

    def _load(self, robots_url: str) -> RobotsRules:
        try:
            status, body = self._fetch(robots_url)
        except Exception:
            return RobotsRules.disallow_all()
        if 200 <= status < 300:
            text = body[:MAX_ROBOTS_BYTES].decode("utf-8", errors="replace")
            return RobotsRules.parse(text, self._user_agent)
        if status in (401, 403) or status >= 500:
            return RobotsRules.disallow_all()
        return RobotsRules.allow_all()
