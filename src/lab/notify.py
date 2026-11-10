"""Phone alerts through ntfy (https://ntfy.sh, or a self-hosted server).

An alert is one HTTP POST. With no topic configured, alerts are only logged. Sending
never raises: a broken alert channel must not hide the failure it was reporting.
"""

from __future__ import annotations

import httpx

from lab.config import NotifyConfig
from lab.logs import get_logger

log = get_logger(__name__)

MAX_TITLE_CHARS = 120
MAX_MESSAGE_CHARS = 1000


def _printable(text: str, limit: int) -> str:
    """Drop control characters and cap the length. Messages may quote untrusted text."""
    cleaned = "".join(ch if ch.isprintable() or ch == "\n" else " " for ch in text)
    return cleaned[:limit]


class Notifier:
    """Sends alerts to one ntfy topic."""

    def __init__(
        self,
        server: str,
        topic: str,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        self._server = server.rstrip("/")
        self._topic = topic
        self._client = client
        self._timeout_s = timeout_s

    @classmethod
    def from_config(cls, config: NotifyConfig) -> Notifier:
        return cls(config.ntfy_server, config.topic)

    @property
    def enabled(self) -> bool:
        return bool(self._topic)

    def send(self, title: str, message: str, *, priority: int = 3) -> bool:
        """Send one alert. priority runs from 1 (silent) to 5 (urgent). True if delivered."""
        title = _printable(title, MAX_TITLE_CHARS)
        message = _printable(message, MAX_MESSAGE_CHARS)
        if not self.enabled:
            log.warning("alert (not sent, NTFY_TOPIC is not set): %s: %s", title, message)
            return False
        payload = {
            "topic": self._topic,
            "title": title,
            "message": message,
            "priority": min(max(priority, 1), 5),
        }
        try:
            if self._client is not None:
                response = self._client.post(self._server, json=payload, timeout=self._timeout_s)
            else:
                response = httpx.post(self._server, json=payload, timeout=self._timeout_s)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.error("alert could not be sent: %s (alert was: %s)", type(exc).__name__, title)
            return False
        log.info("alert sent: %s", title)
        return True
