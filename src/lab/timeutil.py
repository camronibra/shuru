"""Timestamps. Everything stored is UTC, so dates line up across sources and machines."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
# Dates outside this range are treated as unreadable. A year like 1403 is a Solar
# Hijri date (the calendar used in Iran and Afghanistan) written in an ISO-looking
# field; guessing would put the document in the wrong split.
_MIN_YEAR = 1990


def utc_now() -> str:
    """Current time as stored in the database: ISO 8601, second precision, trailing Z."""
    return datetime.now(UTC).strftime(_FORMAT)


def to_utc_iso(moment: datetime) -> str:
    """Format any datetime the way the database stores it. Naive values are taken as UTC."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime(_FORMAT)


def hours_ago(hours: float) -> str:
    return to_utc_iso(datetime.now(UTC) - timedelta(hours=hours))


def days_ago(days: int) -> str:
    """The calendar day (YYYY-MM-DD, UTC) this many days before today."""
    return (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")


def parse_timestamp(value: str | None) -> str | None:
    """Read a date from a feed or a page and return it in storage format, or None.

    Accepts ISO 8601 ("2026-03-01T10:00:00+04:30", "2026-03-01") and the RFC 822 form
    used by RSS ("Sun, 01 Mar 2026 10:00:00 +0430"). Anything unreadable, implausibly
    old or more than a day in the future gives None: a missing date is better than a
    wrong one, because dates decide the train/test split.
    """
    if not value:
        return None
    value = value.strip()
    if not value or len(value) > 64:
        return None
    moment: datetime | None = None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            moment = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    try:
        moment = moment.astimezone(UTC)
    except (OverflowError, ValueError):
        return None
    if moment.year < _MIN_YEAR or moment > datetime.now(UTC) + timedelta(days=1):
        return None
    return moment.strftime(_FORMAT)
