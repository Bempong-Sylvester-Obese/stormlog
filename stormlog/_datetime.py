"""Shared datetime normalization helpers for internal artifact queries."""

from datetime import datetime, timezone


def parse_datetime(value: str | None) -> datetime | None:
    """Parse an ISO-8601 value and normalize it to UTC."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def datetime_to_ns(value: datetime | None) -> int | None:
    """Convert a datetime to Unix-epoch nanoseconds."""
    if value is None:
        return None
    return int(value.timestamp() * 1_000_000_000)
