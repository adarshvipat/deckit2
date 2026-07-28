"""Central store and ICS-shaped calendar event models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Iterable, List, Optional

# The interest buckets an event can be tagged with. The LLM prompt, the client
# survey, and the deck filter all read from here so they can never drift apart.
EVENT_CATEGORIES = ("professional", "fun", "community")

CATEGORY_LABELS = {
    "professional": "Professional",
    "fun": "Fun",
    "community": "Community",
}

CATEGORY_BLURBS = {
    "professional": "Career fairs, info sessions, networking, workshops, talks",
    "fun": "Parties, socials, concerts, food, games, sports",
    "community": "Volunteering, service, cultural events, advocacy",
}


def normalize_category(value: object) -> Optional[str]:
    """Coerce an arbitrary value to a known category, or None."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in EVENT_CATEGORIES else None


@dataclass
class CalendarEvent:
    """ICS-oriented event fields extracted from Instagram captions."""

    summary: str
    dtstart: Optional[str] = None  # ISO-8601 or YYYYMMDD / YYYYMMDDTHHMMSS
    dtend: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    uid: Optional[str] = None
    source_username: Optional[str] = None
    source_url: Optional[str] = None
    source_caption: Optional[str] = None
    category: Optional[str] = None  # one of EVENT_CATEGORIES

    def to_dict(self) -> dict:
        return asdict(self)

    def to_ics_vevent_lines(self) -> List[str]:
        """Render this event as ICS VEVENT lines (without wrapping)."""
        lines = ["BEGIN:VEVENT"]
        if self.uid:
            lines.append(f"UID:{self.uid}")
        if self.dtstart:
            lines.append(f"DTSTART:{_ics_datetime(self.dtstart)}")
        if self.dtend:
            lines.append(f"DTEND:{_ics_datetime(self.dtend)}")
        lines.append(f"SUMMARY:{_ics_escape(self.summary)}")
        if self.location:
            lines.append(f"LOCATION:{_ics_escape(self.location)}")
        if self.description:
            lines.append(f"DESCRIPTION:{_ics_escape(self.description)}")
        if self.category:
            lines.append(f"CATEGORIES:{self.category.upper()}")
        if self.source_url:
            lines.append(f"URL:{self.source_url}")
        lines.append("END:VEVENT")
        return lines


def _ics_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
    )


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string into an aware datetime, or None if unparseable.

    Naive values are assumed to be UTC. Shared by the LLM date-snapping logic
    and by the catalog's upcoming-events filter.
    """
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        # Date-only fallback: leading YYYY-MM-DD of a longer/odd string.
        try:
            parsed = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _ordinal(day: int) -> str:
    if 11 <= day <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def has_time_component(value: Optional[str]) -> bool:
    """True when the value carries a meaningful clock time, not just a date.

    Two cases mean "no time". A bare `YYYY-MM-DD` is obvious. The subtler one is
    exact midnight: `_normalize_event_datetime` runs every extracted date through
    `.isoformat()`, so a source that gave a date and no time is *stored* as
    `T00:00:00+00:00`. Treating that as a real 12:00AM start would both mislabel
    cards and pile every such event at the top of a calendar grid. Events that
    genuinely begin at midnight are rare enough that this is the right trade.
    """
    if not value:
        return False
    if ":" not in value.strip()[10:]:
        return False
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return False
    return not (parsed.hour == 0 and parsed.minute == 0)


def format_event_datetime(value: Optional[str], fallback: str = "") -> str:
    """Render a stored event time for humans, e.g. "July 5th at 11:00PM".

    The wall-clock time is shown exactly as stored and is never shifted between
    timezones: captions say "doors at 7pm" meaning local time, and the extractor
    labels that +00:00, so converting would move every event by several hours.
    The year is shown only when it is not the current one, and date-only values
    render without a time rather than inventing midnight.
    """
    if not value:
        return fallback
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return value.strip() or fallback

    day = f"{parsed.strftime('%B')} {_ordinal(parsed.day)}"
    if parsed.year != datetime.now(timezone.utc).year:
        day = f"{day}, {parsed.year}"

    if not has_time_component(value):
        return day

    hour = parsed.hour % 12 or 12
    meridiem = "AM" if parsed.hour < 12 else "PM"
    return f"{day} at {hour}:{parsed.minute:02d}{meridiem}"


def _ics_datetime(value: str) -> str:
    """Normalize common ISO strings into a compact ICS datetime if possible."""
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return (
            value.strip()
            .replace("-", "")
            .replace(":", "")
            .replace(" ", "T")
        )

    if parsed.tzinfo is None:
        return parsed.strftime("%Y%m%dT%H%M%S")

    utc = parsed.astimezone(timezone.utc)
    return utc.strftime("%Y%m%dT%H%M%SZ")


class EventStore:
    """Thread-safe in-memory list of calendar events for later use."""

    def __init__(self) -> None:
        self._events: List[CalendarEvent] = []
        self._lock = Lock()

    def add(self, event: CalendarEvent) -> None:
        with self._lock:
            self._events.append(event)

    def extend(self, events: Iterable[CalendarEvent]) -> None:
        with self._lock:
            self._events.extend(events)

    def all(self) -> List[CalendarEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def to_dicts(self) -> List[dict]:
        return [event.to_dict() for event in self.all()]


# Module-level singleton — import this for a shared central array.
EVENT_STORE = EventStore()
