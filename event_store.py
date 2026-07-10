"""Central store and ICS-shaped calendar event models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Iterable, List, Optional


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


def _ics_datetime(value: str) -> str:
    """Normalize common ISO strings into a compact ICS datetime if possible."""
    from datetime import datetime, timezone

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
