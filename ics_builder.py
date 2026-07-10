"""Build downloadable ICS calendars from accepted CalendarEvent objects."""

from __future__ import annotations

from typing import Iterable, List

from event_store import CalendarEvent


def build_ics(events: Iterable[CalendarEvent], calendar_name: str = "Instagram Events") -> str:
    """Return a full VCALENDAR document as a string."""
    lines: List[str] = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Instagram Event Review//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{_escape(calendar_name)}",
    ]
    for event in events:
        lines.extend(event.to_ics_vevent_lines())
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def _escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
    )
