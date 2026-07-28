"""Durable catalog of every event the admin has scraped.

Unlike EVENT_STORE (a per-run buffer that gets cleared), the catalog accumulates
across scrapes and survives restarts by writing a JSON file to data/events.json.
Client decks are assembled by querying this.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Sequence

from event_store import (
    EVENT_CATEGORIES,
    CalendarEvent,
    normalize_category,
    parse_iso_datetime,
)

CATALOG_PATH = Path(__file__).resolve().parent / "data" / "events.json"

# Undated and recently-past events stay in a deck for this long; an event that
# started yesterday is usually still worth showing.
_GRACE = timedelta(hours=12)

_WHITESPACE = re.compile(r"\s+")

_EVENT_FIELDS = set(CalendarEvent.__dataclass_fields__)


def _dedupe_key(event: CalendarEvent) -> str:
    """Identity of an event across re-scrapes.

    `uid` is a fresh uuid4 on every extraction, so it cannot be used here — the
    same post scraped twice would otherwise be stored twice.
    """
    summary = _WHITESPACE.sub(" ", (event.summary or "").strip().lower())
    day = (event.dtstart or "")[:10]
    return f"{summary}|{day}"


class EventCatalog:
    """Thread-safe, JSON-backed store of categorized events."""

    def __init__(self, path: Path = CATALOG_PATH) -> None:
        self._path = path
        self._lock = Lock()
        self._events: List[CalendarEvent] = []
        self._sources: List[Dict[str, Any]] = []
        self._loaded = False

    # ---- persistence -----------------------------------------------------

    def _load(self) -> None:
        """Read the JSON file into memory. Caller must hold the lock."""
        if self._loaded:
            return
        self._loaded = True
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logging.error("Could not read event catalog at %s: %s", self._path, exc)
            return

        for item in payload.get("events") or []:
            if not isinstance(item, dict):
                continue
            # Ignore unknown keys so an older/newer file never crashes startup.
            fields = {k: v for k, v in item.items() if k in _EVENT_FIELDS}
            if not fields.get("summary"):
                continue
            self._events.append(CalendarEvent(**fields))

        sources = payload.get("sources")
        if isinstance(sources, list):
            self._sources = [s for s in sources if isinstance(s, dict)]

    def _save(self) -> None:
        """Atomically rewrite the JSON file. Caller must hold the lock."""
        payload = {
            "version": 1,
            "events": [event.to_dict() for event in self._events],
            "sources": self._sources,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
        except OSError as exc:
            logging.error("Could not write event catalog to %s: %s", self._path, exc)

    # ---- writes ----------------------------------------------------------

    def add_all(
        self,
        events: Iterable[CalendarEvent],
        *,
        source_label: str,
        kind: str,
    ) -> int:
        """Store new events, skipping ones already in the catalog.

        Returns the number actually added.
        """
        events = list(events)
        with self._lock:
            self._load()
            seen = {_dedupe_key(existing) for existing in self._events}
            added = 0
            for event in events:
                key = _dedupe_key(event)
                if key in seen:
                    continue
                seen.add(key)
                event.category = normalize_category(event.category)
                self._events.append(event)
                added += 1

            self._touch_source(source_label, kind, added)
            self._save()
            return added

    def _touch_source(self, label: str, kind: str, added: int) -> None:
        """Record/refresh a scrape source. Caller must hold the lock."""
        now = datetime.now(timezone.utc).isoformat()
        for source in self._sources:
            if source.get("label") == label and source.get("kind") == kind:
                source["last_scraped"] = now
                source["event_count"] = int(source.get("event_count") or 0) + added
                return
        self._sources.append(
            {
                "label": label,
                "kind": kind,
                "last_scraped": now,
                "event_count": added,
            }
        )

    def remove(self, uid: str) -> bool:
        with self._lock:
            self._load()
            before = len(self._events)
            self._events = [e for e in self._events if e.uid != uid]
            removed = len(self._events) != before
            if removed:
                self._save()
            return removed

    def update_descriptions(self, updates: Dict[str, str]) -> int:
        """Set description by uid, only where one is missing. Returns count updated."""
        if not updates:
            return 0
        with self._lock:
            self._load()
            changed = 0
            for event in self._events:
                if event.description or not event.uid:
                    continue
                text = (updates.get(event.uid) or "").strip()
                if text:
                    event.description = text
                    changed += 1
            if changed:
                self._save()
            return changed

    def clear(self) -> None:
        with self._lock:
            self._load()
            self._events = []
            self._sources = []
            self._save()

    # ---- reads -----------------------------------------------------------

    def all(self) -> List[CalendarEvent]:
        with self._lock:
            self._load()
            return list(self._events)

    def missing_descriptions(self) -> List[CalendarEvent]:
        """Stored events that still have no description."""
        return [event for event in self.all() if not (event.description or "").strip()]

    def sources(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._load()
            return [dict(source) for source in self._sources]

    def upcoming(
        self,
        *,
        categories: Optional[Sequence[str]] = None,
        now: Optional[datetime] = None,
    ) -> List[CalendarEvent]:
        """Events matching the given categories, soonest first.

        Undated events are kept (an event with no parseable start is still real)
        and sorted to the end. Events with no category match every selection, so
        a mis-tagged event is never invisible to clients.
        """
        wanted = set(categories) if categories else set(EVENT_CATEGORIES)
        cutoff = (now or datetime.now(timezone.utc)) - _GRACE

        matches: List[CalendarEvent] = []
        for event in self.all():
            if event.category is not None and event.category not in wanted:
                continue
            start = parse_iso_datetime(event.dtstart)
            if start is not None and start < cutoff:
                continue
            matches.append(event)

        def sort_key(event: CalendarEvent):
            start = parse_iso_datetime(event.dtstart)
            # Undated events sort last, then alphabetically for stability.
            return (start is None, start or datetime.max.replace(tzinfo=timezone.utc), event.summary or "")

        matches.sort(key=sort_key)
        return matches

    def __len__(self) -> int:
        with self._lock:
            self._load()
            return len(self._events)


# Module-level singleton — import this for the shared catalog.
CATALOG = EventCatalog()
