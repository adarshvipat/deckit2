"""Summary statistics for the stored event catalog.

Pure arithmetic — no Flask, no I/O — so it can be tested directly. Feeds the
stats band at the top of the admin catalog page.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from event_store import (
    CATEGORY_LABELS,
    EVENT_CATEGORIES,
    has_time_component,
    parse_iso_datetime,
)


def _pct(count: int, total: int) -> int:
    return round(100 * count / total) if total else 0


def compute_stats(
    events: Sequence[dict],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Summarize the catalog: size, freshness, data gaps, and category mix."""
    now = now or datetime.now(timezone.utc)
    total = len(events)

    past = 0
    upcoming = 0
    no_date = 0
    no_time = 0
    no_description = 0
    counts: Dict[str, int] = {key: 0 for key in EVENT_CATEGORIES}
    uncategorized = 0

    for event in events:
        start = parse_iso_datetime(event.get("dtstart"))
        if start is None:
            no_date += 1
        else:
            if start < now:
                past += 1
            else:
                upcoming += 1
            # Only meaningful for events that have a date in the first place.
            if not has_time_component(event.get("dtstart")):
                no_time += 1

        if not (event.get("description") or "").strip():
            no_description += 1

        category = event.get("category")
        if category in counts:
            counts[category] += 1
        else:
            uncategorized += 1

    categories: List[Dict[str, Any]] = [
        {
            "key": key,
            "label": CATEGORY_LABELS[key],
            "count": counts[key],
            "pct": _pct(counts[key], total),
        }
        for key in EVENT_CATEGORIES
        if counts[key]
    ]
    if uncategorized:
        categories.append(
            {
                "key": "none",
                "label": "Untagged",
                "count": uncategorized,
                "pct": _pct(uncategorized, total),
            }
        )
    categories.sort(key=lambda item: item["count"], reverse=True)

    return {
        "total": total,
        "upcoming": upcoming,
        "past": past,
        "no_date": no_date,
        "no_date_pct": _pct(no_date, total),
        "no_time": no_time,
        "no_time_pct": _pct(no_time, total),
        "no_description": no_description,
        "no_description_pct": _pct(no_description, total),
        "categories": categories,
    }
