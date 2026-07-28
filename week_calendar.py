"""Lay out a deck of events as a minimal 7-day calendar.

Pure layout math — no Flask, no LLM — so it can be tested on its own. The output
is render-ready: every chip carries an absolute pixel `top`, and the template
just drops them into positioned divs.

Three lanes exist because real scraped data breaks a naive time grid:
  * events with a date but no clock time are stored as midnight, so they would
    all pile up at the top of a column — they go in an all-day band instead;
  * events at the same time would overlap — they are pushed down rather than
    split into side-by-side columns, since there are no time labels to read
    position against and the chips need their full width for a title;
  * events with no date, or beyond the week, have nowhere to sit — they go in an
    overflow tray.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

from event_store import has_time_component, parse_iso_datetime

DAYS = 7

# Geometry. Python owns these so collision math and rendering can't disagree;
# the template applies the computed heights as inline styles.
BASE_GRID_HEIGHT = 440
CHIP_HEIGHT = 22
CHIP_GAP = 2

# The grid never spans a full 24 hours — an empty 2am-7am band just wastes
# vertical space. The window is fitted to the deck, then clamped to these.
EARLIEST_HOUR = 7
LATEST_HOUR = 24
MIN_WINDOW_HOURS = 6

# Statuses a chip can carry. "declined" never reaches this module — those events
# are filtered out upstream so their chips disappear from the calendar.
STATUS_PENDING = "pending"
STATUS_CURRENT = "current"
STATUS_ACCEPTED = "accepted"


def build_week(
    events: Sequence[dict],
    statuses: Dict[str, str],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Lay out `events` across the 7 days starting today.

    `statuses` maps uid -> status. An event whose uid is absent from the map is
    treated as declined and omitted entirely.
    """
    now = now or datetime.now()
    today = now.date()
    window_end = today + timedelta(days=DAYS)

    # day index -> chips, plus the leftovers that don't fit the week at all.
    timed: Dict[int, List[dict]] = {index: [] for index in range(DAYS)}
    allday: Dict[int, List[dict]] = {index: [] for index in range(DAYS)}
    overflow: List[dict] = []

    for event in events:
        uid = event.get("uid")
        status = statuses.get(uid or "")
        if not status:
            continue  # declined

        chip = {
            "uid": uid,
            "summary": event.get("summary") or "Untitled event",
            "status": status,
            "top": 0,
        }

        start = parse_iso_datetime(event.get("dtstart"))
        if start is None or not (today <= start.date() < window_end):
            overflow.append(chip)
            continue

        index = (start.date() - today).days
        if has_time_component(event.get("dtstart")):
            chip["_minutes"] = start.hour * 60 + start.minute
            timed[index].append(chip)
        else:
            allday[index].append(chip)

    start_hour, end_hour = _window_for(timed)
    grid_height = BASE_GRID_HEIGHT

    for index in range(DAYS):
        chips = sorted(timed[index], key=lambda c: c["_minutes"])
        _position(chips, start_hour=start_hour, end_hour=end_hour)
        timed[index] = chips
        if chips:
            grid_height = max(grid_height, chips[-1]["top"] + CHIP_HEIGHT + CHIP_GAP)

    allday_height = max(
        (len(allday[index]) for index in range(DAYS)),
        default=0,
    ) * (CHIP_HEIGHT + CHIP_GAP)

    days = []
    for index in range(DAYS):
        date = today + timedelta(days=index)
        days.append(
            {
                "dow": date.strftime("%a").upper(),
                "num": str(date.day),
                "is_today": index == 0,
                "allday": allday[index],
                "timed": timed[index],
            }
        )

    return {
        "days": days,
        "overflow": overflow,
        "grid_height": grid_height,
        "allday_height": allday_height,
        "total": sum(len(d["allday"]) + len(d["timed"]) for d in days) + len(overflow),
    }


def _window_for(timed: Dict[int, List[dict]]) -> tuple:
    """Fit the vertical window to the deck, padded an hour and clamped."""
    minutes = [chip["_minutes"] for chips in timed.values() for chip in chips]
    if not minutes:
        return EARLIEST_HOUR, EARLIEST_HOUR + MIN_WINDOW_HOURS

    start_hour = max(EARLIEST_HOUR, min(minutes) // 60 - 1)
    end_hour = min(LATEST_HOUR, max(minutes) // 60 + 2)

    # Guarantee a usable span even when every event shares one hour.
    if end_hour - start_hour < MIN_WINDOW_HOURS:
        end_hour = min(LATEST_HOUR, start_hour + MIN_WINDOW_HOURS)
        start_hour = max(EARLIEST_HOUR, end_hour - MIN_WINDOW_HOURS)
    return start_hour, end_hour


def _position(chips: List[dict], *, start_hour: int, end_hour: int) -> None:
    """Assign each chip a pixel `top`, pushing later chips down to avoid overlap.

    Chips must already be sorted by start time. Pushing down rather than
    splitting the column keeps the full width available for the chip's title;
    with no time labels on the grid, the small vertical drift is unreadable
    anyway.
    """
    span = max((end_hour - start_hour) * 60, 1)
    usable = max(BASE_GRID_HEIGHT - CHIP_HEIGHT, 1)

    previous_bottom = -CHIP_GAP
    for chip in chips:
        offset = chip.pop("_minutes") - start_hour * 60
        ratio = min(max(offset / span, 0.0), 1.0)
        natural = int(ratio * usable)
        top = max(natural, previous_bottom + CHIP_GAP)
        chip["top"] = top
        previous_bottom = top + CHIP_HEIGHT
