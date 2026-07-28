"""Convert Instagram captions into ICS-shaped calendar events via OpenRouter."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from event_store import CalendarEvent, normalize_category, parse_iso_datetime
from instagram_scraper import PostCaption

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional until installed
    OpenAI = None  # type: ignore[misc, assignment]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"

# Shared by the extraction prompt and the backfill prompt so the two can never
# drift into writing descriptions in different voices. Spliced in via a marker
# rather than an f-string because both prompts contain literal JSON braces.
DESCRIPTION_RULES = """- REQUIRED for every event. Never null, never an empty string.
- One or two sentences, roughly 15-40 words. State plainly what the event is, who is hosting it, and what happens there.
- Factual and neutral. No hype, no second-person marketing ("you'll love…"), no exclamation marks.
- Ground every detail in the source text. Do not invent activities, prices, guests, or details that are not stated.
- The title, date, and time are displayed next to the description, so repeating them wastes it. Never open with the event's name, and never state the date.
  BAD:  "The 2026 Renegade Craft Summer Fair is hosted at Fort Mason Center on August 1-2, showcasing handmade crafts."
  GOOD: "A weekend market at Fort Mason Center with handmade goods from hundreds of independent artists and makers."
- Never comment on what the source does not say. Write only what is known, even if that makes the description shorter.
  BAD:  "An anticipated meteor shower, though specific details are not provided."
  GOOD: "An annual meteor shower, best viewed away from city lights in the hours before dawn."
- If a listing row gives only a title, an organizer, and a location, write those as a plain sentence rather than giving up. Example: "Martial arts practice session hosted by UMass Wushu at Campus Center 163C."
- Never return just an organizer or venue name on its own — write a sentence."""

_SYSTEM_PROMPT = """You extract calendar events from a block of text — either an
Instagram post caption or text scraped from a web page (e.g. an events listing).

Return ONLY valid JSON with this shape:
{
  "events": [
    {
      "summary": "short event title",
      "dtstart": "ISO-8601 datetime or date, or null if unknown",
      "dtend": "ISO-8601 datetime or date, or null",
      "location": "place or null",
      "description": "1-2 sentence factual summary (REQUIRED, never null)",
      "category": "professional | fun | community"
    }
  ]
}

Description rules:
__DESCRIPTION_RULES__

Category rules:
- Every event MUST get exactly one category. Pick the closest fit; never leave it null.
- "professional" — career fairs, info sessions, recruiting, networking, resume/technical workshops, industry talks, academic conferences.
- "fun" — parties, socials, mixers, concerts, movie nights, food events, games, sports, trips.
- "community" — volunteering, service projects, fundraisers, cultural and heritage celebrations, advocacy, religious gatherings, general body meetings.
- When an event fits two, choose the one matching its primary purpose (e.g. a networking mixer with free food is "professional").

Rules:
- Only create events when the text clearly describes something schedulable (show, launch, meetup, deadline, performance, etc.).
- The text may describe zero, one, or several distinct events (e.g. a page listing many events) — return one entry per distinct event.
- Past-tense thank-you / recap posts with no future date are NOT events — return {"events": []}.
- If there is no event, return {"events": []}.
- Prefer timezone-aware ISO-8601 when possible; otherwise use the date/time as stated.
- Do not invent events that are not supported by the text.

Date / year rules (critical):
- The user message includes "Source timestamp" (when the source was published/captured, if known) and "Reference date".
- If an event gives a month/day but NO year (e.g. "Monday, May 4th"), choose the year so the event date is near the reference date — usually the same year, or the next year only if that month/day has already clearly passed relative to the reference by more than ~2 months AND the text implies a future event.
- Never invent a year far from the reference date (e.g. do not use 2024 when the reference date is in 2025).
- If only a weekday + time is given with no calendar date, set dtstart/dtend to null rather than guessing.
"""

_DESCRIBE_PROMPT = """You write short factual descriptions for calendar events that
were already extracted from a source text.

You are given the original source text and a numbered list of events pulled from it.
Write one description per event, using only what the source text says about that event.

Return ONLY valid JSON with this shape:
{
  "descriptions": {
    "1": "description for event 1",
    "2": "description for event 2"
  }
}

Rules:
- Return exactly one entry per numbered event, keyed by its number as a string.
- If the source text says little about an event, fall back to its title, organizer, and location.
__DESCRIPTION_RULES__
"""

SYSTEM_PROMPT = _SYSTEM_PROMPT.replace("__DESCRIPTION_RULES__", DESCRIPTION_RULES)
DESCRIBE_PROMPT = _DESCRIBE_PROMPT.replace("__DESCRIPTION_RULES__", DESCRIPTION_RULES)

# Events per backfill call. Keeps the JSON response small enough to stay reliable.
DESCRIBE_BATCH_SIZE = 12


class CaptionToEventConverter:
    """Use an OpenRouter LLM to turn captions into CalendarEvent objects."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.base_url = base_url or os.environ.get(
            "OPENROUTER_BASE_URL",
            OPENROUTER_BASE_URL,
        )
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Export it before converting captions."
            )
        if OpenAI is None:
            raise ImportError(
                "openai package is not installed. Run: pip install openai"
            )

        # OpenRouter is OpenAI-compatible; optional headers help with ranking/limits.
        default_headers = {}
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        title = os.environ.get("OPENROUTER_APP_TITLE", "instagram-caption-scraper")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-Title"] = title

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=default_headers or None,
        )

    def convert_captions(
        self,
        captions: List[PostCaption],
        *,
        source_username: Optional[str] = None,
    ) -> List[CalendarEvent]:
        events: List[CalendarEvent] = []
        for caption in captions:
            events.extend(
                self.convert_caption(caption, source_username=source_username)
            )
        return events

    def convert_caption(
        self,
        caption: PostCaption,
        *,
        source_username: Optional[str] = None,
    ) -> List[CalendarEvent]:
        text = (caption.caption or "").strip()
        if not text:
            return []

        reference = _reference_datetime(caption.timestamp)
        user_prompt = (
            f"Source: {source_username or 'unknown'}\n"
            f"Source URL: {caption.url}\n"
            f"Source timestamp: {caption.timestamp or 'unknown'}\n"
            f"Reference date: {reference.date().isoformat()} "
            f"(use this to resolve missing years)\n"
            f"Text:\n{text}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - surface LLM failures cleanly
            logging.error("LLM conversion failed for %s: %s", caption.url, exc)
            return []

        content = response.choices[0].message.content or "{}"
        return self._parse_events(
            content,
            caption=caption,
            source_username=source_username,
            reference=reference,
        )

    def describe_events(
        self,
        events: List[CalendarEvent],
        *,
        max_calls: Optional[int] = None,
    ) -> Dict[str, str]:
        """Write descriptions for events that lack one. Returns {uid: description}.

        Events extracted from the same chunk share an identical source_caption, so
        they are grouped and that text is sent once for the whole group instead of
        once per event — the source text dwarfs everything else in the payload.
        """
        groups: "OrderedDict[str, List[CalendarEvent]]" = OrderedDict()
        for event in events:
            if not event.uid:
                continue
            groups.setdefault((event.source_caption or "").strip(), []).append(event)

        descriptions: Dict[str, str] = {}
        calls = 0
        for source_text, group in groups.items():
            for start in range(0, len(group), DESCRIBE_BATCH_SIZE):
                if max_calls is not None and calls >= max_calls:
                    return descriptions
                batch = group[start : start + DESCRIBE_BATCH_SIZE]
                descriptions.update(self._describe_batch(batch, source_text=source_text))
                calls += 1
        return descriptions

    def describe_call_count(self, events: List[CalendarEvent]) -> int:
        """How many LLM calls describe_events would make for these events."""
        groups: Dict[str, int] = {}
        for event in events:
            if not event.uid:
                continue
            key = (event.source_caption or "").strip()
            groups[key] = groups.get(key, 0) + 1
        return sum(
            (count + DESCRIBE_BATCH_SIZE - 1) // DESCRIBE_BATCH_SIZE
            for count in groups.values()
        )

    def _describe_batch(
        self,
        events: List[CalendarEvent],
        *,
        source_text: str,
    ) -> Dict[str, str]:
        listing = "\n".join(
            f"{index}. title: {event.summary}"
            f" | location: {event.location or 'unknown'}"
            f" | source: {event.source_username or 'unknown'}"
            for index, event in enumerate(events, start=1)
        )
        user_prompt = (
            f"Events needing a description:\n{listing}\n\n"
            + (
                f"Source text they were extracted from:\n{source_text}"
                if source_text
                else "No source text is available — write from the fields above alone."
            )
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": DESCRIBE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - surface LLM failures cleanly
            logging.error("Description backfill failed: %s", exc)
            return {}

        content = response.choices[0].message.content or "{}"
        payload = _loads_json(content, context="description backfill")
        if payload is None:
            return {}

        raw = payload.get("descriptions")
        if not isinstance(raw, dict):
            return {}

        results: Dict[str, str] = {}
        for index, event in enumerate(events, start=1):
            text = _clean_description(raw.get(str(index)))
            if text and event.uid:
                results[event.uid] = text
        return results

    def _parse_events(
        self,
        content: str,
        *,
        caption: PostCaption,
        source_username: Optional[str],
        reference: datetime,
    ) -> List[CalendarEvent]:
        payload = _loads_json(content, context=caption.url)
        if payload is None:
            return []

        raw_events = payload.get("events") or []
        if not isinstance(raw_events, list):
            return []

        events: List[CalendarEvent] = []
        for item in raw_events:
            if not isinstance(item, dict):
                continue
            summary = (item.get("summary") or "").strip()
            if not summary:
                continue

            dtstart = _normalize_event_datetime(
                _nullable_str(item.get("dtstart")),
                reference=reference,
            )
            dtend = _normalize_event_datetime(
                _nullable_str(item.get("dtend")),
                reference=reference,
                prefer_after=dtstart,
            )

            events.append(
                CalendarEvent(
                    summary=summary,
                    dtstart=dtstart,
                    dtend=dtend,
                    location=_nullable_str(item.get("location")),
                    description=_clean_description(item.get("description")),
                    uid=str(uuid.uuid4()),
                    source_username=source_username,
                    source_url=caption.url,
                    source_caption=caption.caption,
                    category=normalize_category(item.get("category")),
                )
            )
        return events


def _loads_json(content: str, *, context: str) -> Optional[dict]:
    """Parse an LLM JSON reply, tolerating prose wrapped around the object."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            logging.warning("LLM returned non-JSON for %s", context)
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            logging.warning("Could not parse LLM JSON for %s", context)
            return None
    return payload if isinstance(payload, dict) else None


# The prompt forbids narrating missing information, but a small model still
# tacks on the occasional "…, though specific details are not provided". Only
# matches a trailing comma-delimited clause, so removing it leaves a sentence.
_FILLER_CLAUSE = re.compile(
    r",\s*(?:though|although|but|however|with)\b[^,.]*?\b"
    r"(?:not\s+(?:provided|specified|mentioned|available|listed|given)"
    r"|unavailable|unspecified|no\s+(?:specific|further)\b[^,.]*)"
    r"[^.]*",
    re.IGNORECASE,
)


def _clean_description(value: object) -> Optional[str]:
    """Coerce to a description, dropping trailing 'details not provided' filler."""
    text = _nullable_str(value)
    if not text:
        return None
    cleaned = _FILLER_CLAUSE.sub("", text).strip()
    if not cleaned or len(cleaned) < 20:
        return text  # Sanitizing gutted it — keep what the model wrote.
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _nullable_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text


def _reference_datetime(timestamp: Optional[str]) -> datetime:
    parsed = _parse_iso(timestamp)
    if parsed is not None:
        return parsed
    return datetime.now(timezone.utc)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    return parse_iso_datetime(value)


def _normalize_event_datetime(
    value: Optional[str],
    *,
    reference: datetime,
    prefer_after: Optional[str] = None,
) -> Optional[str]:
    """Snap missing/wrong years toward the post timestamp."""
    if not value:
        return None

    parsed = _parse_iso(value)
    if parsed is None:
        # Date-only YYYY-MM-DD
        try:
            parsed = datetime.strptime(value[:10], "%Y-%m-%d").replace(
                tzinfo=reference.tzinfo or timezone.utc
            )
        except ValueError:
            return value

    adjusted = _snap_year_to_reference(parsed, reference)

    after = _parse_iso(prefer_after)
    if after is not None and adjusted < after:
        # Keep end after start; if year snap broke ordering, bump a year.
        candidate = adjusted.replace(year=adjusted.year + 1)
        if candidate >= after:
            adjusted = candidate

    return adjusted.isoformat()


def _snap_year_to_reference(event_dt: datetime, reference: datetime) -> datetime:
    """
    Rewrite the event onto the year nearest the post timestamp when the LLM
    guessed a distant year (common when captions omit the year).
    """
    ref = reference.astimezone(event_dt.tzinfo or timezone.utc)

    candidates: List[datetime] = [event_dt]
    for year in (ref.year - 1, ref.year, ref.year + 1):
        try:
            candidates.append(event_dt.replace(year=year))
        except ValueError:
            # e.g. Feb 29 on a non-leap year
            continue

    def score(candidate: datetime) -> Tuple[int, int]:
        delta = candidate - ref
        days = abs(delta.days)
        # Slight preference for dates on/after the post (upcoming events).
        future_bias = 0 if delta.days >= -14 else 45
        return (days + future_bias, days)

    return min(candidates, key=score)
