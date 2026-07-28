"""Convert Instagram captions into ICS-shaped calendar events via OpenRouter."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from event_store import CalendarEvent
from instagram_scraper import PostCaption

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional until installed
    OpenAI = None  # type: ignore[misc, assignment]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"

SYSTEM_PROMPT = """You extract calendar events from a block of text — either an
Instagram post caption or text scraped from a web page (e.g. an events listing).

Return ONLY valid JSON with this shape:
{
  "events": [
    {
      "summary": "short event title",
      "dtstart": "ISO-8601 datetime or date, or null if unknown",
      "dtend": "ISO-8601 datetime or date, or null",
      "location": "place or null",
      "description": "brief description or null"
    }
  ]
}

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

    def _parse_events(
        self,
        content: str,
        *,
        caption: PostCaption,
        source_username: Optional[str],
        reference: datetime,
    ) -> List[CalendarEvent]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                logging.warning("LLM returned non-JSON for %s", caption.url)
                return []
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                logging.warning("Could not parse LLM JSON for %s", caption.url)
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
                    description=_nullable_str(item.get("description")),
                    uid=str(uuid.uuid4()),
                    source_username=source_username,
                    source_url=caption.url,
                    source_caption=caption.caption,
                )
            )
        return events


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
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


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
