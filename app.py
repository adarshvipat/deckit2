#!/usr/bin/env python3
"""Flask app with two panels.

Admin (/admin):  scrape Instagram profiles and event pages → LLM extracts and
                 categorizes events → everything is stored in the catalog.
Client (/):      a short interest survey → a deck is assembled from the catalog
                 → yes/no swipe → download ICS.
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from caption_to_events import CaptionToEventConverter
from catalog_stats import compute_stats
from event_catalog import CATALOG
from event_store import (
    CATEGORY_BLURBS,
    CATEGORY_LABELS,
    EVENT_CATEGORIES,
    CalendarEvent,
    format_event_datetime,
    normalize_category,
    parse_iso_datetime,
)
from ics_builder import build_ics
from instagram_scraper import InstagramProfileParser, InstagramScraper, ScraperConfig
from url_scraper import UrlEventScraper, UrlScraperConfig, normalize_url
from week_calendar import (
    STATUS_ACCEPTED,
    STATUS_CURRENT,
    STATUS_PENDING,
    build_week,
)

load_dotenv(Path(__file__).resolve().parent / ".env")

app = Flask(__name__)
app.secret_key = "instagram-event-review-dev-key"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# Templates render raw ISO strings through this, e.g. "July 5th at 11:00PM".
app.jinja_env.filters["pretty_date"] = format_event_datetime

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Sort sentinel so undated events never compare against a datetime.
_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)

# In-memory scrape jobs for progress UI (single-process Flask).
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()

# Deck/review state lives server-side, keyed by a small id stored in the
# browser's session cookie — events carry their full source text and easily
# blow past the ~4KB signed-cookie session limit if stored there directly.
_REVIEW_SESSIONS: Dict[str, Dict[str, Any]] = {}
_REVIEW_LOCK = threading.Lock()

_EMPTY_REVIEW_STATE: Dict[str, Any] = {
    "pending_events": [],
    "accepted_events": [],
    "review_index": 0,
    "categories": [],
    "deck_id": "",
}


def _get_review_state() -> Dict[str, Any]:
    review_id = session.get("review_id")
    with _REVIEW_LOCK:
        state = _REVIEW_SESSIONS.get(review_id) if review_id else None
        return state if state is not None else dict(_EMPTY_REVIEW_STATE)


def _update_review_state(**fields: Any) -> None:
    review_id = session.get("review_id")
    if not review_id:
        return
    with _REVIEW_LOCK:
        state = _REVIEW_SESSIONS.get(review_id)
        if state is not None:
            state.update(fields)


def _set_pending(events: List[CalendarEvent], *, categories: List[str]) -> None:
    review_id = uuid.uuid4().hex
    with _REVIEW_LOCK:
        _REVIEW_SESSIONS[review_id] = {
            "pending_events": [event.to_dict() for event in events],
            "accepted_events": [],
            "review_index": 0,
            "categories": list(categories),
            # Per-deck ICS filename so two clients never overwrite each other.
            "deck_id": review_id,
        }
    session["review_id"] = review_id


def _update_job(job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job.update(fields)


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _new_job(label: str) -> str:
    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Queued…",
            "error": None,
            "added": 0,
            "failed": 0,
            "problems": [],
            "label": label,
        }
    return job_id


# ---------------------------------------------------------------------------
# Admin panel — feed sources in, events land in the catalog
# ---------------------------------------------------------------------------


def _parse_source_lines(raw: str) -> List[str]:
    """One source per line; blanks and # comments ignored, order preserved."""
    seen = set()
    out: List[str] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def _run_batch_job(
    job_id: str,
    usernames: List[str],
    urls: List[str],
    count: int,
    max_chunks: int,
) -> None:
    """Scrape every source in one pass, then convert and store the results.

    All Instagram profiles share a single browser session and all URLs share
    another, because starting Chrome is the slowest part of a scrape — doing it
    once per source would dominate the runtime of a multi-source batch.
    """
    problems: List[str] = []
    # (label, kind, captions) for everything that scraped successfully.
    harvested: List[tuple] = []

    try:
        total_sources = len(usernames) + len(urls)
        _update_job(job_id, status="running", progress=4, message="Starting up…")

        done = 0
        # 8% -> 45% covers scraping; the rest belongs to the LLM pass. Starts at
        # 8 so the "opening the browser" step never reads ahead of the first
        # source and makes the bar jump backwards.
        def scrape_progress(finished: int) -> int:
            if not total_sources:
                return 45
            return 8 + int(37 * finished / total_sources)

        if usernames:
            _update_job(job_id, progress=8, message="Opening the browser…")
            try:
                with InstagramScraper(
                    ScraperConfig.from_env(count=count, headless=True)
                ) as scraper:
                    for username in usernames:
                        _update_job(
                            job_id,
                            progress=scrape_progress(done),
                            message=f"Scraping @{username} ({done + 1} of {total_sources})",
                        )
                        # scrape_profiles never raises per profile; it reports
                        # failures on the result instead.
                        for result in scraper.scrape_profiles([username]):
                            if result.succeeded:
                                harvested.append((username, "instagram", result.posts))
                            else:
                                problems.append(f"@{username}: {result.error}")
                        done += 1
            except Exception as exc:  # noqa: BLE001 - browser-level failure
                logging.error("Instagram batch failed:\n%s", traceback.format_exc())
                problems.append(f"Instagram scraping failed: {exc}")
                done = len(usernames)

        if urls:
            _update_job(
                job_id,
                progress=scrape_progress(done),
                message="Opening the browser…",
            )
            try:
                with UrlEventScraper(UrlScraperConfig(max_chunks=max_chunks)) as scraper:
                    for url in urls:
                        _update_job(
                            job_id,
                            progress=scrape_progress(done),
                            message=f"Scraping {url} ({done + 1} of {total_sources})",
                        )
                        try:
                            result = scraper.scrape(url)
                        except Exception as exc:  # noqa: BLE001 - one bad page
                            problems.append(f"{url}: {exc}")
                            done += 1
                            continue
                        if result.succeeded:
                            harvested.append((result.label, "url", result.chunks))
                        else:
                            problems.append(f"{url}: {result.error}")
                        done += 1
            except Exception as exc:  # noqa: BLE001 - browser-level failure
                logging.error("URL batch failed:\n%s", traceback.format_exc())
                problems.append(f"Page scraping failed: {exc}")

        if not harvested:
            _update_job(
                job_id,
                status="error",
                progress=100,
                message="Nothing scraped",
                failed=len(problems),
                problems=problems,
                error="; ".join(problems) or "No sources produced any text.",
            )
            return

        _update_job(job_id, progress=48, message="Reading captions…")

        converter = CaptionToEventConverter()
        total_captions = sum(len(captions) for _, _, captions in harvested) or 1
        processed = 0
        added_total = 0

        for label, kind, captions in harvested:
            events: List[CalendarEvent] = []
            for caption in captions:
                processed += 1
                _update_job(
                    job_id,
                    # 48% -> 92% across every caption from every source.
                    progress=min(48 + int(44 * processed / total_captions), 92),
                    message=f"Processing with AI ({label})",
                )
                events.extend(
                    converter.convert_caption(caption, source_username=label)
                )

            if events:
                added_total += CATALOG.add_all(events, source_label=label, kind=kind)
            else:
                problems.append(f"{label}: no calendar events found.")

        _update_job(job_id, progress=96, message="Saving to the catalog…")

        _update_job(
            job_id,
            status="done",
            progress=100,
            message="Saved to the catalog",
            added=added_total,
            failed=len(problems),
            problems=problems,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        logging.error("Pipeline failed:\n%s", traceback.format_exc())
        _update_job(
            job_id,
            status="error",
            progress=100,
            message="Something went wrong",
            error=f"Pipeline failed: {exc}",
        )


def _render_feed(error: Optional[str] = None):
    return render_template("admin.html", error=error, active_page="feed")


def _render_catalog(error: Optional[str] = None):
    events = CATALOG.all()

    def sort_key(event: CalendarEvent):
        start = parse_iso_datetime(event.dtstart)
        # Undated events sort last.
        return (start is None, start or _FAR_FUTURE, event.summary or "")

    events.sort(key=sort_key)
    sources = sorted(
        CATALOG.sources(),
        key=lambda s: s.get("last_scraped") or "",
        reverse=True,
    )
    event_dicts = [event.to_dict() for event in events]
    return render_template(
        "catalog.html",
        error=error,
        active_page="catalog",
        events=event_dicts,
        sources=sources,
        stats=compute_stats(event_dicts),
        missing_count=len(CATALOG.missing_descriptions()),
        category_labels=CATEGORY_LABELS,
    )


@app.route("/admin", methods=["GET"])
def admin():
    return _render_feed()


@app.route("/admin/catalog", methods=["GET"])
def catalog():
    return _render_catalog()


@app.route("/admin/scrape", methods=["POST"])
def scrape():
    """Kick off one job covering every profile and URL that was entered."""
    count_raw = (request.form.get("count") or "5").strip()
    sections_raw = (request.form.get("sections") or "6").strip()
    try:
        count = max(1, int(count_raw))
    except ValueError:
        count = 5
    try:
        max_chunks = max(1, int(sections_raw))
    except ValueError:
        max_chunks = 6

    usernames: List[str] = []
    urls: List[str] = []
    bad: List[str] = []

    for line in _parse_source_lines(request.form.get("profiles")):
        try:
            usernames.append(InstagramProfileParser.parse(line))
        except ValueError as exc:
            bad.append(str(exc))

    for line in _parse_source_lines(request.form.get("urls")):
        try:
            urls.append(normalize_url(line))
        except ValueError as exc:
            bad.append(str(exc))

    if bad:
        return _render_feed(error=" · ".join(bad))
    if not usernames and not urls:
        return _render_feed(
            error="Add at least one Instagram profile or page URL to scrape."
        )

    # Dedupe while preserving entry order.
    usernames = list(dict.fromkeys(usernames))
    urls = list(dict.fromkeys(urls))

    parts = []
    if usernames:
        parts.append(f"{len(usernames)} profile{'s' if len(usernames) != 1 else ''}")
    if urls:
        parts.append(f"{len(urls)} page{'s' if len(urls) != 1 else ''}")
    job_id = _new_job(" · ".join(parts))

    thread = threading.Thread(
        target=_run_batch_job,
        args=(job_id, usernames, urls, count, max_chunks),
        daemon=True,
    )
    thread.start()
    return redirect(url_for("loading", job_id=job_id))


@app.route("/admin/loading/<job_id>", methods=["GET"])
def loading(job_id: str):
    job = _get_job(job_id)
    if not job:
        return _render_feed(error="That scrape job was not found. Try again.")
    return render_template(
        "loading.html",
        job_id=job_id,
        label=job.get("label", ""),
    )


@app.route("/api/job/<job_id>", methods=["GET"])
def job_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        return jsonify({"status": "missing", "error": "Job not found"}), 404
    return jsonify(
        {
            "status": job.get("status"),
            "progress": job.get("progress", 0),
            "message": job.get("message", ""),
            "error": job.get("error"),
            "label": job.get("label"),
            "added": job.get("added", 0),
            "failed": job.get("failed", 0),
            "problems": job.get("problems") or [],
        }
    )


@app.route("/admin/events/<uid>/delete", methods=["POST"])
def delete_event(uid: str):
    CATALOG.remove(uid)
    return redirect(url_for("catalog"))


@app.route("/admin/events/clear", methods=["POST"])
def clear_events():
    CATALOG.clear()
    return redirect(url_for("catalog"))


@app.route("/admin/events/purge_past", methods=["POST"])
def purge_past():
    """Drop events that have already happened; undated ones are kept."""
    removed = CATALOG.remove_past()
    return redirect(url_for("catalog", purged=removed))


# Bound one click's work; a large catalog is handled by clicking again.
BACKFILL_MAX_CALLS = 25


@app.route("/admin/events/backfill", methods=["POST"])
def backfill_descriptions():
    """Generate descriptions for stored events that are missing one."""
    pending = CATALOG.missing_descriptions()
    if not pending:
        return redirect(url_for("catalog"))

    try:
        converter = CaptionToEventConverter()
    except (ValueError, ImportError) as exc:
        return _render_catalog(error=str(exc))

    descriptions = converter.describe_events(pending, max_calls=BACKFILL_MAX_CALLS)
    filled = CATALOG.update_descriptions(descriptions)
    remaining = len(CATALOG.missing_descriptions())
    return redirect(url_for("catalog", filled=filled, remaining=remaining))


# ---------------------------------------------------------------------------
# Client panel — survey, deck, swipe, download
# ---------------------------------------------------------------------------


def _render_survey(error: Optional[str] = None):
    return render_template(
        "index.html",
        error=error,
        categories=EVENT_CATEGORIES,
        category_labels=CATEGORY_LABELS,
        category_blurbs=CATEGORY_BLURBS,
    )


@app.route("/", methods=["GET"])
def index():
    return _render_survey()


@app.route("/deck", methods=["POST"])
def build_deck():
    selected = [
        category
        for category in (
            normalize_category(value) for value in request.form.getlist("categories")
        )
        if category
    ]
    # No selection means "surprise me" — everything.
    categories = selected or list(EVENT_CATEGORIES)

    events = CATALOG.upcoming(categories=categories)
    if not events:
        if len(CATALOG) == 0:
            return _render_survey(
                error="No events have been collected yet. Ask an admin to scrape some sources."
            )
        return _render_survey(
            error="No upcoming events match those interests yet. Try picking more."
        )

    _set_pending(events, categories=categories)
    return redirect(url_for("review"))


def _chip_statuses(pending: List[dict], accepted: List[dict], index: int) -> Dict[str, str]:
    """Map uid -> calendar status for the deck at the current review position.

    Declined events (already passed, never accepted) are left out of the map
    entirely, which is what makes their chips disappear from the calendar.
    """
    accepted_uids = {item.get("uid") for item in accepted}
    statuses: Dict[str, str] = {}
    for position, item in enumerate(pending):
        uid = item.get("uid")
        if not uid:
            continue
        if position > index:
            statuses[uid] = STATUS_PENDING
        elif position == index:
            statuses[uid] = STATUS_CURRENT
        elif uid in accepted_uids:
            statuses[uid] = STATUS_ACCEPTED
    return statuses


@app.route("/review", methods=["GET"])
def review():
    state = _get_review_state()
    pending = state["pending_events"]
    index = int(state["review_index"])
    accepted = state["accepted_events"]

    if not pending:
        return redirect(url_for("index"))

    if index >= len(pending):
        return redirect(url_for("done"))

    event = pending[index]
    return render_template(
        "review.html",
        event=event,
        index=index + 1,
        total=len(pending),
        accepted_count=len(accepted),
        category_labels=CATEGORY_LABELS,
        calendar=build_week(pending, _chip_statuses(pending, accepted, index)),
    )


@app.route("/decide", methods=["POST"])
def decide():
    state = _get_review_state()
    pending = state["pending_events"]
    index = int(state["review_index"])
    decision = (request.form.get("decision") or "").strip().lower()

    if not pending or index >= len(pending):
        return redirect(url_for("done"))

    if decision == "yes":
        accepted = state["accepted_events"]
        accepted.append(pending[index])
        _update_review_state(accepted_events=accepted)
        _write_ics_file(accepted, deck_id=state["deck_id"])

    next_index = index + 1
    _update_review_state(review_index=next_index)

    if next_index >= len(pending):
        return redirect(url_for("done"))
    return redirect(url_for("review"))


@app.route("/done", methods=["GET"])
def done():
    state = _get_review_state()
    accepted = state["accepted_events"]
    has_file = bool(accepted) and _deck_ics_path(state["deck_id"]).exists()
    return render_template(
        "done.html",
        accepted=accepted,
        accepted_count=len(accepted),
        categories=state["categories"],
        category_labels=CATEGORY_LABELS,
        has_file=has_file,
    )


@app.route("/download", methods=["GET"])
def download():
    ics_path = _deck_ics_path(_get_review_state()["deck_id"])
    if not ics_path.exists():
        return redirect(url_for("done"))
    return send_file(
        ics_path,
        as_attachment=True,
        download_name="my_events.ics",
        mimetype="text/calendar",
    )


def _deck_ics_path(deck_id: str) -> Path:
    return OUTPUT_DIR / f"deck_{deck_id or 'events'}.ics"


def _write_ics_file(accepted_dicts: List[dict], *, deck_id: str) -> Path:
    events = [CalendarEvent(**item) for item in accepted_dicts]
    content = build_ics(events, calendar_name="My events")
    path = _deck_ics_path(deck_id)
    path.write_text(content, encoding="utf-8")
    return path


if __name__ == "__main__":
    # 5050 avoids macOS AirPlay Receiver, which often binds to 5000.
    # use_reloader=False so background scrape threads are not killed by the reloader.
    app.run(debug=True, port=5050, use_reloader=False)
