#!/usr/bin/env python3
"""Flask app: scrape → LLM events → yes/no review → download ICS."""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
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
from event_store import EVENT_STORE, CalendarEvent
from ics_builder import build_ics
from instagram_scraper import InstagramProfileParser, InstagramScraper, ScraperConfig
from url_scraper import UrlEventScraper, UrlScraperConfig, normalize_url

load_dotenv(Path(__file__).resolve().parent / ".env")

app = Flask(__name__)
app.secret_key = "instagram-event-review-dev-key"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory scrape jobs for progress UI (single-process Flask).
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()

# Review state (pending/accepted events) lives server-side, keyed by a small
# id stored in the browser's session cookie — URL-mode events carry their
# full source text and easily blow past the ~4KB signed-cookie session limit
# if stored there directly.
_REVIEW_SESSIONS: Dict[str, Dict[str, Any]] = {}
_REVIEW_LOCK = threading.Lock()

_EMPTY_REVIEW_STATE: Dict[str, Any] = {
    "pending_events": [],
    "accepted_events": [],
    "review_index": 0,
    "profile": "",
    "kind": "instagram",
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


def _set_pending(events: List[CalendarEvent], *, profile: str, kind: str) -> None:
    review_id = uuid.uuid4().hex
    with _REVIEW_LOCK:
        _REVIEW_SESSIONS[review_id] = {
            "pending_events": [event.to_dict() for event in events],
            "accepted_events": [],
            "review_index": 0,
            "profile": profile,
            "kind": kind,
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


def _run_scrape_job(job_id: str, username: str, count: int) -> None:
    try:
        _update_job(
            job_id,
            status="running",
            progress=8,
            message="Starting up…",
        )
        EVENT_STORE.clear()
        config = ScraperConfig.from_env(count=count, headless=True)

        _update_job(job_id, progress=18, message="Opening the browser…")
        with InstagramScraper(config) as scraper:
            _update_job(
                job_id,
                progress=32,
                message="Scraping the web…",
            )
            results = scraper.scrape_profiles([username])

        for result in results:
            if not result.succeeded:
                _update_job(
                    job_id,
                    status="error",
                    progress=100,
                    message="Scrape failed",
                    error=f"Scrape failed for @{username}: {result.error}",
                )
                return

        _update_job(
            job_id,
            progress=58,
            message="Reading captions…",
        )

        converter = CaptionToEventConverter()
        posts = results[0].posts if results else []
        total = max(len(posts), 1)
        events: List[CalendarEvent] = []

        for index, post in enumerate(posts):
            pct = 58 + int(32 * (index + 1) / total)
            _update_job(
                job_id,
                progress=min(pct, 90),
                message="Processing with AI…",
            )
            events.extend(
                converter.convert_caption(
                    post,
                    source_username=username,
                )
            )

        EVENT_STORE.extend(events)
        _update_job(job_id, progress=95, message="Almost done…")

        if not events:
            _update_job(
                job_id,
                status="error",
                progress=100,
                message="No events found",
                error=(
                    f"No calendar events found in the last {count} posts "
                    f"from @{username}."
                ),
            )
            return

        _update_job(
            job_id,
            status="done",
            progress=100,
            message="Ready to review",
            events=[event.to_dict() for event in events],
            profile=username,
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


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", error=None)


@app.route("/scrape", methods=["POST"])
def scrape():
    profile = (request.form.get("profile") or "").strip()
    count_raw = (request.form.get("count") or "5").strip()
    try:
        count = max(1, int(count_raw))
    except ValueError:
        count = 5

    if not profile:
        return render_template("index.html", error="Enter an Instagram username or profile URL.")

    try:
        username = InstagramProfileParser.parse(profile)
    except ValueError as exc:
        return render_template("index.html", error=str(exc))

    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Queued…",
            "error": None,
            "events": [],
            "profile": username,
            "kind": "instagram",
        }

    thread = threading.Thread(
        target=_run_scrape_job,
        args=(job_id, username, count),
        daemon=True,
    )
    thread.start()
    session["active_job_id"] = job_id
    return redirect(url_for("loading", job_id=job_id))


def _run_url_scrape_job(job_id: str, url: str, max_chunks: int) -> None:
    try:
        _update_job(
            job_id,
            status="running",
            progress=8,
            message="Starting up…",
        )
        EVENT_STORE.clear()
        config = UrlScraperConfig(max_chunks=max_chunks)

        _update_job(job_id, progress=18, message="Opening the browser…")
        with UrlEventScraper(config) as scraper:
            _update_job(
                job_id,
                progress=32,
                message="Scraping the web…",
            )
            result = scraper.scrape(url)

        if not result.succeeded:
            _update_job(
                job_id,
                status="error",
                progress=100,
                message="Scrape failed",
                error=f"Scrape failed for {url}: {result.error}",
            )
            return

        _update_job(
            job_id,
            progress=58,
            message="Reading page text…",
        )

        converter = CaptionToEventConverter()
        chunks = result.chunks
        total = max(len(chunks), 1)
        events: List[CalendarEvent] = []

        for index, chunk in enumerate(chunks):
            pct = 58 + int(32 * (index + 1) / total)
            _update_job(
                job_id,
                progress=min(pct, 90),
                message="Processing with AI…",
            )
            events.extend(
                converter.convert_caption(
                    chunk,
                    source_username=result.label,
                )
            )

        EVENT_STORE.extend(events)
        _update_job(job_id, progress=95, message="Almost done…")

        if not events:
            _update_job(
                job_id,
                status="error",
                progress=100,
                message="No events found",
                error=f"No calendar events found on {url}.",
            )
            return

        _update_job(
            job_id,
            status="done",
            progress=100,
            message="Ready to review",
            events=[event.to_dict() for event in events],
            profile=result.label,
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


@app.route("/scrape_url", methods=["POST"])
def scrape_url():
    raw_url = (request.form.get("url") or "").strip()
    sections_raw = (request.form.get("sections") or "6").strip()
    try:
        max_chunks = max(1, int(sections_raw))
    except ValueError:
        max_chunks = 6

    if not raw_url:
        return render_template("index.html", error="Enter a page URL to scrape.")

    try:
        url = normalize_url(raw_url)
    except ValueError as exc:
        return render_template("index.html", error=str(exc))

    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Queued…",
            "error": None,
            "events": [],
            "profile": url,
            "kind": "url",
        }

    thread = threading.Thread(
        target=_run_url_scrape_job,
        args=(job_id, url, max_chunks),
        daemon=True,
    )
    thread.start()
    session["active_job_id"] = job_id
    return redirect(url_for("loading", job_id=job_id))


@app.route("/loading/<job_id>", methods=["GET"])
def loading(job_id: str):
    job = _get_job(job_id)
    if not job:
        return render_template(
            "index.html",
            error="That scrape job was not found. Try again.",
        )
    return render_template(
        "loading.html",
        job_id=job_id,
        profile=job.get("profile", ""),
        kind=job.get("kind", "instagram"),
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
            "profile": job.get("profile"),
            "event_count": len(job.get("events") or []),
        }
    )


@app.route("/jobs/<job_id>/claim", methods=["POST"])
def claim_job(job_id: str):
    job = _get_job(job_id)
    if not job:
        return render_template("index.html", error="That scrape job was not found.")

    if job.get("status") == "error":
        return render_template("index.html", error=job.get("error") or "Scrape failed.")

    if job.get("status") != "done":
        return redirect(url_for("loading", job_id=job_id))

    events = [CalendarEvent(**item) for item in (job.get("events") or [])]
    _set_pending(
        events,
        profile=job.get("profile") or "events",
        kind=job.get("kind", "instagram"),
    )
    session.pop("active_job_id", None)

    with _JOBS_LOCK:
        _JOBS.pop(job_id, None)

    return redirect(url_for("review"))


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
        profile=state["profile"],
        kind=state["kind"],
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
        _write_ics_file(accepted, profile=state["profile"])

    next_index = index + 1
    _update_review_state(review_index=next_index)

    if next_index >= len(pending):
        return redirect(url_for("done"))
    return redirect(url_for("review"))


@app.route("/done", methods=["GET"])
def done():
    state = _get_review_state()
    accepted = state["accepted_events"]
    profile = state["profile"] or "events"
    ics_path = OUTPUT_DIR / f"{profile}_events.ics"
    has_file = ics_path.exists() and bool(accepted)
    return render_template(
        "done.html",
        accepted=accepted,
        accepted_count=len(accepted),
        profile=profile,
        has_file=has_file,
    )


@app.route("/download", methods=["GET"])
def download():
    profile = _get_review_state()["profile"] or "events"
    ics_path = OUTPUT_DIR / f"{profile}_events.ics"
    if not ics_path.exists():
        return redirect(url_for("done"))
    return send_file(
        ics_path,
        as_attachment=True,
        download_name=f"{profile}_events.ics",
        mimetype="text/calendar",
    )


def _write_ics_file(accepted_dicts: List[dict], *, profile: str) -> Path:
    profile = profile or "events"
    events = [CalendarEvent(**item) for item in accepted_dicts]
    content = build_ics(events, calendar_name=f"@{profile} events")
    path = OUTPUT_DIR / f"{profile}_events.ics"
    path.write_text(content, encoding="utf-8")
    return path


if __name__ == "__main__":
    # 5050 avoids macOS AirPlay Receiver, which often binds to 5000.
    # use_reloader=False so background scrape threads are not killed by the reloader.
    app.run(debug=True, port=5050, use_reloader=False)
