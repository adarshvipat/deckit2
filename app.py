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


def _session_events() -> List[dict]:
    return session.get("pending_events", [])


def _accepted_events() -> List[dict]:
    return session.get("accepted_events", [])


def _set_pending(events: List[CalendarEvent]) -> None:
    session["pending_events"] = [event.to_dict() for event in events]
    session["accepted_events"] = []
    session["review_index"] = 0


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
        }

    thread = threading.Thread(
        target=_run_scrape_job,
        args=(job_id, username, count),
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
    _set_pending(events)
    session["profile"] = job.get("profile") or "events"
    session.pop("active_job_id", None)

    with _JOBS_LOCK:
        _JOBS.pop(job_id, None)

    return redirect(url_for("review"))


@app.route("/review", methods=["GET"])
def review():
    pending = _session_events()
    index = int(session.get("review_index", 0))
    accepted = _accepted_events()

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
        profile=session.get("profile", ""),
    )


@app.route("/decide", methods=["POST"])
def decide():
    pending = _session_events()
    index = int(session.get("review_index", 0))
    decision = (request.form.get("decision") or "").strip().lower()

    if not pending or index >= len(pending):
        return redirect(url_for("done"))

    if decision == "yes":
        accepted = _accepted_events()
        accepted.append(pending[index])
        session["accepted_events"] = accepted
        _write_ics_file(accepted)

    session["review_index"] = index + 1

    if session["review_index"] >= len(pending):
        return redirect(url_for("done"))
    return redirect(url_for("review"))


@app.route("/done", methods=["GET"])
def done():
    accepted = _accepted_events()
    profile = session.get("profile", "events")
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
    profile = session.get("profile", "events")
    ics_path = OUTPUT_DIR / f"{profile}_events.ics"
    if not ics_path.exists():
        return redirect(url_for("done"))
    return send_file(
        ics_path,
        as_attachment=True,
        download_name=f"{profile}_events.ics",
        mimetype="text/calendar",
    )


def _write_ics_file(accepted_dicts: List[dict]) -> Path:
    profile = session.get("profile", "events")
    events = [CalendarEvent(**item) for item in accepted_dicts]
    content = build_ics(events, calendar_name=f"@{profile} events")
    path = OUTPUT_DIR / f"{profile}_events.ics"
    path.write_text(content, encoding="utf-8")
    return path


if __name__ == "__main__":
    # 5050 avoids macOS AirPlay Receiver, which often binds to 5000.
    # use_reloader=False so background scrape threads are not killed by the reloader.
    app.run(debug=True, port=5050, use_reloader=False)
