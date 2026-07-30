# deckit2

Turn Instagram accounts and event-listing pages into a personal, swipeable calendar.

An admin feeds in Instagram profiles and/or event page URLs. Selenium scrapes the
captions/text, an LLM (via OpenRouter) extracts structured calendar events from
them and tags each with a category, and everything lands in a shared catalog.
Visitors take a short interest survey, get served a personalized deck of
upcoming events, swipe yes/no on each one, and download the accepted events as
an `.ics` file they can import into any calendar app.

## How it works

1. **Admin panel (`/admin`)** — paste in Instagram usernames/URLs and/or event
   page URLs, kick off a scrape. A background job scrapes each source with
   Selenium, sends the extracted text to an LLM to pull out event details
   (title, start/end time, location, description, category), and saves the
   results to the catalog (`/admin/catalog`), where events can be reviewed,
   deleted, purged of past events, or backfilled with missing descriptions.
2. **Client flow (`/`)** — a visitor picks the categories they're interested in
   (`professional`, `fun`, `community`, or "surprise me"), which builds a deck
   of matching upcoming events from the catalog.
3. **Review (`/review`)** — the visitor swipes through the deck one event at a
   time, accepting or declining each, with a live week-calendar view of where
   accepted/pending events fall.
4. **Download (`/download`)** — accepted events are written out as a `.ics`
   file for the visitor to download and import.

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Flask app — admin panel, client survey/review/download flow |
| `instagram_scraper.py` | Selenium-based Instagram profile/caption scraper |
| `url_scraper.py` | Generic Selenium scraper for arbitrary event pages |
| `caption_to_events.py` | Sends scraped text to an LLM (OpenRouter) and parses events out |
| `event_store.py` | `CalendarEvent` model, category definitions, ICS/date helpers |
| `event_catalog.py` | Persistent catalog of scraped events (backing store for the app) |
| `catalog_stats.py` | Aggregate stats shown on the catalog page |
| `week_calendar.py` | Builds the week-grid view used in the review UI |
| `ics_builder.py` | Renders a list of events into an `.ics` calendar file |
| `main.py` | CLI entry point for scraping Instagram profiles without the web UI |
| `templates/`, `static/` | Flask templates and stylesheet |

## Setup

Requires Python 3.9+ and Google Chrome (Selenium drives it via
`webdriver-manager`, which downloads a matching chromedriver automatically).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in real values:

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Yes | Used to call an LLM for caption → event extraction |
| `OPENROUTER_MODEL` | No | Overrides the default model (`openai/gpt-4o-mini`) |
| `INSTAGRAM_USERNAME` / `INSTAGRAM_PASSWORD` | No | Optional login for a dedicated scraper account |

## Running the web app

```bash
python app.py
```

Serves at `http://localhost:5050`. Visit `/admin` to scrape sources into the
catalog, or `/` for the visitor survey/review flow.

## Running the CLI scraper

Scrape Instagram profiles directly, without the web UI:

```bash
python main.py nasa --count 5
python main.py --file accounts.txt --count 5 --delay 45
```

Run `python main.py --help` for the full list of options.
