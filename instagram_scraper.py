#!/usr/bin/env python3
"""Fetch captions from the last N posts on a public Instagram profile.

Uses Selenium with Chrome to load profile and post pages in a real browser.

Usage:
    python instagram_scraper.py "https://www.instagram.com/nasa/" --count 10
    python instagram_scraper.py nasa --count 10 --json
    python instagram_scraper.py --file accounts.txt --count 10 --delay 45

Rate-limit guidance:
    Instagram limits requests per IP (~200/hour). When scraping many accounts,
    use --delay (30-60 seconds recommended) and optional login via env vars
    INSTAGRAM_USERNAME / INSTAGRAM_PASSWORD (use a dedicated scraper account).

Requires Google Chrome installed. ChromeDriver is managed automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from typing import Iterator, List, Optional, Set
from urllib.parse import urlparse

from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

INSTAGRAM_PROFILE_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?instagram\.com/([A-Za-z0-9._]+)/?$"
)
POST_PATH_RE = re.compile(r"/(?:p|reel|reels)/([A-Za-z0-9_-]+)")
INVALID_PROFILE_PATHS = {
    "p",
    "reel",
    "reels",
    "stories",
    "explore",
    "accounts",
    "direct",
    "tv",
    "about",
    "legal",
    "developer",
}
OG_CAPTION_RE = re.compile(r':\s*"(.+)"\s*$', re.DOTALL)

DEFAULT_TIMEOUT = 20
SCROLL_PAUSE = 1.5
MAX_SCROLL_ATTEMPTS = 8


class ScraperError(Exception):
    """Base error for scraper failures."""


class ProfileNotFoundError(ScraperError):
    """Profile does not exist or is unavailable."""


class LoginRequiredError(ScraperError):
    """Profile is private or Instagram requires login."""


class RateLimitError(ScraperError):
    """Instagram rate-limited or blocked the request."""


@dataclass
class PostCaption:
    shortcode: str
    caption: Optional[str]
    timestamp: Optional[str]
    url: str


def setup_logging(log_file: Optional[str]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def parse_username(value: str) -> str:
    """Extract an Instagram username from a URL or bare username."""
    value = value.strip()
    if not value:
        raise ValueError("Empty profile input.")

    if "instagram.com" in value:
        parsed = urlparse(value if "://" in value else f"https://{value}")
        path = parsed.path.strip("/")
        if not path:
            raise ValueError(f"Invalid Instagram profile URL: {value}")

        first_segment = path.split("/")[0].lower()
        if first_segment in INVALID_PROFILE_PATHS:
            raise ValueError(
                f"URL looks like a post or special page, not a profile: {value}"
            )

        username = path.split("/")[0]
    elif INSTAGRAM_PROFILE_RE.match(value):
        username = INSTAGRAM_PROFILE_RE.match(value).group(1)  # type: ignore[union-attr]
    else:
        username = value.lstrip("@")

    username = username.strip()
    if not username or not re.fullmatch(r"[A-Za-z0-9._]+", username):
        raise ValueError(f"Invalid Instagram username: {value}")

    return username


def create_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(60)
    return driver


def dismiss_cookie_banner(driver: webdriver.Chrome) -> None:
    for label in ("Allow all cookies", "Accept All", "Allow essential and optional cookies"):
        try:
            button = driver.find_element(
                By.XPATH,
                f"//button[contains(., '{label}')]",
            )
            button.click()
            time.sleep(0.5)
            return
        except NoSuchElementException:
            continue


def login(driver: webdriver.Chrome, username: str, password: str) -> None:
    driver.get("https://www.instagram.com/accounts/login/")
    wait = WebDriverWait(driver, DEFAULT_TIMEOUT)

    try:
        wait.until(EC.presence_of_element_located((By.NAME, "username")))
    except TimeoutException as exc:
        raise ScraperError("Login page did not load.") from exc

    dismiss_cookie_banner(driver)

    driver.find_element(By.NAME, "username").send_keys(username)
    driver.find_element(By.NAME, "password").send_keys(password)
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

    time.sleep(4)
    dismiss_cookie_banner(driver)

    for label in ("Not Now", "Not now"):
        try:
            driver.find_element(By.XPATH, f"//button[contains(., '{label}')]").click()
            time.sleep(0.5)
        except NoSuchElementException:
            continue

    if "accounts/login" in driver.current_url:
        raise ScraperError("Login failed — check INSTAGRAM_USERNAME / INSTAGRAM_PASSWORD.")

    logging.info("Logged in as %s", username)


def ensure_logged_in(driver: webdriver.Chrome) -> None:
    username = os.environ.get("INSTAGRAM_USERNAME")
    password = os.environ.get("INSTAGRAM_PASSWORD")
    if username and password:
        login(driver, username, password)


def page_text(driver: webdriver.Chrome) -> str:
    try:
        return driver.find_element(By.TAG_NAME, "body").text
    except NoSuchElementException:
        return ""


def check_profile_page(driver: webdriver.Chrome, username: str) -> None:
    text = page_text(driver)
    lowered = text.lower()

    if "sorry, this page isn't available" in lowered:
        raise ProfileNotFoundError(f"Profile {username} does not exist.")
    if "this account is private" in lowered:
        raise LoginRequiredError(f"Profile {username} is private.")
    if "log in to instagram" in lowered and "accounts/login" in driver.current_url:
        raise LoginRequiredError(
            f"Instagram requires login to view {username}. "
            "Set INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD."
        )
    if "please wait a few minutes" in lowered or "try again later" in lowered:
        raise RateLimitError("Instagram rate limit detected on profile page.")


def extract_shortcode(url: str) -> Optional[str]:
    match = POST_PATH_RE.search(url)
    return match.group(1) if match else None


def collect_post_urls(driver: webdriver.Chrome, count: int) -> List[str]:
    """Scroll the profile grid and collect unique post/reel URLs."""
    seen_shortcodes: Set[str] = set()
    post_urls: List[str] = []

    for _ in range(MAX_SCROLL_ATTEMPTS):
        anchors = driver.find_elements(By.CSS_SELECTOR, "a[href*='/p/'], a[href*='/reel/']")
        for anchor in anchors:
            href = anchor.get_attribute("href") or ""
            shortcode = extract_shortcode(href)
            if not shortcode or shortcode in seen_shortcodes:
                continue
            seen_shortcodes.add(shortcode)
            post_urls.append(normalize_post_url(href))
            if len(post_urls) >= count:
                return post_urls[:count]

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)

    return post_urls[:count]


def normalize_post_url(url: str) -> str:
    shortcode = extract_shortcode(url)
    if not shortcode:
        return url
    return f"https://www.instagram.com/p/{shortcode}/"


def parse_og_description(content: str) -> Optional[str]:
    if not content:
        return None
    match = OG_CAPTION_RE.search(content)
    if match:
        text = match.group(1).strip().rstrip('"').rstrip(".")
    elif ": " in content:
        text = content.rsplit(": ", 1)[-1].strip().strip('"')
    else:
        text = content.strip()

    text = text.replace("\\n", "\n").strip()
    return text or None


def extract_caption_from_post_page(driver: webdriver.Chrome) -> Optional[str]:
    wait = WebDriverWait(driver, DEFAULT_TIMEOUT)
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "article")))
    except TimeoutException:
        pass

    for selector in ("article h1", "article ul li span"):
        for element in driver.find_elements(By.CSS_SELECTOR, selector):
            text = element.text.strip()
            if text and len(text) > 1:
                return text

    try:
        meta = driver.find_element(By.CSS_SELECTOR, 'meta[property="og:description"]')
        return parse_og_description(meta.get_attribute("content") or "")
    except NoSuchElementException:
        return None


def extract_timestamp_from_post_page(driver: webdriver.Chrome) -> Optional[str]:
    try:
        time_element = driver.find_element(By.CSS_SELECTOR, "article time")
        return time_element.get_attribute("datetime")
    except NoSuchElementException:
        return None


def load_profile(driver: webdriver.Chrome, username: str) -> None:
    driver.get(f"https://www.instagram.com/{username}/")
    WebDriverWait(driver, DEFAULT_TIMEOUT).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    time.sleep(2)
    dismiss_cookie_banner(driver)
    check_profile_page(driver, username)


def fetch_captions(
    driver: webdriver.Chrome,
    username: str,
    count: int,
    max_retries: int = 3,
    retry_backoff: float = 5.0,
) -> List[PostCaption]:
    """Fetch the last `count` captions for a profile, with retry/backoff."""
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            load_profile(driver, username)
            post_urls = collect_post_urls(driver, count)
            if not post_urls:
                raise ScraperError(f"No posts found for {username}.")

            captions: List[PostCaption] = []
            for post_url in post_urls:
                driver.get(post_url)
                WebDriverWait(driver, DEFAULT_TIMEOUT).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                time.sleep(1.5)

                shortcode = extract_shortcode(post_url)
                if not shortcode:
                    continue

                captions.append(
                    PostCaption(
                        shortcode=shortcode,
                        caption=extract_caption_from_post_page(driver),
                        timestamp=extract_timestamp_from_post_page(driver),
                        url=post_url,
                    )
                )

            return captions
        except (RateLimitError, TimeoutException, WebDriverException) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            sleep_for = retry_backoff * attempt
            logging.warning(
                "Retryable error for %s (attempt %d/%d): %s. Sleeping %.1fs.",
                username,
                attempt,
                max_retries,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)

    if last_error is not None:
        raise last_error
    raise ScraperError(f"Failed to fetch captions for {username}.")


def read_accounts_file(path: str) -> List[str]:
    usernames: List[str] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                usernames.append(parse_username(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not usernames:
        raise ValueError(f"No accounts found in {path}")
    return usernames


def print_results(username: str, captions: List[PostCaption], as_json: bool) -> None:
    if as_json:
        payload = {"username": username, "posts": [asdict(item) for item in captions]}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"@{username} — last {len(captions)} captions:")
    for index, item in enumerate(captions, start=1):
        caption_text = item.caption if item.caption else "(no caption)"
        print(f"\n[{index}] {item.url}")
        if item.timestamp:
            print(f"    {item.timestamp}")
        print(f"    {caption_text}")


def scrape_accounts(
    accounts: Iterator[str],
    count: int,
    as_json: bool,
    delay: float,
    max_retries: int,
    headless: bool,
) -> int:
    failures = 0
    accounts_list = list(accounts)
    driver = create_driver(headless=headless)

    try:
        ensure_logged_in(driver)

        for index, username in enumerate(accounts_list):
            logging.info("Fetching %s", username)
            try:
                captions = fetch_captions(
                    driver,
                    username,
                    count=count,
                    max_retries=max_retries,
                )
                print_results(username, captions, as_json=as_json)
                logging.info("Fetched %d posts for %s", len(captions), username)
            except ProfileNotFoundError as exc:
                failures += 1
                logging.error("Profile not found: %s", exc)
            except LoginRequiredError as exc:
                failures += 1
                logging.error("%s", exc)
            except ScraperError as exc:
                failures += 1
                logging.error("Failed to fetch %s: %s", username, exc)

            if delay > 0 and index < len(accounts_list) - 1:
                logging.info("Waiting %.1fs before next profile", delay)
                time.sleep(delay)
    finally:
        driver.quit()

    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch captions from the last N posts on public Instagram profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python instagram_scraper.py "https://www.instagram.com/nasa/" --count 10\n'
            "  python instagram_scraper.py nasa --count 10 --json\n"
            "  python instagram_scraper.py --file accounts.txt --count 10 --delay 45"
        ),
    )
    parser.add_argument(
        "profile",
        nargs="?",
        help="Instagram profile URL or username",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="Text file with one profile URL or username per line",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of recent posts to fetch (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between profiles in batch mode (default: 0)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts for rate limits / connection errors (default: 3)",
    )
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        help="Optional log file path",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show the browser window (useful for debugging login/captcha issues)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.log_file)

    if args.count < 1:
        parser.error("--count must be at least 1")

    if args.file and args.profile:
        parser.error("Use either a single profile argument or --file, not both")

    if not args.file and not args.profile:
        parser.error("Provide a profile URL/username or --file")

    try:
        if args.file:
            accounts = read_accounts_file(args.file)
        else:
            accounts = [parse_username(args.profile)]

        failures = scrape_accounts(
            accounts=iter(accounts),
            count=args.count,
            as_json=args.json,
            delay=args.delay,
            max_retries=args.retries,
            headless=not args.no_headless,
        )
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
