"""Instagram caption scraper library using Selenium."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Set
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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScraperConfig:
    count: int = 10
    max_retries: int = 3
    retry_backoff: float = 5.0
    delay: float = 0.0
    headless: bool = True
    instagram_username: Optional[str] = None
    instagram_password: Optional[str] = None

    @classmethod
    def from_env(cls, **overrides) -> ScraperConfig:
        config = cls(
            instagram_username=os.environ.get("INSTAGRAM_USERNAME"),
            instagram_password=os.environ.get("INSTAGRAM_PASSWORD"),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(config, key, value)
        return config


@dataclass
class ProfileScrapeResult:
    username: str
    posts: List[PostCaption] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "username": self.username,
            "posts": [post.to_dict() for post in self.posts],
            "error": self.error,
        }


class InstagramProfileParser:
    """Parse profile URLs and account list files."""

    @staticmethod
    def parse(value: str) -> str:
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

    @classmethod
    def read_accounts_file(cls, path: str) -> List[str]:
        usernames: List[str] = []
        with open(path, encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    usernames.append(cls.parse(line))
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not usernames:
            raise ValueError(f"No accounts found in {path}")
        return usernames


class InstagramScraper:
    """Scrape recent post captions from public Instagram profiles."""

    def __init__(self, config: Optional[ScraperConfig] = None) -> None:
        self.config = config or ScraperConfig()
        self._driver: Optional[webdriver.Chrome] = None

    def __enter__(self) -> InstagramScraper:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self._driver is not None:
            return
        self._driver = self._create_driver()
        self._login_if_needed()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.quit()
            self._driver = None

    @property
    def driver(self) -> webdriver.Chrome:
        if self._driver is None:
            raise ScraperError("Scraper is not started. Use start() or a with block.")
        return self._driver

    def fetch_captions(self, username: str) -> List[PostCaption]:
        """Fetch captions for one profile."""
        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                return self._fetch_captions_once(username)
            except (RateLimitError, TimeoutException, WebDriverException) as exc:
                last_error = exc
                if attempt == self.config.max_retries:
                    break
                sleep_for = self.config.retry_backoff * attempt
                logging.warning(
                    "Retryable error for %s (attempt %d/%d): %s. Sleeping %.1fs.",
                    username,
                    attempt,
                    self.config.max_retries,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)

        if last_error is not None:
            raise last_error
        raise ScraperError(f"Failed to fetch captions for {username}.")

    def scrape_profiles(self, usernames: List[str]) -> List[ProfileScrapeResult]:
        """Scrape multiple profiles, continuing after individual failures."""
        self.start()
        results: List[ProfileScrapeResult] = []

        for index, username in enumerate(usernames):
            logging.info("Fetching %s", username)
            try:
                posts = self.fetch_captions(username)
                results.append(ProfileScrapeResult(username=username, posts=posts))
                logging.info("Fetched %d posts for %s", len(posts), username)
            except ProfileNotFoundError as exc:
                results.append(ProfileScrapeResult(username=username, error=str(exc)))
                logging.error("Profile not found: %s", exc)
            except LoginRequiredError as exc:
                results.append(ProfileScrapeResult(username=username, error=str(exc)))
                logging.error("%s", exc)
            except ScraperError as exc:
                results.append(ProfileScrapeResult(username=username, error=str(exc)))
                logging.error("Failed to fetch %s: %s", username, exc)

            if self.config.delay > 0 and index < len(usernames) - 1:
                logging.info("Waiting %.1fs before next profile", self.config.delay)
                time.sleep(self.config.delay)

        return results

    def _fetch_captions_once(self, username: str) -> List[PostCaption]:
        self._load_profile(username)
        post_urls = self._collect_post_urls(self.config.count)
        if not post_urls:
            raise ScraperError(f"No posts found for {username}.")

        captions: List[PostCaption] = []
        for post_url in post_urls:
            self.driver.get(post_url)
            WebDriverWait(self.driver, DEFAULT_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(1.5)

            shortcode = self._extract_shortcode(post_url)
            if not shortcode:
                continue

            captions.append(
                PostCaption(
                    shortcode=shortcode,
                    caption=self._extract_caption_from_post_page(),
                    timestamp=self._extract_timestamp_from_post_page(),
                    url=post_url,
                )
            )

        return captions

    def _create_driver(self) -> webdriver.Chrome:
        options = Options()
        if self.config.headless:
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

    def _login_if_needed(self) -> None:
        username = self.config.instagram_username
        password = self.config.instagram_password
        if username and password:
            self._login(username, password)

    def _login(self, username: str, password: str) -> None:
        self.driver.get("https://www.instagram.com/accounts/login/")
        wait = WebDriverWait(self.driver, DEFAULT_TIMEOUT)

        try:
            wait.until(EC.presence_of_element_located((By.NAME, "username")))
        except TimeoutException as exc:
            raise ScraperError("Login page did not load.") from exc

        self._dismiss_cookie_banner()
        self.driver.find_element(By.NAME, "username").send_keys(username)
        self.driver.find_element(By.NAME, "password").send_keys(password)
        self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

        time.sleep(4)
        self._dismiss_cookie_banner()

        for label in ("Not Now", "Not now"):
            try:
                self.driver.find_element(
                    By.XPATH,
                    f"//button[contains(., '{label}')]",
                ).click()
                time.sleep(0.5)
            except NoSuchElementException:
                continue

        if "accounts/login" in self.driver.current_url:
            raise ScraperError(
                "Login failed — check INSTAGRAM_USERNAME / INSTAGRAM_PASSWORD."
            )

        logging.info("Logged in as %s", username)

    def _dismiss_cookie_banner(self) -> None:
        for label in (
            "Allow all cookies",
            "Accept All",
            "Allow essential and optional cookies",
        ):
            try:
                button = self.driver.find_element(
                    By.XPATH,
                    f"//button[contains(., '{label}')]",
                )
                button.click()
                time.sleep(0.5)
                return
            except NoSuchElementException:
                continue

    def _load_profile(self, username: str) -> None:
        self.driver.get(f"https://www.instagram.com/{username}/")
        WebDriverWait(self.driver, DEFAULT_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)
        self._dismiss_cookie_banner()
        self._check_profile_page(username)

    def _check_profile_page(self, username: str) -> None:
        text = self._page_text()
        lowered = text.lower()

        if "sorry, this page isn't available" in lowered:
            raise ProfileNotFoundError(f"Profile {username} does not exist.")
        if "this account is private" in lowered:
            raise LoginRequiredError(f"Profile {username} is private.")
        if "log in to instagram" in lowered and "accounts/login" in self.driver.current_url:
            raise LoginRequiredError(
                f"Instagram requires login to view {username}. "
                "Set INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD."
            )
        if "please wait a few minutes" in lowered or "try again later" in lowered:
            raise RateLimitError("Instagram rate limit detected on profile page.")

    def _collect_post_urls(self, count: int) -> List[str]:
        seen_shortcodes: Set[str] = set()
        post_urls: List[str] = []

        for _ in range(MAX_SCROLL_ATTEMPTS):
            anchors = self.driver.find_elements(
                By.CSS_SELECTOR,
                "a[href*='/p/'], a[href*='/reel/']",
            )
            for anchor in anchors:
                href = anchor.get_attribute("href") or ""
                shortcode = self._extract_shortcode(href)
                if not shortcode or shortcode in seen_shortcodes:
                    continue
                seen_shortcodes.add(shortcode)
                post_urls.append(self._normalize_post_url(href))
                if len(post_urls) >= count:
                    return post_urls[:count]

            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE)

        return post_urls[:count]

    def _extract_caption_from_post_page(self) -> Optional[str]:
        wait = WebDriverWait(self.driver, DEFAULT_TIMEOUT)
        try:
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "article")))
        except TimeoutException:
            pass

        for selector in ("article h1", "article ul li span"):
            for element in self.driver.find_elements(By.CSS_SELECTOR, selector):
                text = element.text.strip()
                if text and len(text) > 1:
                    return text

        try:
            meta = self.driver.find_element(
                By.CSS_SELECTOR,
                'meta[property="og:description"]',
            )
            return self._parse_og_description(meta.get_attribute("content") or "")
        except NoSuchElementException:
            return None

    def _extract_timestamp_from_post_page(self) -> Optional[str]:
        try:
            time_element = self.driver.find_element(By.CSS_SELECTOR, "article time")
            return time_element.get_attribute("datetime")
        except NoSuchElementException:
            return None

    def _page_text(self) -> str:
        try:
            return self.driver.find_element(By.TAG_NAME, "body").text
        except NoSuchElementException:
            return ""

    @staticmethod
    def _extract_shortcode(url: str) -> Optional[str]:
        match = POST_PATH_RE.search(url)
        return match.group(1) if match else None

    @staticmethod
    def _normalize_post_url(url: str) -> str:
        shortcode = InstagramScraper._extract_shortcode(url)
        if not shortcode:
            return url
        return f"https://www.instagram.com/p/{shortcode}/"

    @staticmethod
    def _parse_og_description(content: str) -> Optional[str]:
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
