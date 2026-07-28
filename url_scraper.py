"""Generic web page scraper: pull visible text out of any URL using a real browser.

Mirrors the Selenium approach in instagram_scraper.py but for arbitrary pages
instead of Instagram profiles. The extracted text is split into PostCaption
chunks — the same shape InstagramScraper produces — so CaptionToEventConverter
can turn it into calendar events without any changes to that pipeline.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional
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

from instagram_scraper import PostCaption

DEFAULT_TIMEOUT = 20
SCROLL_PAUSE = 1.5
DEFAULT_MAX_SCROLLS = 6
DEFAULT_CHUNK_SIZE = 4000
DEFAULT_MAX_CHUNKS = 6

# Many event listings paginate behind a button instead of infinite scroll.
LOAD_MORE_PATTERNS = ("load more", "show more", "view more", "see more")


class PageScraperError(Exception):
    """Base error for page scraper failures."""


@dataclass
class UrlScraperConfig:
    max_scrolls: int = DEFAULT_MAX_SCROLLS
    chunk_size: int = DEFAULT_CHUNK_SIZE
    max_chunks: int = DEFAULT_MAX_CHUNKS
    headless: bool = True


@dataclass
class PageScrapeResult:
    url: str
    title: Optional[str] = None
    chunks: List[PostCaption] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def label(self) -> str:
        """Short identifier for session/filename purposes — the URL-mode analogue
        of an Instagram username."""
        return _slugify(self.title) if self.title else _slugify(urlparse(self.url).netloc or self.url)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "error": self.error,
        }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "page"


def normalize_url(value: str) -> str:
    """Accept bare domains like 'example.com/page' as well as full URLs."""
    value = value.strip()
    if not value:
        raise ValueError("Empty URL.")
    parsed = urlparse(value if "://" in value else f"https://{value}")
    if not parsed.netloc:
        raise ValueError(f"Invalid URL: {value}")
    return parsed.geturl()


class UrlEventScraper:
    """Load an arbitrary web page and pull out its visible text."""

    def __init__(self, config: Optional[UrlScraperConfig] = None) -> None:
        self.config = config or UrlScraperConfig()
        self._driver: Optional[webdriver.Chrome] = None

    def __enter__(self) -> "UrlEventScraper":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self._driver is not None:
            return
        self._driver = self._create_driver()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.quit()
            self._driver = None

    @property
    def driver(self) -> webdriver.Chrome:
        if self._driver is None:
            raise PageScraperError("Scraper is not started. Use start() or a with block.")
        return self._driver

    def scrape(self, url: str) -> PageScrapeResult:
        self.start()
        try:
            text, title = self._load_and_extract(url)
        except (TimeoutException, WebDriverException) as exc:
            return PageScrapeResult(url=url, error=f"Could not load page: {exc}")
        except PageScraperError as exc:
            return PageScrapeResult(url=url, error=str(exc))

        if not text.strip():
            return PageScrapeResult(
                url=url, error="No readable text found on the page."
            )

        chunks = self._chunk_text(text, url)
        if not chunks:
            return PageScrapeResult(
                url=url, error="Page text could not be split into chunks."
            )
        return PageScrapeResult(url=url, title=title, chunks=chunks)

    def _load_and_extract(self, url: str) -> tuple[str, Optional[str]]:
        self.driver.get(url)
        WebDriverWait(self.driver, DEFAULT_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)

        text = self._body_text()
        for _ in range(self.config.max_scrolls):
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(SCROLL_PAUSE)
            clicked = self._click_load_more()
            if clicked:
                time.sleep(SCROLL_PAUSE)

            new_text = self._body_text()
            grew = len(new_text) > len(text)
            text = new_text
            if not grew and not clicked:
                break

        return text, (self.driver.title or None)

    def _body_text(self) -> str:
        try:
            return self.driver.find_element(By.TAG_NAME, "body").text
        except Exception:  # noqa: BLE001 - defensive, page may not be ready
            return ""

    def _click_load_more(self) -> bool:
        """Click a 'load more' / 'show more' style button if one is visible."""
        for pattern in LOAD_MORE_PATTERNS:
            xpath = (
                "//button[contains(translate(normalize-space(.), "
                "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
                f"'{pattern}')] | "
                "//a[contains(translate(normalize-space(.), "
                "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
                f"'{pattern}')]"
            )
            try:
                element = self.driver.find_element(By.XPATH, xpath)
                if element.is_displayed() and element.is_enabled():
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'});", element
                    )
                    element.click()
                    return True
            except (NoSuchElementException, WebDriverException):
                continue
        return False

    def _chunk_text(self, text: str, url: str) -> List[PostCaption]:
        # Pack by line rather than blank-line paragraphs — many rendered SPAs
        # (e.g. event list cards) have no blank lines between entries at all.
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        chunk_texts: List[str] = []
        current_lines: List[str] = []
        current_len = 0
        for line in lines:
            line_len = len(line) + 1
            if current_lines and current_len + line_len > self.config.chunk_size:
                chunk_texts.append("\n".join(current_lines))
                current_lines = []
                current_len = 0
            current_lines.append(line)
            current_len += line_len
        if current_lines:
            chunk_texts.append("\n".join(current_lines))

        chunk_texts = chunk_texts[: self.config.max_chunks]
        return [
            PostCaption(shortcode=f"chunk-{index}", caption=chunk, timestamp=None, url=url)
            for index, chunk in enumerate(chunk_texts, start=1)
        ]

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
