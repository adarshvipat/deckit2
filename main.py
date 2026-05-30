#!/usr/bin/env python3
"""CLI entry point for the Instagram caption scraper."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from instagram_scraper import (
    InstagramProfileParser,
    InstagramScraper,
    ProfileScrapeResult,
    ScraperConfig,
)


def setup_logging(log_file: Optional[str]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def print_result(result: ProfileScrapeResult, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return

    if not result.succeeded:
        print(f"@{result.username} — error: {result.error}")
        return

    print(f"@{result.username} — last {len(result.posts)} captions:")
    for index, item in enumerate(result.posts, start=1):
        caption_text = item.caption if item.caption else "(no caption)"
        print(f"\n[{index}] {item.url}")
        if item.timestamp:
            print(f"    {item.timestamp}")
        print(f"    {caption_text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch captions from the last N posts on public Instagram profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py "https://www.instagram.com/nasa/" --count 10\n'
            "  python main.py nasa --count 10 --json\n"
            "  python main.py --file accounts.txt --count 10 --delay 45"
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
            accounts = InstagramProfileParser.read_accounts_file(args.file)
        else:
            accounts = [InstagramProfileParser.parse(args.profile)]
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    config = ScraperConfig.from_env(
        count=args.count,
        max_retries=args.retries,
        delay=args.delay,
        headless=not args.no_headless,
    )

    with InstagramScraper(config) as scraper:
        results = scraper.scrape_profiles(accounts)

    for result in results:
        print_result(result, as_json=args.json)

    if any(not result.succeeded for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
