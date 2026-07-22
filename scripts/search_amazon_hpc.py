#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from amazon_scraper_common import (
    DEFAULT_HEADERS,
    amazon_search_query_candidates,
    amazon_search_url,
    append_to_csv,
    build_not_found_placeholder,
    build_source_merge_key,
    choose_rows_for_pending_only,
    extract_book_details_from_soup,
    load_title_rows,
    merge_input_columns_into_book,
    normalized_title_value,
    rate_limit_sleep,
)


def first_result_url_from_search_html(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    for result in soup.select("div[data-component-type='s-search-result']"):
        link = result.select_one("h2 a") or result.select_one("a[href*='/dp/']")
        if link and link.get("href"):
            href = link["href"]
            if href.startswith("/"):
                href = "https://www.amazon.com" + href
            if "/dp/" in href:
                return href.split("?")[0].split("ref=")[0]
        asin = result.get("data-asin")
        if asin:
            return f"https://www.amazon.com/dp/{asin}"
    return None


def fetch_html(session: requests.Session, url: str, timeout: int) -> str:
    resp = session.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def search_amazon_book(session: requests.Session, search_query: str, timeout: int) -> Optional[str]:
    html = fetch_html(session, amazon_search_url(search_query), timeout=timeout)
    return first_result_url_from_search_html(html)


def get_book_details(session: requests.Session, book_url: str, timeout: int):
    html = fetch_html(session, book_url, timeout=timeout)
    soup = BeautifulSoup(html, "html.parser")
    return extract_book_details_from_soup(soup, book_url=book_url)


def process_book_titles(
    input_csv: Path,
    output_csv: Path,
    max_books: Optional[int],
    skip_rows: int,
    pending_only: bool,
    sleep: float,
    timeout: int,
) -> int:
    df_titles, title_column = load_title_rows(input_csv, skip_rows=skip_rows)
    if pending_only:
        df_titles = choose_rows_for_pending_only(df_titles, output_csv, title_column)
    session = requests.Session()
    processed = 0
    failed = 0
    skipped = 0
    for idx, row in df_titles.iterrows():
        if max_books is not None and (processed + failed) >= max_books:
            break
        book_title = normalized_title_value(row.get(title_column))
        if len(book_title) < 3:
            skipped += 1
            continue
        story_title = normalized_title_value(row.get("story_title"))
        author = normalized_title_value(row.get("author"))
        video_title = normalized_title_value(row.get("video_title"))
        queries = amazon_search_query_candidates(book_title, story_title, author, video_title)
        primary_label = queries[0] if queries else book_title
        try:
            book_url = None
            search_query_used = None
            for query in queries:
                book_url = search_amazon_book(session, query, timeout=timeout)
                if book_url:
                    search_query_used = query
                    break
                rate_limit_sleep(sleep)
            if not book_url:
                append_to_csv([build_not_found_placeholder(row, book_title, primary_label, title_column)], output_csv)
                failed += 1
                continue
            details = get_book_details(session, book_url, timeout=timeout)
            details["extracted_book_title"] = book_title
            details["amazon_search_query"] = search_query_used or primary_label
            details["amazon_match_status"] = "FOUND"
            details["source_merge_key"] = build_source_merge_key(row, title_column, fallback_title=book_title)
            merge_input_columns_into_book(details, row)
            append_to_csv([details], output_csv)
            processed += 1
        except Exception:
            append_to_csv([build_not_found_placeholder(row, book_title, primary_label, title_column)], output_csv)
            failed += 1
        rate_limit_sleep(sleep)
    print(f"Processed={processed} Failed={failed} Skipped={skipped} Output={output_csv}")
    return 0 if processed > 0 or skipped > 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="HPC Amazon scraper using requests/BeautifulSoup.")
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--max-books", type=int, default=None)
    ap.add_argument("--skip-rows", type=int, default=0)
    ap.add_argument("--pending-only", action="store_true")
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()
    raise SystemExit(
        process_book_titles(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            max_books=args.max_books,
            skip_rows=args.skip_rows,
            pending_only=args.pending_only,
            sleep=args.sleep,
            timeout=args.timeout,
        )
    )


if __name__ == "__main__":
    main()
