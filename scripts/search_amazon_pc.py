#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from amazon_scraper_common import (
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


def setup_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(5)
    driver.set_page_load_timeout(30)
    return driver


def search_amazon_book(driver: webdriver.Chrome, search_query: str) -> Optional[str]:
    driver.get(amazon_search_url(search_query))
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-component-type='s-search-result']"))
    )
    results = driver.find_elements(By.CSS_SELECTOR, "div[data-component-type='s-search-result']")
    if not results:
        return None
    first = results[0]
    for selector in ["h2 a", "a[href*='/dp/']"]:
        try:
            href = first.find_element(By.CSS_SELECTOR, selector).get_attribute("href")
            if href and "/dp/" in href:
                return href.split("?")[0].split("ref=")[0]
        except Exception:
            continue
    asin = first.get_attribute("data-asin")
    return f"https://www.amazon.com/dp/{asin}" if asin else None


def get_book_details(driver: webdriver.Chrome, book_url: str):
    driver.get(book_url)
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "productTitle")))
    soup = BeautifulSoup(driver.page_source, "html.parser")
    return extract_book_details_from_soup(soup, book_url=book_url)


def process_book_titles(
    input_csv: Path,
    output_csv: Path,
    max_books: Optional[int],
    skip_rows: int,
    pending_only: bool,
    sleep: float,
    headless: bool,
) -> int:
    df_titles, title_column = load_title_rows(input_csv, skip_rows=skip_rows)
    if pending_only:
        df_titles = choose_rows_for_pending_only(df_titles, output_csv, title_column)
    driver = setup_driver(headless=headless)
    processed = 0
    failed = 0
    skipped = 0
    try:
        for _, row in df_titles.iterrows():
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
                    book_url = search_amazon_book(driver, query)
                    if book_url:
                        search_query_used = query
                        break
                    rate_limit_sleep(sleep)
                if not book_url:
                    append_to_csv([build_not_found_placeholder(row, book_title, primary_label, title_column)], output_csv)
                    failed += 1
                    continue
                details = get_book_details(driver, book_url)
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
    finally:
        driver.quit()
    print(f"Processed={processed} Failed={failed} Skipped={skipped} Output={output_csv}")
    return 0 if processed > 0 or skipped > 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="PC Amazon scraper using Selenium/Chrome.")
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--max-books", type=int, default=None)
    ap.add_argument("--skip-rows", type=int, default=0)
    ap.add_argument("--pending-only", action="store_true")
    ap.add_argument("--sleep", type=float, default=5.0)
    ap.add_argument("--show-browser", action="store_true")
    args = ap.parse_args()
    raise SystemExit(
        process_book_titles(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            max_books=args.max_books,
            skip_rows=args.skip_rows,
            pending_only=args.pending_only,
            sleep=args.sleep,
            headless=not args.show_browser,
        )
    )


if __name__ == "__main__":
    main()
