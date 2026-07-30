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
    asin_from_book_url,
    amazon_search_query_candidates,
    amazon_search_url,
    append_to_csv,
    build_not_found_placeholder,
    build_source_merge_key,
    choose_rows_for_pending_only,
    extract_book_details_from_soup,
    load_title_rows,
    load_master_csv,
    merge_input_columns_into_book,
    normalized_title_value,
    rate_limit_sleep,
    save_master_csv,
    update_master_with_amazon_result,
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
    try:
        read_more_buttons = driver.find_elements(By.XPATH, "//a[contains(text(),'Read more')]")
        if read_more_buttons:
            read_more_buttons[0].click()
            time.sleep(1)
    except Exception:
        pass
    try:
        for selector in ["#reviews-medley", "#cr-original-review-content", "[data-hook='review']", "#cm_cr-review_list"]:
            try:
                elem = driver.find_element(By.CSS_SELECTOR, selector)
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
                time.sleep(2)
                break
            except Exception:
                continue
    except Exception:
        pass
    soup = BeautifulSoup(driver.page_source, "html.parser")
    review_soups = []
    asin = asin_from_book_url(book_url)
    if asin:
        try:
            review_url = f"https://www.amazon.com/product-reviews/{asin}/?reviewerType=all_reviews"
            driver.get(review_url)
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-hook='review'], #cm_cr-review_list, #cr-filter-info-review-rating-count"))
            )
            time.sleep(2)
            review_soups.append(BeautifulSoup(driver.page_source, "html.parser"))
        except Exception:
            pass
        finally:
            driver.get(book_url)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "productTitle")))
            time.sleep(1)
            soup = BeautifulSoup(driver.page_source, "html.parser")
    return extract_book_details_from_soup(soup, book_url=book_url, review_soups=review_soups)


def process_book_titles(
    input_csv: Path,
    output_csv: Path,
    master_csv: Optional[Path],
    max_books: Optional[int],
    skip_rows: int,
    pending_only: bool,
    sleep: float,
    headless: bool,
    status_column: Optional[str],
) -> int:
    df_titles, title_column = load_title_rows(input_csv, skip_rows=skip_rows, status_column=status_column)
    if pending_only:
        df_titles = choose_rows_for_pending_only(df_titles, output_csv, title_column)
    master_df = load_master_csv(master_csv)
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
                    not_found = build_not_found_placeholder(row, book_title, primary_label, title_column)
                    append_to_csv([not_found], output_csv)
                    update_master_with_amazon_result(master_df, row, not_found)
                    failed += 1
                    continue
                details = get_book_details(driver, book_url)
                details["extracted_book_title"] = book_title
                details["amazon_search_query"] = search_query_used or primary_label
                details["amazon_match_status"] = "FOUND"
                details["source_merge_key"] = build_source_merge_key(row, title_column, fallback_title=book_title)
                merge_input_columns_into_book(details, row)
                append_to_csv([details], output_csv)
                update_master_with_amazon_result(master_df, row, details)
                processed += 1
            except Exception:
                not_found = build_not_found_placeholder(row, book_title, primary_label, title_column)
                append_to_csv([not_found], output_csv)
                update_master_with_amazon_result(master_df, row, not_found)
                failed += 1
            rate_limit_sleep(sleep)
    finally:
        driver.quit()
    save_master_csv(master_df, master_csv)
    print(f"Processed={processed} Failed={failed} Skipped={skipped} Output={output_csv}")
    return 0 if processed > 0 or skipped > 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="PC Amazon scraper using Selenium/Chrome.")
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--master-csv", type=Path, default=None)
    ap.add_argument("--max-books", type=int, default=None)
    ap.add_argument("--skip-rows", type=int, default=0)
    ap.add_argument("--pending-only", action="store_true")
    ap.add_argument("--status-column", type=str, default="ready_for_amazon_search")
    ap.add_argument("--sleep", type=float, default=5.0)
    ap.add_argument("--show-browser", action="store_true")
    args = ap.parse_args()
    raise SystemExit(
        process_book_titles(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            master_csv=args.master_csv,
            max_books=args.max_books,
            skip_rows=args.skip_rows,
            pending_only=args.pending_only,
            sleep=args.sleep,
            headless=not args.show_browser,
            status_column=args.status_column,
        )
    )


if __name__ == "__main__":
    main()
