#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.amazon_scraper_helpers import (
    asin_from_book_url,
    amazon_search_query_candidates,
    amazon_search_url,
    append_to_csv,
    build_not_found_placeholder,
    build_source_merge_key,
    choose_rows_for_pending_only,
    extract_book_details_from_soup,
    jittered_sleep,
    load_cookie_payload,
    load_title_rows,
    load_master_csv,
    merge_input_columns_into_book,
    normalized_title_value,
    score_amazon_result_text,
    save_master_csv,
    update_master_with_amazon_result,
)


def log_progress(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def cache_key_for_row(book_title: str, story_title: str, author: str) -> Tuple[str, str, str]:
    return (
        normalized_title_value(book_title).casefold(),
        normalized_title_value(story_title).casefold(),
        normalized_title_value(author).casefold(),
    )


def clone_cached_result(template: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(template)


def setup_driver(
    headless: bool = True,
    cookie_json: Optional[Path] = None,
    chrome_binary: Optional[str] = None,
    chromedriver_binary: Optional[str] = None,
) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    if chrome_binary:
        options.binary_location = chrome_binary
    service = Service(executable_path=chromedriver_binary) if chromedriver_binary else None
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(5)
    driver.set_page_load_timeout(30)
    if cookie_json is not None:
        driver.get("https://www.amazon.com/")
        for cookie in load_cookie_payload(cookie_json):
            selenium_cookie = {
                "name": cookie["name"],
                "value": cookie["value"],
                "domain": cookie.get("domain") or ".amazon.com",
                "path": cookie.get("path") or "/",
                "secure": bool(cookie.get("secure", False)),
            }
            expiry = cookie.get("expiry")
            if expiry not in {None, "", "None"}:
                try:
                    selenium_cookie["expiry"] = int(expiry)
                except Exception:
                    pass
            try:
                driver.add_cookie(selenium_cookie)
            except Exception:
                continue
        driver.get("https://www.amazon.com/")
    return driver


def search_amazon_book(
    driver: webdriver.Chrome,
    search_query: str,
    book_title: str,
    story_title: str = "",
    author: str = "",
) -> Optional[str]:
    driver.get(amazon_search_url(search_query))
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-component-type='s-search-result']"))
    )
    results = driver.find_elements(By.CSS_SELECTOR, "div[data-component-type='s-search-result']")
    if not results:
        return None
    ranked: list[tuple[float, str]] = []
    for result in results[:10]:
        try:
            title_el = result.find_element(By.CSS_SELECTOR, "h2")
            result_title = title_el.text.strip()
        except Exception:
            result_title = ""
        try:
            subtext = result.text[:1000]
        except Exception:
            subtext = result_title
        href = None
        for selector in ["h2 a", "a[href*='/dp/']"]:
            try:
                href = result.find_element(By.CSS_SELECTOR, selector).get_attribute("href")
                if href and "/dp/" in href:
                    break
            except Exception:
                continue
        if not href:
            asin = result.get_attribute("data-asin")
            href = f"https://www.amazon.com/dp/{asin}" if asin else None
        if not href:
            continue
        clean_href = href.split("?")[0].split("ref=")[0]
        score = score_amazon_result_text(
            result_title=result_title,
            result_subtext=subtext,
            expected_title=book_title,
            expected_story_title=story_title,
            expected_author=author,
        )
        ranked.append((score, clean_href))
    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]
    return None


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
    sleep_jitter: float,
    headless: bool,
    status_column: Optional[str],
    cookie_json: Optional[Path],
    chrome_binary: Optional[str],
    chromedriver_binary: Optional[str],
    allow_existing_input_overlap: bool,
) -> int:
    df_titles, title_column = load_title_rows(input_csv, skip_rows=skip_rows, status_column=status_column)
    if pending_only:
        df_titles = choose_rows_for_pending_only(df_titles, output_csv, title_column)
    completed_video_ids: set[str] = set()
    if output_csv.exists() and "video_id" in df_titles.columns:
        existing_output = pd.read_csv(output_csv, low_memory=False)
        if "video_id" in existing_output.columns:
            input_ids = df_titles["video_id"].fillna("").astype(str).str.strip()
            completed_video_ids = set(existing_output["video_id"].fillna("").astype(str).str.strip())
            overlap_mask = input_ids.isin(completed_video_ids)
            overlap_count = int(overlap_mask.sum())
            if overlap_count:
                sample_ids = input_ids[overlap_mask].head(10).tolist()
                message = (
                    f"Input CSV contains {overlap_count} video_id rows already present in {output_csv}. "
                    f"These rows will be skipped automatically. Sample overlapping video_ids: {sample_ids}"
                )
                if not allow_existing_input_overlap:
                    log_progress(f"WARNING: {message}")
                else:
                    log_progress(f"WARNING: {message}")
                df_titles = df_titles.loc[~overlap_mask].copy().reset_index(drop=True)
    master_df = load_master_csv(master_csv)
    result_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    driver = setup_driver(
        headless=headless,
        cookie_json=cookie_json,
        chrome_binary=chrome_binary,
        chromedriver_binary=chromedriver_binary,
    )
    processed = 0
    failed = 0
    skipped = 0
    total_rows = len(df_titles)
    try:
        for row_num, (_, row) in enumerate(df_titles.iterrows(), start=1):
            if max_books is not None and (processed + failed) >= max_books:
                break
            book_title = normalized_title_value(row.get(title_column))
            video_id = normalized_title_value(row.get("video_id"))
            if video_id and video_id in completed_video_ids:
                log_progress(
                    f"skip row {row_num}/{total_rows} video_id={video_id} reason=already_written_output"
                )
                skipped += 1
                continue
            if len(book_title) < 3:
                log_progress(
                    f"skip row {row_num}/{total_rows} video_id={video_id or '[NA]'} reason=short_book_title"
                )
                skipped += 1
                continue
            story_title = normalized_title_value(row.get("story_title"))
            author = normalized_title_value(row.get("author"))
            video_title = normalized_title_value(row.get("video_title"))
            queries = amazon_search_query_candidates(book_title, story_title, author, video_title)
            primary_label = queries[0] if queries else book_title
            cache_key = cache_key_for_row(book_title, story_title, author)
            log_progress(
                f"start row {row_num}/{total_rows} video_id={video_id or '[NA]'} "
                f"book_title={book_title!r} story_title={story_title or '[NA]'} "
                f"author={author or '[NA]'} queries={len(queries)}"
            )
            try:
                cached = result_cache.get(cache_key)
                if cached is not None:
                    result = clone_cached_result(cached)
                    result["extracted_book_title"] = book_title
                    result["amazon_scrape_run"] = output_csv.stem
                    result["source_merge_key"] = build_source_merge_key(row, title_column, fallback_title=book_title)
                    merge_input_columns_into_book(result, row)
                    new_rows, updated_rows = append_to_csv([result], output_csv)
                    update_master_with_amazon_result(master_df, row, result)
                    if video_id and (new_rows > 0 or updated_rows > 0):
                        completed_video_ids.add(video_id)
                    log_progress(
                        f"saved row {row_num}/{total_rows} video_id={video_id or '[NA]'} "
                        f"via=cache status={result.get('amazon_match_status','[NA]')} "
                        f"age={result.get('reading_age','[NA]')} new_rows={new_rows} updated_rows={updated_rows}"
                    )
                    if result.get("amazon_match_status") == "FOUND":
                        processed += 1
                    else:
                        failed += 1
                    continue
                book_url = None
                search_query_used = None
                for query in queries:
                    log_progress(
                        f"search row {row_num}/{total_rows} video_id={video_id or '[NA]'} query={query!r}"
                    )
                    book_url = search_amazon_book(
                        driver,
                        query,
                        book_title=book_title,
                        story_title=story_title,
                        author=author,
                    )
                    if book_url:
                        search_query_used = query
                        log_progress(
                            f"match row {row_num}/{total_rows} video_id={video_id or '[NA]'} "
                            f"query={query!r} book_url={book_url}"
                        )
                        break
                    jittered_sleep(sleep, sleep_jitter)
                if not book_url:
                    not_found = build_not_found_placeholder(row, book_title, primary_label, title_column)
                    not_found["amazon_scrape_run"] = output_csv.stem
                    result_cache[cache_key] = {
                        "amazon_match_status": not_found.get("amazon_match_status", "NOT_FOUND"),
                        "amazon_search_query": not_found.get("amazon_search_query", primary_label),
                        "title": not_found.get("title", f"[NO_AMAZON_MATCH] {book_title}"),
                        "book_url": not_found.get("book_url", "N/A"),
                        "reading_age": not_found.get("reading_age", "N/A"),
                        "reading_age_source": not_found.get("reading_age_source", "N/A"),
                        "reading_age_customer": not_found.get("reading_age_customer", "N/A"),
                        "reading_age_publisher": not_found.get("reading_age_publisher", "N/A"),
                    }
                    new_rows, updated_rows = append_to_csv([not_found], output_csv)
                    update_master_with_amazon_result(master_df, row, not_found)
                    if video_id and (new_rows > 0 or updated_rows > 0):
                        completed_video_ids.add(video_id)
                    log_progress(
                        f"saved row {row_num}/{total_rows} video_id={video_id or '[NA]'} "
                        f"status=NOT_FOUND new_rows={new_rows} updated_rows={updated_rows}"
                    )
                    failed += 1
                    continue
                details = get_book_details(driver, book_url)
                details["extracted_book_title"] = book_title
                details["amazon_search_query"] = search_query_used or primary_label
                details["amazon_match_status"] = "FOUND"
                details["amazon_scrape_run"] = output_csv.stem
                details["source_merge_key"] = build_source_merge_key(row, title_column, fallback_title=book_title)
                result_cache[cache_key] = {
                    key: copy.deepcopy(value)
                    for key, value in details.items()
                    if key != "source_merge_key"
                }
                merge_input_columns_into_book(details, row)
                new_rows, updated_rows = append_to_csv([details], output_csv)
                update_master_with_amazon_result(master_df, row, details)
                if video_id and (new_rows > 0 or updated_rows > 0):
                    completed_video_ids.add(video_id)
                log_progress(
                    f"saved row {row_num}/{total_rows} video_id={video_id or '[NA]'} "
                    f"status=FOUND age={details.get('reading_age','[NA]')} "
                    f"age_source={details.get('reading_age_source','[NA]')} "
                    f"new_rows={new_rows} updated_rows={updated_rows}"
                )
                processed += 1
            except Exception as exc:
                not_found = build_not_found_placeholder(row, book_title, primary_label, title_column)
                not_found["amazon_scrape_run"] = output_csv.stem
                result_cache[cache_key] = {
                    "amazon_match_status": not_found.get("amazon_match_status", "NOT_FOUND"),
                    "amazon_search_query": not_found.get("amazon_search_query", primary_label),
                    "title": not_found.get("title", f"[NO_AMAZON_MATCH] {book_title}"),
                    "book_url": not_found.get("book_url", "N/A"),
                    "reading_age": not_found.get("reading_age", "N/A"),
                    "reading_age_source": not_found.get("reading_age_source", "N/A"),
                    "reading_age_customer": not_found.get("reading_age_customer", "N/A"),
                    "reading_age_publisher": not_found.get("reading_age_publisher", "N/A"),
                }
                new_rows, updated_rows = append_to_csv([not_found], output_csv)
                update_master_with_amazon_result(master_df, row, not_found)
                if video_id and (new_rows > 0 or updated_rows > 0):
                    completed_video_ids.add(video_id)
                log_progress(
                    f"error row {row_num}/{total_rows} video_id={video_id or '[NA]'} "
                    f"exc={type(exc).__name__}: {exc} fallback_status=NOT_FOUND "
                    f"new_rows={new_rows} updated_rows={updated_rows}"
                )
                failed += 1
            jittered_sleep(sleep, sleep_jitter)
    finally:
        driver.quit()
    save_master_csv(master_df, master_csv)
    log_progress(f"Processed={processed} Failed={failed} Skipped={skipped} Output={output_csv}")
    return 0 if processed > 0 or skipped > 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="PC Amazon scraper using Selenium/Chrome.")
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--master-csv", type=Path, default=Path("data/processed/youtube_catalog.csv.gz"))
    ap.add_argument("--max-books", type=int, default=None)
    ap.add_argument("--skip-rows", type=int, default=0)
    ap.add_argument("--pending-only", action="store_true")
    ap.add_argument("--status-column", type=str, default="ready_for_amazon_search")
    ap.add_argument("--sleep", type=float, default=5.0)
    ap.add_argument("--sleep-jitter", type=float, default=0.0)
    ap.add_argument("--cookie-json", type=Path, default=None)
    ap.add_argument("--chrome-binary", type=str, default=None)
    ap.add_argument("--chromedriver-binary", type=str, default=None)
    ap.add_argument("--show-browser", action="store_true")
    ap.add_argument("--allow-existing-input-overlap", action="store_true")
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
            sleep_jitter=args.sleep_jitter,
            headless=not args.show_browser,
            status_column=args.status_column,
            cookie_json=args.cookie_json,
            chrome_binary=args.chrome_binary,
            chromedriver_binary=args.chromedriver_binary,
            allow_existing_input_overlap=args.allow_existing_input_overlap,
        )
    )


if __name__ == "__main__":
    main()
