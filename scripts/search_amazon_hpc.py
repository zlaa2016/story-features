#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

from amazon_scraper_common import (
    DEFAULT_HEADERS,
    asin_from_book_url,
    amazon_search_query_candidates,
    amazon_search_url,
    append_to_csv,
    build_not_found_placeholder,
    build_pending_placeholder,
    build_source_merge_key,
    choose_rows_for_pending_only,
    extract_book_details_from_soup,
    load_title_rows,
    load_master_csv,
    merge_input_columns_into_book,
    normalized_title_value,
    rate_limit_sleep,
    save_master_csv,
    simplify_video_title_for_query,
    update_master_with_amazon_result,
)


POSITIVE_HINTS = {
    "children",
    "childrens",
    "kids",
    "kid",
    "picture book",
    "board book",
    "little golden",
    "beginner readers",
    "readers",
    "read aloud",
    "storybook",
    "storybooks",
    "paw patrol",
    "disney",
}

NEGATIVE_HINTS = {
    "romance",
    "regency",
    "duke",
    "billionaire",
    "polyamory",
    "erotica",
    "new adult",
    "college romance",
    "motor sports",
    "motorcycle",
    "driving",
    "sports & outdoors",
    "contemporary fiction",
    "historical romance",
    "victorian",
    "kindle store > kindle ebooks > romance",
}

STOPWORDS = {"the", "a", "an", "and", "of", "to", "for", "in", "on", "with", "book", "series"}


class AmazonTemporaryBlockError(RuntimeError):
    pass


def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    return {token for token in normalize_for_match(text).split() if token and token not in STOPWORDS}


def overlap_score(query: str, candidate: str) -> float:
    query_tokens = token_set(query)
    cand_tokens = token_set(candidate)
    if not query_tokens or not cand_tokens:
        return 0.0
    return len(query_tokens & cand_tokens) / max(1, len(query_tokens))


def title_contains_phrase(query: str, candidate: str) -> bool:
    norm_query = normalize_for_match(query)
    norm_candidate = normalize_for_match(candidate)
    return bool(norm_query and norm_query in norm_candidate)


def parse_search_results(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict[str, str]] = []
    for result in soup.select("div[data-component-type='s-search-result']"):
        title_node = result.select_one("h2 span") or result.select_one("h2 a")
        title = title_node.get_text(" ", strip=True) if title_node else ""
        link = result.select_one("h2 a") or result.select_one("a[href*='/dp/']")
        href = ""
        if link and link.get("href"):
            href = link["href"]
            if href.startswith("/"):
                href = "https://www.amazon.com" + href
            if "/dp/" in href:
                href = href.split("?")[0].split("ref=")[0]
        asin = result.get("data-asin")
        if not href and asin:
            href = f"https://www.amazon.com/dp/{asin}"
        byline = " ".join(node.get_text(" ", strip=True) for node in result.select("div.a-row.a-size-base.a-color-secondary"))
        whole_text = result.get_text(" ", strip=True)
        if href:
            rows.append({"title": title, "url": href, "byline": byline, "text": whole_text})
    return rows


def score_result(
    result: Dict[str, str],
    book_title: str,
    story_title: str,
    author: str,
    video_title: str,
) -> Tuple[float, List[str]]:
    title = result["title"]
    blob = f"{result.get('title', '')} {result.get('byline', '')} {result.get('text', '')}"
    score = 0.0
    reasons: List[str] = []

    book_overlap = overlap_score(book_title, title)
    story_overlap = overlap_score(story_title, title) if story_title else 0.0
    video_overlap = overlap_score(video_title, title) if video_title else 0.0
    author_overlap = overlap_score(author, blob) if author else 0.0

    score += 4.0 * book_overlap
    if book_overlap:
        reasons.append(f"book={book_overlap:.2f}")
    score += 3.0 * story_overlap
    if story_overlap:
        reasons.append(f"story={story_overlap:.2f}")
    score += 1.5 * video_overlap
    if video_overlap:
        reasons.append(f"video={video_overlap:.2f}")
    score += 2.5 * author_overlap
    if author_overlap:
        reasons.append(f"author={author_overlap:.2f}")

    if normalize_for_match(book_title) == normalize_for_match(title):
        score += 5.0
        reasons.append("book_exact")
    if story_title and normalize_for_match(story_title) == normalize_for_match(title):
        score += 4.0
        reasons.append("story_exact")

    if title_contains_phrase(book_title, title):
        score += 2.0
        reasons.append("book_phrase")
    if story_title and title_contains_phrase(story_title, title):
        score += 1.5
        reasons.append("story_phrase")

    blob_norm = normalize_for_match(blob)
    for hint in POSITIVE_HINTS:
        if hint in blob_norm:
            score += 0.75
            reasons.append(f"+{hint}")
    for hint in NEGATIVE_HINTS:
        if hint in blob_norm:
            score -= 3.0
            reasons.append(f"-{hint}")

    if book_overlap < 0.34 and story_overlap < 0.34 and not title_contains_phrase(book_title, title):
        score -= 3.5
        reasons.append("weak_title_match")
    if author and author_overlap == 0 and book_overlap < 0.6 and story_overlap < 0.6:
        score -= 1.0
        reasons.append("no_author_support")
    if "kindle store" in blob_norm and "children" not in blob_norm and "kids" not in blob_norm:
        score -= 1.5
        reasons.append("generic_kindle_penalty")

    return score, reasons


def looks_like_amazon_block_page(html: str) -> bool:
    blob = html.lower()
    checks = [
        "sorry, we just need to make sure you're not a robot",
        "enter the characters you see below",
        "type the characters you see in this image",
        "validatecaptcha",
        "api-services-support@amazon.com",
    ]
    return any(token in blob for token in checks)


def fetch_html(session: requests.Session, url: str, timeout: int, sleep: float, max_attempts: int = 4) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            resp.raise_for_status()
            html = resp.text
            if looks_like_amazon_block_page(html):
                raise AmazonTemporaryBlockError(f"Amazon temporary block detected for {url}")
            return html
        except AmazonTemporaryBlockError as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            rate_limit_sleep(max(sleep * (3 ** attempt), 15.0))
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            rate_limit_sleep(max(sleep * (2 ** attempt), 10.0))
    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to fetch {url}")


def fetch_optional_soup(session: requests.Session, url: str, timeout: int) -> Optional[BeautifulSoup]:
    try:
        html = fetch_html(session, url, timeout=timeout, sleep=5.0)
        return BeautifulSoup(html, "html.parser")
    except Exception:
        return None


def search_amazon_book(
    session: requests.Session,
    search_query: str,
    timeout: int,
    sleep: float,
    book_title: str,
    story_title: str,
    author: str,
    video_title: str,
) -> Tuple[Optional[str], Optional[str]]:
    html = fetch_html(session, amazon_search_url(search_query), timeout=timeout, sleep=sleep)
    results = parse_search_results(html)
    if not results:
        return None, None
    ranked: List[Tuple[float, Dict[str, str], List[str]]] = []
    for result in results[:12]:
        score, reasons = score_result(result, book_title, story_title, author, video_title)
        ranked.append((score, result, reasons))
    ranked.sort(key=lambda row: row[0], reverse=True)
    best_score, best_result, _reasons = ranked[0]
    strong_phrase = title_contains_phrase(book_title, best_result["title"]) or (
        story_title and title_contains_phrase(story_title, best_result["title"])
    )
    exactish_match = normalize_for_match(book_title) == normalize_for_match(best_result["title"]) or (
        story_title and normalize_for_match(story_title) == normalize_for_match(best_result["title"])
    )
    if best_score < 1.5 and not strong_phrase and not exactish_match:
        return None, None
    return best_result["url"], best_result["title"]


def get_book_details(session: requests.Session, book_url: str, timeout: int, sleep: float):
    html = fetch_html(session, book_url, timeout=timeout, sleep=sleep)
    soup = BeautifulSoup(html, "html.parser")
    review_soups: List[BeautifulSoup] = []
    asin = asin_from_book_url(book_url)
    if asin:
        review_url = f"https://www.amazon.com/product-reviews/{asin}/?reviewerType=all_reviews"
        review_soup = fetch_optional_soup(session, review_url, timeout=timeout)
        if review_soup is not None:
            review_soups.append(review_soup)
    return extract_book_details_from_soup(soup, book_url=book_url, review_soups=review_soups)


def process_book_titles(
    input_csv: Path,
    output_csv: Path,
    master_csv: Optional[Path],
    max_books: Optional[int],
    skip_rows: int,
    pending_only: bool,
    sleep: float,
    timeout: int,
    status_column: Optional[str],
) -> int:
    df_titles, title_column = load_title_rows(input_csv, skip_rows=skip_rows, status_column=status_column)
    if pending_only:
        df_titles = choose_rows_for_pending_only(df_titles, output_csv, title_column)
    master_df = load_master_csv(master_csv)
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
        video_title = simplify_video_title_for_query(normalized_title_value(row.get("video_title")))
        queries = amazon_search_query_candidates(book_title, story_title, author, video_title)
        primary_label = queries[0] if queries else book_title
        try:
            book_url = None
            search_query_used = None
            matched_title = None
            for query in queries:
                book_url, matched_title = search_amazon_book(
                    session,
                    query,
                    timeout=timeout,
                    sleep=sleep,
                    book_title=book_title,
                    story_title=story_title,
                    author=author,
                    video_title=video_title,
                )
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
            details = get_book_details(session, book_url, timeout=timeout, sleep=sleep)
            matched_overlap = max(
                overlap_score(book_title, details.get("title", "")),
                overlap_score(story_title, details.get("title", "")) if story_title else 0.0,
            )
            details_blob = f"{details.get('categories','')} {details.get('summary','')} {details.get('title','')}".lower()
            if (matched_overlap < 0.20 and not title_contains_phrase(book_title, details.get("title", ""))) or any(
                hint in details_blob for hint in NEGATIVE_HINTS
            ):
                not_found = build_not_found_placeholder(row, book_title, primary_label, title_column)
                append_to_csv([not_found], output_csv)
                update_master_with_amazon_result(master_df, row, not_found)
                failed += 1
                rate_limit_sleep(sleep)
                continue
            details["extracted_book_title"] = book_title
            details["amazon_search_query"] = search_query_used or primary_label
            details["amazon_match_status"] = "FOUND"
            details["source_merge_key"] = build_source_merge_key(row, title_column, fallback_title=book_title)
            if matched_title:
                details["matched_search_title"] = matched_title
            merge_input_columns_into_book(details, row)
            append_to_csv([details], output_csv)
            update_master_with_amazon_result(master_df, row, details)
            processed += 1
        except AmazonTemporaryBlockError:
            pending = build_pending_placeholder(row, book_title, primary_label, title_column, reason="TEMPORARY_BLOCK")
            append_to_csv([pending], output_csv)
            update_master_with_amazon_result(master_df, row, pending)
            skipped += 1
        except Exception:
            not_found = build_not_found_placeholder(row, book_title, primary_label, title_column)
            append_to_csv([not_found], output_csv)
            update_master_with_amazon_result(master_df, row, not_found)
            failed += 1
        rate_limit_sleep(sleep)
    save_master_csv(master_df, master_csv)
    print(f"Processed={processed} Failed={failed} Skipped={skipped} Output={output_csv}")
    return 0 if processed > 0 or skipped > 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="HPC Amazon scraper using requests/BeautifulSoup.")
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--master-csv", type=Path, default=None)
    ap.add_argument("--max-books", type=int, default=None)
    ap.add_argument("--skip-rows", type=int, default=0)
    ap.add_argument("--pending-only", action="store_true")
    ap.add_argument("--status-column", type=str, default="ready_for_amazon_search")
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--timeout", type=int, default=30)
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
            timeout=args.timeout,
            status_column=args.status_column,
        )
    )


if __name__ == "__main__":
    main()
