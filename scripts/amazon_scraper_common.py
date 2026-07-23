#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup


OUTPUT_COLUMN_MAPPING = {
    "title": "Title",
    "extracted_book_title": "Extracted Book Title",
    "amazon_search_query": "Amazon Search Query",
    "amazon_match_status": "Amazon Match Status",
    "source_merge_key": "Source Merge Key",
    "author": "Author",
    "publisher": "Publisher",
    "published_date": "Published Date",
    "categories": "Categories",
    "summary": "Summary",
    "book_url": "Book URL",
    "reading_age": "Reading Age",
    "reading_age_source": "Reading Age Source",
    "reading_age_customer": "Reading Age Customer",
    "reading_age_publisher": "Reading Age Publisher",
    "grade_level": "Grade Level",
    "print_length": "Print Length",
    "isbn_10": "ISBN-10",
    "isbn_13": "ISBN-13",
    "language": "Language",
    "format_binding": "Format",
    "dimensions": "Dimensions",
    "item_weight": "Item Weight",
    "series": "Series",
    "average_rating": "Average Rating",
    "num_reviews": "Num Reviews",
    "price": "Price",
    "customer_reviews_text": "Review Summary",
    "editorial_reviews_text": "Editorial Reviews",
    "about_author": "About Author",
    "customer_reviews_json": "Customer Reviews",
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def read_csv_any_encoding(path: Path) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except Exception as exc:
            last_err = exc
    if last_err:
        raise last_err
    return pd.read_csv(path, low_memory=False)


def _serialize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def normalize_label(label: str) -> str:
    clean = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", str(label))
    return re.sub(r"\s+", " ", clean).strip().lower().rstrip(":").strip()


def clean_text(text: Optional[str]) -> str:
    if not text:
        return "N/A"
    out = text.strip()
    out = re.sub(r"\s+", " ", out)
    return out or "N/A"


def metadata_key_to_column(label: str) -> str:
    stripped = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", str(label))
    clean = re.sub(r"\s+", " ", stripped).strip()
    return clean.title() if clean else "Unknown"


def metadata_should_skip(label: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()
    return normalized in {"age range", "target age", "customer reviews"}


def passthrough_csv_column_name(key: str) -> str:
    return key


def merge_input_columns_into_book(book_details: Dict[str, Any], row: pd.Series) -> None:
    keys_in_book = set(book_details.keys())
    for col in row.index:
        value = _serialize_cell(row[col])
        key = col
        if key in keys_in_book:
            key = f"extracted_{col}"
            suffix = 1
            while key in keys_in_book:
                key = f"extracted_{col}_{suffix}"
                suffix += 1
        book_details[key] = value
        keys_in_book.add(key)


def build_source_merge_key(row: pd.Series, title_column: str, fallback_title: str = "") -> str:
    extracted_title = _serialize_cell(row.get(title_column, fallback_title)).strip()
    video_id = _serialize_cell(row.get("video_id", "")).strip() if "video_id" in row.index else ""
    return f"{video_id}||{extracted_title}" if video_id else f"NO_VIDEO_ID||{extracted_title}"


def amazon_search_query_candidates(book_title: str, story_title: str, author: str, video_title: str = "") -> List[str]:
    seen = set()

    def add(q: str) -> None:
        q = (q or "").strip()
        if len(q) < 3:
            return
        lowered = q.lower()
        if lowered in seen:
            return
        seen.add(lowered)
        out.append(q)

    out: List[str] = []
    if author:
        add(f"{book_title} {author}")
    else:
        add(book_title)
    if story_title and story_title.lower() not in {"n/a", "nan"}:
        for q in [
            f"{story_title} {author}" if author else story_title,
            story_title,
            f"{book_title} {story_title}",
            f"{story_title} {book_title}",
            f"{book_title} {story_title} {author}" if author else "",
        ]:
            add(q)
    if video_title and video_title.lower() not in {"n/a", "nan"}:
        add(video_title)
        if author:
            add(f"{video_title} {author}")
    return out


def build_not_found_placeholder(row: pd.Series, book_title: str, search_query: str, title_column: str) -> Dict[str, Any]:
    source_id = _serialize_cell(row.get("video_id", "")) if "video_id" in row.index else ""
    marker = f"{book_title}::{source_id}" if source_id else book_title
    payload = {
        "title": f"[NO_AMAZON_MATCH] {marker}",
        "author": "N/A",
        "publisher": "N/A",
        "published_date": "N/A",
        "categories": "N/A",
        "summary": "N/A",
        "book_url": "N/A",
        "reading_age": "N/A",
        "reading_age_source": "N/A",
        "reading_age_customer": "N/A",
        "reading_age_publisher": "N/A",
        "grade_level": "N/A",
        "print_length": "N/A",
        "isbn_10": "N/A",
        "isbn_13": "N/A",
        "language": "N/A",
        "format_binding": "N/A",
        "dimensions": "N/A",
        "item_weight": "N/A",
        "series": "N/A",
        "average_rating": "N/A",
        "num_reviews": "N/A",
        "price": "N/A",
        "customer_reviews_text": "N/A",
        "editorial_reviews_text": "N/A",
        "about_author": "N/A",
        "customer_reviews_json": "N/A",
        "all_metadata": {},
        "extracted_book_title": _serialize_cell(row.get(title_column, book_title)),
        "amazon_search_query": search_query,
        "amazon_match_status": "NOT_FOUND",
        "source_merge_key": build_source_merge_key(row, title_column, fallback_title=book_title),
    }
    merge_input_columns_into_book(payload, row)
    return payload


def extract_product_details_kv(soup: BeautifulSoup) -> Dict[str, str]:
    details: Dict[str, str] = {}
    for table_id in ["productDetailsTable", "productDetails_detailBullets_sections1"]:
        table = soup.find("table", id=table_id)
        if not table:
            continue
        for row in table.find_all("tr"):
            header = row.find("th")
            value = row.find("td")
            if header and value:
                details[normalize_label(header.get_text(" ", strip=True))] = value.get_text(" ", strip=True)
    return details


def extract_detail_bullets_kv(soup: BeautifulSoup) -> Dict[str, str]:
    details: Dict[str, str] = {}
    section = soup.find("div", id="detailBullets_feature_div")
    if not section:
        return details
    for item in section.find_all("li"):
        bold = item.find("span", class_="a-text-bold")
        if not bold:
            continue
        label = normalize_label(bold.get_text(" ", strip=True))
        value = ""
        next_span = bold.find_next_sibling("span")
        if next_span and "a-text-bold" not in (next_span.get("class") or []):
            value = next_span.get_text(" ", strip=True)
        if not value:
            full_text = item.get_text(" ", strip=True)
            value = full_text.replace(bold.get_text(" ", strip=True), "", 1).strip().lstrip(":").strip()
        if value:
            details[label] = value
    return details


def extract_rpi_attributes_kv(soup: BeautifulSoup) -> Dict[str, str]:
    details: Dict[str, str] = {}
    for div in soup.find_all("div", id=re.compile(r"^rpi-attribute-")):
        label_el = div.find("div", class_=re.compile(r"rpi-attribute-label"))
        value_el = div.find("div", class_=re.compile(r"rpi-attribute-value"))
        if label_el and value_el:
            label = normalize_label(label_el.get_text(" ", strip=True))
            value = value_el.get_text(" ", strip=True)
            if label and value:
                details[label] = value
    return details


def extract_all_metadata(soup: BeautifulSoup) -> Dict[str, str]:
    out = extract_product_details_kv(soup)
    for source in [extract_detail_bullets_kv(soup), extract_rpi_attributes_kv(soup)]:
        for key, value in source.items():
            out.setdefault(key, value)
    return out


def parse_reading_age_value(value: str) -> str:
    if not value:
        return "N/A"
    match = re.search(
        r"(\d+\s*[-–—to]\s*\d+\s*years?|\d+\s*\+?\s*years?|\d+\s*years? and up|baby\s*[-–—to]\s*\d+\s*years?)",
        value,
        re.I,
    )
    return match.group(1).strip() if match else value.strip()


def _extract_review_country(review: Any) -> Optional[str]:
    date_tag = review.select_one("span[data-hook='review-date']")
    if not date_tag:
        return None
    text = date_tag.get_text(" ", strip=True)
    match = re.search(r"Reviewed in (.+?) on", text)
    return match.group(1).strip() if match else None


def _review_to_dict(review: Any) -> Dict[str, str]:
    payload = {
        "author": "N/A",
        "rating": "N/A",
        "title": "N/A",
        "date": "N/A",
        "country": _extract_review_country(review) or "N/A",
        "text": "N/A",
        "verified": "N/A",
        "format": "N/A",
    }
    name_tag = review.select_one("span.a-profile-name")
    if name_tag:
        payload["author"] = name_tag.get_text(strip=True)
    rating_tag = review.select_one("i[data-hook='review-star-rating'] span.a-icon-alt")
    if rating_tag:
        payload["rating"] = rating_tag.get_text(strip=True)
    title_anchor = review.select_one("a[data-hook='review-title']")
    if title_anchor:
        payload["title"] = clean_text(title_anchor.get_text(" ", strip=True))
    date_tag = review.select_one("span[data-hook='review-date']")
    if date_tag:
        payload["date"] = date_tag.get_text(" ", strip=True)
    body_tag = review.select_one("div[data-hook='review-collapsed']") or review.select_one("span[data-hook='review-body']")
    if body_tag:
        payload["text"] = clean_text(body_tag.get_text(" ", strip=True))
    verified_tag = review.select_one("span[data-hook='avp-badge']")
    if verified_tag and verified_tag.get_text(strip=True):
        payload["verified"] = verified_tag.get_text(strip=True)
    format_tag = review.select_one("a[data-hook='review-format']") or review.select_one("span[data-hook='format-strip-linkless']")
    if format_tag:
        payload["format"] = clean_text(format_tag.get_text(strip=True).replace("Format:", ""))
    return payload


def extract_reviews_json(soup: BeautifulSoup, max_us: int = 10) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for review in soup.select("[data-hook='review']"):
        if len(out) >= max_us:
            break
        parsed = _review_to_dict(review)
        if len(parsed.get("text", "")) < 20:
            continue
        if "United States" in parsed.get("country", ""):
            out.append(parsed)
    return out


def extract_reviews_json_from_soups(soups: List[BeautifulSoup], max_us: int = 10) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for soup in soups:
        for review in extract_reviews_json(soup, max_us=max_us):
            key = (
                review.get("author", ""),
                review.get("title", ""),
                review.get("date", ""),
                review.get("text", "")[:120],
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(review)
            if len(out) >= max_us:
                return out
    return out


def extract_editorial_sections(soup: BeautifulSoup) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    editorial_section = soup.find("div", id="editorialReviews_feature_div") or soup.find("div", {"data-hook": "editorial-reviews"})
    if not editorial_section:
        return sections
    for heading in editorial_section.find_all(["h2", "h3", "h4"]):
        label = clean_text((heading.find("span") or heading).get_text(" ", strip=True))
        if label.lower() in {"editorial reviews", "n/a"}:
            continue
        parts: List[str] = []
        node = heading.find_next_sibling()
        while node and getattr(node, "name", None) not in {"h2", "h3", "h4"}:
            if getattr(node, "name", None) == "div":
                text = clean_text(node.get_text(" ", strip=True))
                if text != "N/A" and len(text) > 10:
                    parts.append(text)
            node = node.find_next_sibling() if node else None
        if parts:
            sections[label] = " ".join(parts)
    return sections


def extract_about_author(soup: BeautifulSoup) -> str:
    sections = extract_editorial_sections(soup)
    for key in ["About the Author", "About The Author"]:
        if key in sections:
            return sections[key][:3000]
    return "N/A"


def extract_customer_review_summary(soup: BeautifulSoup) -> str:
    customer_reviews_text = "N/A"
    try:
        widgets_container = soup.find("div", class_="cm_cr_grid_center_right_non_images_widgets")
        if widgets_container:
            product_summary = widgets_container.find("div", id="product-summary")
            if product_summary:
                summary_para = product_summary.find("p", class_="a-spacing-small")
                if summary_para:
                    summary_span = summary_para.find("span") or summary_para
                    summary_text = summary_span.get_text(separator=" ", strip=True)
                    if summary_text and len(summary_text) > 30:
                        customer_reviews_text = summary_text[:2000]
            if customer_reviews_text == "N/A":
                widget_summary = widgets_container.find("div", {"data-hook": "cr-insights-widget-summary"})
                if widget_summary:
                    summary_para = widget_summary.find("p", class_="a-spacing-small")
                    if summary_para:
                        summary_span = summary_para.find("span") or summary_para
                        summary_text = summary_span.get_text(separator=" ", strip=True)
                        if summary_text and len(summary_text) > 30:
                            customer_reviews_text = summary_text[:2000]
        if customer_reviews_text == "N/A":
            product_summary = soup.find("div", id="product-summary")
            if product_summary:
                summary_para = product_summary.find("p", class_="a-spacing-small")
                if summary_para:
                    summary_span = summary_para.find("span") or summary_para
                    summary_text = summary_span.get_text(separator=" ", strip=True)
                    if summary_text and len(summary_text) > 30:
                        customer_reviews_text = summary_text[:2000]
        if customer_reviews_text == "N/A":
            customers_say_heading = soup.find("h3", {"data-hook": "cr-insights-heading-label"})
            if customers_say_heading and "customers say" in customers_say_heading.get_text().lower():
                parent = customers_say_heading.find_parent("div")
                if parent:
                    summary_para = parent.find("p", class_="a-spacing-small")
                    if summary_para:
                        summary_span = summary_para.find("span") or summary_para
                        summary_text = summary_span.get_text(separator=" ", strip=True)
                        if summary_text and len(summary_text) > 30:
                            customer_reviews_text = summary_text[:2000]
    except Exception:
        pass
    return customer_reviews_text


def extract_customer_review_summary_from_soups(soups: List[BeautifulSoup]) -> str:
    for soup in soups:
        text = extract_customer_review_summary(soup)
        if text != "N/A":
            return text
    return "N/A"


def asin_from_book_url(book_url: Optional[str]) -> Optional[str]:
    if not book_url:
        return None
    match = re.search(r"/(?:dp|gp/product)/([A-Z0-9]{10})", str(book_url), re.I)
    return match.group(1) if match else None


def extract_book_details_from_soup(
    soup: BeautifulSoup,
    book_url: Optional[str],
    review_soups: Optional[List[BeautifulSoup]] = None,
) -> Dict[str, Any]:
    review_soups = review_soups or []
    all_review_soups = [soup, *review_soups]
    details = extract_all_metadata(soup)
    title_tag = soup.find("span", id="productTitle")
    title = clean_text(title_tag.get_text(" ", strip=True) if title_tag else None)

    summary_tag = soup.find("div", id="bookDescription_feature_div")
    summary = clean_text(summary_tag.get_text(" ", strip=True) if summary_tag else None)
    if summary.endswith("Read more"):
        summary = clean_text(summary[:-9])

    authors = []
    for tag in soup.find_all("span", class_="author notFaded"):
        name = tag.find("a", class_="a-link-normal") or tag.find("span", class_="contributorNameID")
        if name:
            authors.append(name.get_text(strip=True))
    author = ", ".join(authors) if authors else "N/A"

    def first_detail(*labels: str) -> str:
        for label in labels:
            if label in details and details[label]:
                return details[label]
        return "N/A"

    publisher = first_detail("publisher")
    published_date = first_detail("publication date")
    print_length = first_detail("print length", "fiona pages")
    isbn_10 = first_detail("isbn 10", "isbn-10")
    isbn_13 = first_detail("isbn 13", "isbn-13")
    language = first_detail("language")
    format_binding = first_detail("edition", "format", "binding")
    dimensions = first_detail("dimensions", "product dimensions", "package dimensions")
    item_weight = first_detail("item weight")
    series = first_detail("part of series", "series")

    reading_age_customer = "N/A"
    reading_age_publisher = "N/A"
    reading_age = "N/A"
    reading_age_source = "N/A"
    for label, value in details.items():
        if "reading age" in label or "age range" in label or "target age" in label:
            if "from customers" in value.lower():
                reading_age_customer = parse_reading_age_value(value)
            else:
                reading_age_publisher = parse_reading_age_value(value)
        elif "grade level" in label and value:
            grade_level = value
            break
    else:
        grade_level = "N/A"
    if reading_age_customer != "N/A":
        reading_age = reading_age_customer
        reading_age_source = "customer"
    elif reading_age_publisher != "N/A":
        reading_age = reading_age_publisher
        reading_age_source = "publisher"

    average_rating = "N/A"
    rating_section = soup.find(id="averageCustomerReviews")
    if rating_section:
        rating_tag = rating_section.find("span", class_="a-icon-alt")
        if rating_tag:
            average_rating = rating_tag.get_text(strip=True).split()[0]

    num_reviews = "N/A"
    reviews_section = soup.find(id="acrCustomerReviewText")
    if reviews_section:
        numbers = re.findall(r"\d+", reviews_section.get_text(strip=True).replace(",", ""))
        if numbers:
            num_reviews = numbers[0]

    price = "N/A"
    price_tag = soup.find("span", id="priceblock_ourprice") or soup.find("span", id="kindle-price")
    if price_tag:
        price = clean_text(price_tag.get_text(strip=True))
    if price == "N/A":
        whole = soup.find("span", class_="a-price-whole")
        frac = soup.find("span", class_="a-price-fraction")
        if whole:
            price = whole.get_text(strip=True) + (frac.get_text(strip=True) if frac else "")

    categories = "N/A"
    breadcrumb = soup.find("div", id="wayfinding-breadcrumbs_feature_div")
    if breadcrumb:
        nodes = breadcrumb.find_all("a", class_="a-link-normal a-color-tertiary")
        categories = " > ".join(n.get_text(strip=True) for n in nodes) or "N/A"

    editorial_reviews_text = "N/A"
    sections = extract_editorial_sections(soup)
    if sections:
        editorial_reviews_text = "\n\n".join(
            f"{k}: {v[:1500]}" for k, v in sections.items() if v and len(v.strip()) > 10
        ) or "N/A"

    customer_reviews_text = extract_customer_review_summary_from_soups(all_review_soups)
    customer_reviews_json = extract_reviews_json_from_soups(all_review_soups, max_us=10)
    about_author = extract_about_author(soup)
    return {
        "title": title,
        "author": author,
        "publisher": publisher,
        "published_date": published_date,
        "categories": categories,
        "summary": summary,
        "book_url": book_url or "N/A",
        "reading_age": reading_age,
        "reading_age_source": reading_age_source,
        "reading_age_customer": reading_age_customer,
        "reading_age_publisher": reading_age_publisher,
        "grade_level": grade_level,
        "print_length": print_length,
        "isbn_10": isbn_10,
        "isbn_13": isbn_13,
        "language": language,
        "format_binding": format_binding,
        "dimensions": dimensions,
        "item_weight": item_weight,
        "series": series,
        "average_rating": average_rating,
        "num_reviews": num_reviews,
        "price": price,
        "customer_reviews_text": customer_reviews_text,
        "editorial_reviews_text": editorial_reviews_text,
        "about_author": about_author,
        "customer_reviews_json": json.dumps(customer_reviews_json, ensure_ascii=True) if customer_reviews_json else "N/A",
        "all_metadata": details,
    }


def append_to_csv(books_data: List[Dict[str, Any]], csv_file: Path) -> None:
    if csv_file.exists():
        existing_data = pd.read_csv(csv_file, low_memory=False)
    else:
        existing_data = pd.DataFrame(columns=list(OUTPUT_COLUMN_MAPPING.values()))
    title_col = "Title" if "Title" in existing_data.columns else "Book Title"
    if "Source Merge Key" not in existing_data.columns:
        existing_data["Source Merge Key"] = ""
    metadata_columns: List[str] = []
    for book in books_data:
        for key in (book.get("all_metadata") or {}).keys():
            col = metadata_key_to_column(key)
            if col not in metadata_columns:
                metadata_columns.append(col)
    for col in metadata_columns:
        if col not in existing_data.columns:
            existing_data[col] = ""
    new_books: List[Dict[str, Any]] = []
    updated_rows = 0
    existing_titles = set(existing_data.get(title_col, pd.Series(dtype=str)).astype(str).str.lower().dropna().tolist())
    for book in books_data:
        row: Dict[str, Any] = {}
        for key, value in book.items():
            if key == "all_metadata":
                continue
            row[OUTPUT_COLUMN_MAPPING.get(key, passthrough_csv_column_name(key))] = value
        for key, value in (book.get("all_metadata") or {}).items():
            if not metadata_should_skip(key):
                row[metadata_key_to_column(key)] = value
        source_key = str(row.get("Source Merge Key", "")).strip()
        if source_key:
            matches = existing_data["Source Merge Key"].astype(str).str.strip() == source_key
            if matches.any():
                idx = existing_data[matches].index[0]
                for col, value in row.items():
                    if col not in existing_data.columns:
                        existing_data[col] = ""
                    existing_data.at[idx, col] = value
                updated_rows += 1
                continue
        if str(book.get("title", "")).lower() not in existing_titles:
            new_books.append(row)
    for row in new_books:
        for col in row:
            if col not in existing_data.columns:
                existing_data[col] = ""
    if new_books:
        existing_data = pd.concat([existing_data, pd.DataFrame(new_books)], ignore_index=True)
    if new_books or updated_rows:
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        existing_data.to_csv(csv_file, index=False)


def load_title_rows(input_csv: Path, skip_rows: int = 0) -> Tuple[pd.DataFrame, str]:
    df = read_csv_any_encoding(input_csv)
    if skip_rows:
        df = df.iloc[int(skip_rows):].copy().reset_index(drop=True)
    if "book_title" in df.columns:
        title_column = "book_title"
    elif "extracted_title" in df.columns:
        title_column = "extracted_title"
    else:
        raise ValueError("Input CSV must have 'book_title' or 'extracted_title' column")
    return df, title_column


def choose_rows_for_pending_only(df_titles: pd.DataFrame, output_csv: Path, title_column: str) -> pd.DataFrame:
    if not output_csv.exists():
        raise FileNotFoundError(f"--pending-only requested but output CSV does not exist: {output_csv}")
    existing = pd.read_csv(output_csv, low_memory=False)
    if "Title" not in existing.columns:
        raise ValueError(f"--pending-only requested but {output_csv} has no Title column")
    pending = existing[existing["Title"].astype(str).str.startswith("[PENDING_AMAZON]") | existing["Title"].astype(str).str.startswith("[NO_AMAZON_MATCH]")]
    if pending.empty:
        return df_titles.iloc[0:0].copy()
    if "Source Merge Key" in pending.columns:
        pending_keys = set(pending["Source Merge Key"].astype(str).str.strip())
        df_titles = df_titles.copy()
        df_titles["__merge_key__"] = df_titles.apply(
            lambda row: build_source_merge_key(row, title_column, fallback_title=str(row.get(title_column, ""))), axis=1
        )
        df_titles = df_titles[df_titles["__merge_key__"].isin(pending_keys)].drop(columns=["__merge_key__"])
    return df_titles.reset_index(drop=True)


def normalized_title_value(value: Any) -> str:
    text = _serialize_cell(value).strip()
    return "" if text.lower() in {"", "n/a", "nan"} else text


def rate_limit_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def amazon_search_url(query: str) -> str:
    return f"https://www.amazon.com/s?k={urllib.parse.quote(query)}&i=stripbooks"
