#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from analysis.natural_language.perplexity.table_io import read_table, write_table
except ModuleNotFoundError:
    from table_io import read_table, write_table


RANGE_RE = re.compile(
    r"(?P<lo>\d+(?:\.\d+)?|baby)\s*(?:-|–|—|\bto\b)\s*(?P<hi>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
PLUS_RE = re.compile(r"(?P<lo>\d+(?:\.\d+)?)\s*(?:\+|(?:years?|yrs?)?\s+and\s+up)", re.IGNORECASE)
BABY_PLUS_RE = re.compile(r"\bbaby\s+and\s+up\b", re.IGNORECASE)
SINGLE_AGE_RE = re.compile(r"(?P<age>\d+(?:\.\d+)?)\s*(?:years?|yrs?)", re.IGNORECASE)
MONTH_RE = re.compile(r"(?P<lo>\d+(?:\.\d+)?)\s*(?:-|–|—|\bto\b)\s*(?P<hi>\d+(?:\.\d+)?)\s*months?", re.IGNORECASE)


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def parse_age_bounds(value: Any, and_up_max: float = 18.0) -> tuple[Optional[float], Optional[float]]:
    if _is_null(value):
        return None, None
    text = str(value).strip().lower()
    if not text or text == "nan" or text == "n/a":
        return None, None

    month_match = MONTH_RE.search(text)
    if month_match:
        lo = float(month_match.group("lo")) / 12.0
        hi = float(month_match.group("hi")) / 12.0
        return (lo, hi) if lo <= hi else (hi, lo)

    match = RANGE_RE.search(text)
    if match:
        lo_raw = match.group("lo")
        lo = 0.0 if lo_raw.lower() == "baby" else float(lo_raw)
        hi = float(match.group("hi"))
        return (lo, hi) if lo <= hi else (hi, lo)

    match = PLUS_RE.search(text)
    if match:
        lo = float(match.group("lo"))
        hi = max(lo, and_up_max)
        return lo, hi

    if BABY_PLUS_RE.search(text):
        return 0.0, and_up_max

    match = SINGLE_AGE_RE.search(text)
    if match:
        age = float(match.group("age"))
        return age, age

    return None, None


def first_existing(columns: list[str], df: pd.DataFrame) -> Optional[str]:
    for col in columns:
        if col in df.columns:
            return col
    return None


def unique_existing(columns: list[str], df: pd.DataFrame) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in columns:
        if col in seen or col not in df.columns:
            continue
        seen.add(col)
        out.append(col)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare story text and Amazon median-age labels for src/analysis/natural_language/perplexity analysis.")
    ap.add_argument("--master", type=Path, default=Path("data/processed/master_stories.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/analysis/natural_language/perplexity/stories_for_perplexity.parquet"))
    ap.add_argument("--text-col", type=str, default="transcript_best")
    ap.add_argument("--id-col", type=str, default="video_id")
    ap.add_argument("--age-col", type=str, default="Reading Age")
    ap.add_argument("--and-up-max-age", type=float, default=18.0)
    ap.add_argument("--min-words", type=int, default=20)
    ap.add_argument("--cleaned-only", action="store_true", help="Keep only rows with cleaned == 1 if a cleaned column exists.")
    ap.add_argument("--dedupe", action="store_true", help="Drop duplicate ids after sorting by age and text length.")
    args = ap.parse_args()

    df = read_table(args.master).copy()
    if args.cleaned_only and "cleaned" in df.columns:
        df = df[pd.to_numeric(df["cleaned"], errors="coerce").fillna(0).astype(int) == 1].copy()

    if args.text_col not in df.columns:
        fallback = first_existing(["transcript_best", "yt_cleaned_subtitles", "transcript_txt", "cleaned_subtitles"], df)
        if fallback is None:
            raise ValueError(f"Missing text column {args.text_col}; no known fallback text column found.")
        args.text_col = fallback

    if args.id_col not in df.columns:
        fallback_id = first_existing(["video_id", "story_id", "item_id"], df)
        if fallback_id is None:
            raise ValueError(f"Missing id column {args.id_col}; no known fallback id column found.")
        args.id_col = fallback_id

    text = df[args.text_col].fillna("").astype(str).str.strip()
    work = df[text != ""].copy()
    work["text"] = text[text != ""]
    work["n_chars"] = work["text"].map(len)
    work["n_words"] = work["text"].map(lambda s: len(str(s).split()))
    work = work[work["n_words"] >= args.min_words].copy()

    if "age_lo" in work.columns and "age_hi" in work.columns:
        work["age_lo"] = pd.to_numeric(work["age_lo"], errors="coerce")
        work["age_hi"] = pd.to_numeric(work["age_hi"], errors="coerce")
        age_values = work[args.age_col] if args.age_col in work.columns else pd.Series([None] * len(work), index=work.index)
        missing_bounds = work["age_lo"].isna() | work["age_hi"].isna()
        if missing_bounds.any():
            bounds = age_values.loc[missing_bounds].map(lambda value: parse_age_bounds(value, args.and_up_max_age))
            work.loc[missing_bounds, "age_lo"] = pd.to_numeric(bounds.map(lambda t: t[0]), errors="coerce")
            work.loc[missing_bounds, "age_hi"] = pd.to_numeric(bounds.map(lambda t: t[1]), errors="coerce")
    else:
        age_values = work[args.age_col] if args.age_col in work.columns else pd.Series([None] * len(work), index=work.index)
        bounds = age_values.map(lambda value: parse_age_bounds(value, args.and_up_max_age))
        work["age_lo"] = bounds.map(lambda t: t[0])
        work["age_hi"] = bounds.map(lambda t: t[1])

    if "age_median" not in work.columns:
        if "age_mid" in work.columns:
            work["age_median"] = pd.to_numeric(work["age_mid"], errors="coerce")
        else:
            work["age_median"] = pd.NA
    missing_age = pd.to_numeric(work["age_median"], errors="coerce").isna()
    work.loc[missing_age, "age_median"] = (
        pd.to_numeric(work.loc[missing_age, "age_lo"], errors="coerce")
        + pd.to_numeric(work.loc[missing_age, "age_hi"], errors="coerce")
    ) / 2.0

    keep_cols = unique_existing([
        args.id_col,
        "story_id",
        "video_id",
        "youtube_url",
        "Title",
        "Book Title",
        "Author",
        args.age_col,
        "amazon_age_source",
        "age_lo",
        "age_hi",
        "age_median",
        "n_chars",
        "n_words",
        "text",
    ], work)
    original_id_col = args.id_col
    out = work[keep_cols].copy()
    out = out.rename(columns={args.id_col: "item_id", args.age_col: "reading_age"})
    if original_id_col == "video_id" and "video_id" not in out.columns:
        out["video_id"] = out["item_id"]
    if original_id_col == "story_id" and "story_id" not in out.columns:
        out["story_id"] = out["item_id"]
    out["item_id"] = out["item_id"].astype(str)
    out["age_lo"] = pd.to_numeric(out["age_lo"], errors="coerce")
    out["age_hi"] = pd.to_numeric(out["age_hi"], errors="coerce")
    out["age_median"] = pd.to_numeric(out["age_median"], errors="coerce")
    out = out[out["age_median"].notna()].copy()
    out = out.sort_values(["age_median", "item_id"]).reset_index(drop=True)

    if args.dedupe:
        out = out.sort_values(["item_id", "n_words"], ascending=[True, False]).drop_duplicates("item_id", keep="first")
        out = out.sort_values(["age_median", "item_id"]).reset_index(drop=True)

    written = write_table(out, args.out)
    print(f"Wrote {written}")
    print(f"Rows: {len(out)}")
    if len(out):
        print(f"Age median range: {out['age_median'].min():.2f} - {out['age_median'].max():.2f}")
        print(f"Median words/story: {out['n_words'].median():.0f}")


if __name__ == "__main__":
    main()
