#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.amazon_scraper_helpers import ensure_operational_master_schema


VALID_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,}$")


def read_csv_any_encoding(path: Path) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    return pd.read_csv(path, low_memory=False)


def norm_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s or None


def youtube_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def first_non_null(series: pd.Series) -> Any:
    for v in series:
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def load_transcript_scrapes(output_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for p in sorted(output_dir.glob("*.csv")):
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        if "id" not in df.columns:
            continue
        df = df.copy()
        df["video_id"] = df["id"].astype(str).str.strip()
        df = df[df["video_id"].str.match(VALID_VIDEO_ID_RE, na=False)]
        df["transcript_source_csv"] = p.name
        if "playlist" not in df.columns:
            df["playlist"] = p.stem
        rows.append(df)

    if not rows:
        return pd.DataFrame(
            columns=[
                "video_id",
                "youtube_url",
                "video_title",
                "description",
                "transcript_source_csv",
                "transcript_source",
                "cleaned_subtitles",
                "playlist",
                "has_transcript_scrape",
            ]
        )

    all_df = pd.concat(rows, ignore_index=True)
    agg: Dict[str, Any] = {}
    for col in all_df.columns:
        if col == "video_id":
            continue
        agg[col] = first_non_null

    merged = all_df.groupby("video_id", as_index=False).agg(agg)
    merged = merged.rename(
        columns={
            "webpage_url": "youtube_url",
            "title": "video_title",
        }
    )
    if "youtube_url" not in merged.columns:
        merged["youtube_url"] = merged["video_id"].map(youtube_watch_url)
    else:
        merged["youtube_url"] = merged["youtube_url"].fillna(merged["video_id"].map(youtube_watch_url))
    merged["has_transcript_scrape"] = True
    return merged


def pick_best_row(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer successful extraction + more non-null fields.
    def score_row(row: pd.Series) -> int:
        status = norm_str(row.get("extraction_status"))
        is_success = int((status or "").lower() == "success")
        nn = 0
        for k in [
            "video_title",
            "description",
            "book_title",
            "story_title",
            "author",
            "youtube_url",
        ]:
            nn += int(norm_str(row.get(k)) is not None)
        return 100 * is_success + nn

    scores = df.apply(score_row, axis=1)
    best_idx = scores.groupby(df["video_id"]).idxmax()
    return df.loc[best_idx].reset_index(drop=True)


def load_titles_dir(titles_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for p in sorted(titles_dir.rglob("*_titles.csv")):
        df = read_csv_any_encoding(p)
        if "video_id" not in df.columns:
            continue
        df = df.copy()
        df["video_id"] = df["video_id"].astype(str).str.strip()
        df = df[df["video_id"].str.match(VALID_VIDEO_ID_RE, na=False)]

        playlist = p.name.replace("_titles.csv", "")
        df["source_playlist"] = playlist
        df["source_titles_csv"] = str(p.relative_to(titles_dir))
        rows.append(df)

    if not rows:
        return pd.DataFrame(
            columns=[
                "video_id",
                "youtube_url",
                "video_title",
                "description",
                "extraction_status",
                "book_title",
                "story_title",
                "author",
                "source_csv",
                "source_playlist",
                "source_titles_csv",
            ]
        )
    all_df = pd.concat(rows, ignore_index=True)

    rename = {}
    for col in ["Book Title", "book_title"]:
        if col in all_df.columns:
            rename[col] = "book_title"
    for col in ["Story Title", "story_title"]:
        if col in all_df.columns:
            rename[col] = "story_title"
    for col in ["Author", "author"]:
        if col in all_df.columns:
            rename[col] = "author"
    all_df = all_df.rename(columns=rename)

    # Normalize expected columns.
    for col in ["book_title", "story_title", "author", "video_title", "description", "extraction_status", "source_csv"]:
        if col not in all_df.columns:
            all_df[col] = None

    # Build youtube_url if missing.
    if "youtube_url" not in all_df.columns:
        all_df["youtube_url"] = None
    if "video_url" in all_df.columns:
        all_df["youtube_url"] = all_df["youtube_url"].fillna(all_df["video_url"])
    all_df["youtube_url"] = all_df.apply(
        lambda r: norm_str(r.get("youtube_url")) or youtube_watch_url(str(r["video_id"])),
        axis=1,
    )

    # Aggregate playlists per video_id (a video can appear in multiple lists).
    playlists = (
        all_df.groupby("video_id", as_index=False)["source_playlist"]
        .agg(lambda s: "|".join(sorted({norm_str(x) for x in s if norm_str(x)})))
        .rename(columns={"source_playlist": "source_playlists"})
    )

    # Choose one best row per video_id for the main descriptive fields.
    best = pick_best_row(all_df)
    best = best.merge(playlists, on="video_id", how="left")

    keep = [
        "video_id",
        "youtube_url",
        "video_title",
        "description",
        "extraction_status",
        "book_title",
        "story_title",
        "author",
        "source_csv",
        "source_playlists",
        "source_titles_csv",
    ]
    for c in keep:
        if c not in best.columns:
            best[c] = None
    return best[keep]


def load_amazon_merge(path: Path) -> pd.DataFrame:
    df = read_csv_any_encoding(path)
    if "video_id" not in df.columns:
        if "Source Merge Key" in df.columns:
            df = df.copy()
            df["video_id"] = df["Source Merge Key"].astype(str).str.split("||").str[0].str.strip()
        else:
            raise ValueError(f"Expected video_id or Source Merge Key column in {path}")
    df = df.copy()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df = df[df["video_id"].str.match(VALID_VIDEO_ID_RE, na=False)]

    wanted = [
        "video_id",
        "youtube_url",
        "video_url",
        "has_amazon_match",
        "Amazon Match Status",
        "Title",
        "Author",
        "Reading Age",
        "Reading Age Source",
        "Reading Age Customer",
        "Reading Age Publisher",
        "Grade Level",
        "Categories",
        "Num Reviews",
        "Average Rating",
        "Price",
        "Book URL",
        "story_id",
        "story_id_str",
        "source_csv_base",
    ]
    cols = [c for c in wanted if c in df.columns]
    out = df[cols].copy()
    # Prefer youtube_url, else video_url, else build from id.
    if "youtube_url" not in out.columns:
        out["youtube_url"] = None
    if "video_url" in out.columns:
        out["youtube_url"] = out["youtube_url"].combine_first(out["video_url"])
    out["youtube_url"] = out["youtube_url"].combine_first(out["video_id"].map(youtube_watch_url))

    agg: Dict[str, Any] = {}
    for c in out.columns:
        if c in {"video_id", "youtube_url"}:
            continue
        agg[c] = first_non_null
    out = out.groupby(["video_id", "youtube_url"], as_index=False).agg(agg)

    if "has_amazon_match" not in out.columns:
        out["has_amazon_match"] = None
    if "Amazon Match Status" in out.columns:
        status = out["Amazon Match Status"].fillna("").astype(str).str.strip().str.upper()
        inferred = status.eq("FOUND")
        missing = out["has_amazon_match"].isna() | out["has_amazon_match"].astype(str).str.strip().eq("")
        out.loc[missing, "has_amazon_match"] = inferred.loc[missing]

    # Prefix to avoid collisions with titles fields.
    ren: Dict[str, str] = {}
    for c in out.columns:
        if c == "video_id" or c == "youtube_url":
            continue
        ren[c] = f"amz_{c}"
    return out.rename(columns=ren)


def load_supplemental_age_merge(paths: List[Path]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    wanted = [
        "video_id",
        "Title",
        "Author",
        "Reading Age",
        "Reading Age Source",
        "Reading Age Customer",
        "Reading Age Publisher",
        "Grade Level",
        "Categories",
        "Book URL",
    ]
    for path in paths:
        if not path.exists():
            continue
        df = read_csv_any_encoding(path)
        if "video_id" not in df.columns:
            continue
        keep = [c for c in wanted if c in df.columns]
        if "video_id" not in keep:
            keep = ["video_id", *keep]
        df = df[keep].copy()
        df["video_id"] = df["video_id"].astype(str).str.strip()
        df = df[df["video_id"].str.match(VALID_VIDEO_ID_RE, na=False)]
        df["supp_source_table"] = path.name
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["video_id"])

    all_df = pd.concat(rows, ignore_index=True)
    agg: Dict[str, Any] = {}
    for c in all_df.columns:
        if c == "video_id":
            continue
        agg[c] = first_non_null
    out = all_df.groupby("video_id", as_index=False).agg(agg)

    ren: Dict[str, str] = {}
    for c in out.columns:
        if c == "video_id":
            continue
        ren[c] = f"supp_{c}"
    return out.rename(columns=ren)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_operational_master_schema(df)
    df["catalog_row_key"] = df["video_id"].astype(str)
    df["catalog_version"] = "youtube_catalog_v1"

    for col in [
        "has_transcript_scrape",
        "has_title_extraction_attempt",
        "has_title_extraction_success",
        "needs_title_extraction",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    age_cols = [
        c
        for c in [
            "amz_Reading Age",
            "amz_Reading Age Customer",
            "amz_Reading Age Publisher",
            "supp_Reading Age",
            "supp_Reading Age Customer",
            "supp_Reading Age Publisher",
        ]
        if c in df.columns
    ]
    if age_cols:
        df["has_amazon_age"] = df[age_cols].notna().any(axis=1)
    else:
        df["has_amazon_age"] = False
    df["amazon_age_success"] = df["has_amazon_age"].fillna(False).astype(bool)

    if "amz_has_amazon_match" in df.columns:
        match_norm = df["amz_has_amazon_match"].fillna("").astype(str).str.strip().str.lower()
        df["has_amazon_match"] = match_norm.isin(["true", "1", "yes"])
    else:
        df["has_amazon_match"] = False
    df["amazon_match_success"] = df["has_amazon_match"].fillna(False).astype(bool)

    def choose_first(*values: Any) -> Optional[str]:
        for value in values:
            value = norm_str(value)
            if value is not None:
                return value
        return None

    df["search_book_title"] = [
        choose_first(book_title, story_title, amz_title, video_title)
        for book_title, story_title, amz_title, video_title in zip(
            df.get("book_title", pd.Series([None] * len(df))),
            df.get("story_title", pd.Series([None] * len(df))),
            df.get("amz_Title", pd.Series([None] * len(df))),
            df.get("video_title", pd.Series([None] * len(df))),
        )
    ]
    df["search_story_title"] = [
        choose_first(story_title, book_title)
        for story_title, book_title in zip(
            df.get("story_title", pd.Series([None] * len(df))),
            df.get("book_title", pd.Series([None] * len(df))),
        )
    ]
    df["search_author"] = [
        choose_first(author, amz_author)
        for author, amz_author in zip(
            df.get("author", pd.Series([None] * len(df))),
            df.get("amz_Author", pd.Series([None] * len(df))),
        )
    ]
    df["search_query"] = [
        " ".join([part for part in [book, story, author] if part]).strip() or None
        for book, story, author in zip(df["search_book_title"], df["search_story_title"], df["search_author"])
    ]

    for base_col in ["Title", "Author", "Reading Age", "Reading Age Source", "Grade Level", "Categories", "Book URL"]:
        best_col = f"best_{base_col}"
        amz_col = f"amz_{base_col}"
        supp_col = f"supp_{base_col}"
        df[best_col] = [
            choose_first(amz_val, supp_val)
            for amz_val, supp_val in zip(
                df.get(amz_col, pd.Series([None] * len(df))),
                df.get(supp_col, pd.Series([None] * len(df))),
            )
        ]
    df["best_amazon_source_table"] = [
        choose_first(amz_src, supp_src)
        for amz_src, supp_src in zip(
            df.get("amz_source_csv_base", pd.Series([None] * len(df))),
            df.get("supp_source_table", pd.Series([None] * len(df))),
        )
    ]
    df["title_extraction_attempted"] = df["has_title_extraction_attempt"].fillna(False).astype(bool)
    df["title_extraction_success"] = df["has_title_extraction_success"].fillna(False).astype(bool)
    df["amazon_scrape_attempted"] = (
        df["has_amazon_match"].fillna(False).astype(bool)
        | df["has_amazon_age"].fillna(False).astype(bool)
        | df.get("Amazon Match Status", pd.Series([""] * len(df))).fillna("").astype(str).str.strip().ne("")
        | df.get("amz_Amazon Match Status", pd.Series([""] * len(df))).fillna("").astype(str).str.strip().ne("")
    )
    return df


def export_queue(df: pd.DataFrame, out_path: Path, mask: pd.Series) -> None:
    queue_cols = [
        "catalog_row_key",
        "video_id",
        "youtube_url",
        "video_title",
        "channel",
        "channel_id",
        "channel_url",
        "uploader",
        "uploader_id",
        "uploader_url",
        "playlist",
        "playlist_title",
        "playlist_id",
        "playlist_webpage_url",
        "transcript_source_csv",
        "source_csv",
        "source_titles_csv",
        "source_playlists",
        "book_title",
        "story_title",
        "author",
        "extraction_status",
        "search_book_title",
        "search_story_title",
        "search_author",
        "search_query",
        "has_title_extraction_attempt",
        "has_title_extraction_success",
        "needs_title_extraction",
        "ready_for_amazon_search",
        "matched_but_missing_amazon_age",
        "has_amazon_match",
        "has_amazon_age",
        "best_Title",
        "best_Author",
        "best_Reading Age",
        "best_Reading Age Source",
        "best_Grade Level",
        "best_Categories",
        "best_Book URL",
        "best_amazon_source_table",
        "amz_Title",
        "amz_Author",
        "amz_Reading Age",
        "amz_Categories",
        "amz_Book URL",
        "amz_source_csv_base",
        "supp_Title",
        "supp_Author",
        "supp_Reading Age",
        "supp_Reading Age Source",
        "supp_Grade Level",
        "supp_Categories",
        "supp_Book URL",
        "supp_source_table",
        "amz_story_id",
        "amz_story_id_str",
    ]
    queue_cols = [c for c in queue_cols if c in df.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[mask, queue_cols].to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile a single YouTube catalog from data/extracted_titles/*_titles.csv.")
    ap.add_argument("--titles-dir", type=Path, default=Path("data/extracted_titles"))
    ap.add_argument("--output-dir", type=Path, default=Path("output"))
    ap.add_argument("--with-amazon", type=Path, default=None, help="Optional merge file (e.g. data/amazon_data/all_extracted_titles_with_amazon.csv).")
    ap.add_argument(
        "--supplemental-age",
        type=Path,
        nargs="*",
        default=[
            Path("data/amazon_data/amazon_youtube_batch1.csv"),
            Path("story_wAmazonData_042626/amazon_youtube_batch1_wid.csv"),
            Path("data/amazon_data/missing_reading_age_extracted_success_amazon.csv"),
        ],
        help="Extra Amazon-derived tables to merge for age coverage before building queues.",
    )
    ap.add_argument("--out", type=Path, default=Path("data/processed/youtube_catalog.csv.gz"))
    ap.add_argument("--queue-dir", type=Path, default=None, help="Optional queue export directory. Omit to keep only the master table.")
    args = ap.parse_args()

    transcripts = load_transcript_scrapes(args.output_dir)
    titles = load_titles_dir(args.titles_dir)
    if len(transcripts) and len(titles):
        df = transcripts.merge(titles, on="video_id", how="outer", suffixes=("", "_titles"))
        if "has_transcript_scrape" not in df.columns:
            df["has_transcript_scrape"] = False
        df["has_transcript_scrape"] = df["has_transcript_scrape"].fillna(False).astype(bool)

        # Prefer transcript scrape metadata as primary, but retain titles-only rows.
        if "youtube_url_titles" in df.columns:
            if "youtube_url" not in df.columns:
                df["youtube_url"] = None
            df["youtube_url"] = df["youtube_url"].fillna(df["youtube_url_titles"])
            df = df.drop(columns=["youtube_url_titles"])
        elif "youtube_url" not in df.columns:
            df["youtube_url"] = None
        df["youtube_url"] = df["youtube_url"].fillna(df["video_id"].map(youtube_watch_url))

        for col in ["video_title", "description"]:
            other = f"{col}_titles"
            if other in df.columns:
                if col not in df.columns:
                    df[col] = None
                df[col] = df[col].fillna(df[other])
                df = df.drop(columns=[other])
    else:
        df = titles.copy() if len(titles) else transcripts.copy()
        if len(df) and "has_transcript_scrape" not in df.columns:
            df["has_transcript_scrape"] = False

    if "extraction_status" not in df.columns:
        df["extraction_status"] = None
    status_norm = df["extraction_status"].fillna("").astype(str).str.lower().str.strip()
    df["has_title_extraction_attempt"] = status_norm.ne("")
    df["has_title_extraction_success"] = status_norm.eq("success")
    df["needs_title_extraction"] = df["has_transcript_scrape"].fillna(False) & ~df["has_title_extraction_success"]

    if args.with_amazon:
        amz = load_amazon_merge(args.with_amazon)
        df = df.merge(amz, on=["video_id", "youtube_url"], how="left")
    if args.supplemental_age:
        supp = load_supplemental_age_merge(args.supplemental_age)
        if len(supp):
            df = df.merge(supp, on="video_id", how="left")

    # Keep one row per video_id if upstream merges introduce duplicates.
    if df["video_id"].duplicated().any():
        df = df.sort_values(["video_id"]).groupby("video_id", as_index=False).agg(first_non_null)

    df = add_derived_columns(df)
    df["ready_for_amazon_search"] = df["has_transcript_scrape"] & df["has_title_extraction_success"] & ~df["has_amazon_age"]
    df["matched_but_missing_amazon_age"] = df["has_transcript_scrape"] & df["has_amazon_match"] & ~df["has_amazon_age"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, compression="infer")

    if args.queue_dir is not None:
        export_queue(
            df,
            args.queue_dir / "needs_title_extraction.csv",
            df["has_transcript_scrape"] & ~df["has_title_extraction_success"],
        )
        export_queue(
            df,
            args.queue_dir / "ready_for_amazon_search.csv",
            df["ready_for_amazon_search"],
        )
        export_queue(
            df,
            args.queue_dir / "matched_but_missing_amazon_age.csv",
            df["matched_but_missing_amazon_age"],
        )

    n_scraped = int(df["has_transcript_scrape"].sum()) if "has_transcript_scrape" in df.columns else 0
    n_attempted = int(df["has_title_extraction_attempt"].sum())
    n_success = int(df["has_title_extraction_success"].sum())
    n_left = int(df["needs_title_extraction"].sum())
    n_ready_for_amazon = int(df["ready_for_amazon_search"].sum())
    n_missing_age_after_match = int(df["matched_but_missing_amazon_age"].sum())
    print(f"Wrote: {args.out}")
    print(f"Rows: {len(df)} | unique videos: {df['video_id'].nunique()}")
    print(f"Transcript scrapes: {n_scraped}")
    print(f"Title extraction attempted: {n_attempted}")
    print(f"Title extraction success: {n_success}")
    print(f"Left to scrape titles: {n_left}")
    print(f"Ready for Amazon search: {n_ready_for_amazon}")
    print(f"Matched but missing Amazon age: {n_missing_age_after_match}")
    if args.queue_dir is not None:
        print(f"Queue dir: {args.queue_dir}")


if __name__ == "__main__":
    main()
