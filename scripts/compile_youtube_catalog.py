#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


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
    for p in sorted(titles_dir.glob("*_titles.csv")):
        df = read_csv_any_encoding(p)
        if "video_id" not in df.columns:
            continue
        df = df.copy()
        df["video_id"] = df["video_id"].astype(str).str.strip()
        df = df[df["video_id"].str.match(VALID_VIDEO_ID_RE, na=False)]

        playlist = p.name.replace("_titles.csv", "")
        df["source_playlist"] = playlist
        df["source_titles_csv"] = p.name
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
    for col in ["book_title", "story_title", "author", "video_title", "description", "extraction_status"]:
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
        raise ValueError(f"Expected video_id column in {path}")
    df = df.copy()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df = df[df["video_id"].str.match(VALID_VIDEO_ID_RE, na=False)]

    wanted = [
        "video_id",
        "youtube_url",
        "video_url",
        "has_amazon_match",
        "Title",
        "Author",
        "Reading Age",
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
        out["youtube_url"] = out["youtube_url"].fillna(out["video_url"])
    out["youtube_url"] = out["youtube_url"].fillna(out["video_id"].map(youtube_watch_url))

    # Prefix to avoid collisions with titles fields.
    ren: Dict[str, str] = {}
    for c in out.columns:
        if c == "video_id" or c == "youtube_url":
            continue
        ren[c] = f"amz_{c}"
    return out.rename(columns=ren)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile a single YouTube catalog from data/extracted_titles/*_titles.csv.")
    ap.add_argument("--titles-dir", type=Path, default=Path("data/extracted_titles"))
    ap.add_argument("--with-amazon", type=Path, default=None, help="Optional merge file (e.g. data/amazon_data/all_extracted_titles_with_amazon.csv).")
    ap.add_argument("--out", type=Path, default=Path("data/processed/youtube_catalog.csv.gz"))
    args = ap.parse_args()

    df = load_titles_dir(args.titles_dir)
    if args.with_amazon:
        amz = load_amazon_merge(args.with_amazon)
        df = df.merge(amz, on=["video_id", "youtube_url"], how="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, compression="infer")

    n_success = int((df["extraction_status"].fillna("").str.lower() == "success").sum()) if "extraction_status" in df.columns else 0
    print(f"Wrote: {args.out}")
    print(f"Rows: {len(df)} | unique videos: {df['video_id'].nunique()} | success rows: {n_success}")


if __name__ == "__main__":
    main()
