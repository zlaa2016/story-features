#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


AGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)")
YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})")


def parse_age_range(value: Any) -> Optional[Tuple[float, float]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = AGE_RE.search(str(value))
    if not match:
        return None
    lo, hi = float(match.group(1)), float(match.group(2))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def parse_youtube_id(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    match = YOUTUBE_ID_RE.search(s)
    if match:
        return match.group(1)
    # Sometimes the sheet already stores the id directly.
    if re.fullmatch(r"[A-Za-z0-9_-]{6,}", s):
        return s
    return None


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


def load_youtube_outputs(output_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for csv_path in sorted(output_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue
        if "id" not in df.columns:
            continue
        df = df.copy()
        df["id"] = df["id"].astype(str)
        df["source_output_csv"] = csv_path.name
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["video_id"])

    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df.rename(columns={"id": "video_id"})

    # Prefer non-null per video_id per column.
    agg: Dict[str, Any] = {}
    for col in all_df.columns:
        if col == "video_id":
            continue
        agg[col] = first_non_null

    merged = all_df.groupby("video_id", as_index=False).agg(agg)
    # Prefix to avoid collisions with Amazon fields.
    merged = merged.rename(columns={c: f"yt_{c}" for c in merged.columns if c != "video_id"})
    return merged


def load_story_transcripts(transcripts_dir: Path) -> pd.DataFrame:
    story_ids: List[str] = []
    texts: List[str] = []
    paths: List[str] = []
    for txt_path in sorted(transcripts_dir.glob("*.txt")):
        match = re.match(r"(\d{6})_", txt_path.name)
        if not match:
            continue
        story_id = match.group(1)
        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            text = ""
        story_ids.append(story_id)
        texts.append(text)
        paths.append(str(txt_path))
    return pd.DataFrame(
        {
            "story_id": story_ids,
            "transcript_txt": texts,
            "transcript_txt_path": paths,
        }
    )


@dataclass
class ReviewParseResult:
    story_id: str
    n_sentences: Optional[int]
    n_words: Optional[int]
    n_events: Optional[int]
    method: Optional[str]
    events: List[Dict[str, Any]]


def _strip_md_emphasis(s: str) -> str:
    return re.sub(r"[*_`]+", "", s).strip()


def parse_review_sample(md_path: Path) -> Optional[ReviewParseResult]:
    match = re.match(r"(\d{6})_", md_path.name)
    if not match:
        return None
    story_id = match.group(1)
    text = md_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    n_sentences = None
    n_words = None
    n_events = None
    method = None
    events: List[Dict[str, Any]] = []

    # Header line like: **74 sentences | 711 words | 12 events**
    for ln in lines[:20]:
        m = re.search(r"\*\*(\d+)\s+sentences\s+\|\s+(\d+)\s+words\s+\|\s+(\d+)\s+events\*\*", ln)
        if m:
            n_sentences = int(m.group(1))
            n_words = int(m.group(2))
            n_events = int(m.group(3))
            break

    # Method line like: ## Method O v2.1 — 12 events
    for ln in lines[:60]:
        m = re.match(r"##\s+(.*?)\s+—\s+(\d+)\s+events\s*$", ln.strip())
        if m:
            method = m.group(1).strip()
            if n_events is None:
                n_events = int(m.group(2))
            break

    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        m = re.match(r"###\s+Event\s+(\d+)\s+—\s+sents\s+(\d+)[-–](\d+)\s+\((\d+)\s+sentences\)\s*$", ln)
        if not m:
            i += 1
            continue

        event_index = int(m.group(1))
        sent_start = int(m.group(2))
        sent_end = int(m.group(3))
        event_sent_count = int(m.group(4))

        event_type = None
        boundary = None
        summary = None
        sentences: List[Dict[str, Any]] = []

        # Next non-empty line is often **type**
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            mtype = re.match(r"\*\*(.+?)\*\*", lines[j].strip())
            if mtype:
                event_type = _strip_md_emphasis(mtype.group(1))
                j += 1

        # Next italic lines: *Boundary: ...* then *Summary...*
        italic_lines = 0
        while j < len(lines) and italic_lines < 2:
            s = lines[j].strip()
            if s.startswith("*") and s.endswith("*"):
                content = _strip_md_emphasis(s)
                if content.lower().startswith("boundary:"):
                    boundary = content[len("boundary:") :].strip()
                else:
                    summary = content
                italic_lines += 1
            elif s.startswith("*") or s.endswith("*"):
                # ignore malformed emphasis
                pass
            elif s:
                # stop when leaving preamble
                break
            j += 1

        # Collect blockquote sentences until next header.
        k = j
        while k < len(lines):
            s = lines[k].rstrip()
            if s.startswith("### "):
                break
            if s.startswith(">"):
                s2 = s.lstrip("> ").strip()
                sm = re.match(r"\[(\d+)\]\s+(.*)$", s2)
                if sm:
                    idx = int(sm.group(1))
                    sent = sm.group(2).replace("&nbsp;", " ").strip()
                    sentences.append({"i": idx, "text": sent})
            k += 1

        events.append(
            {
                "event": event_index,
                "sent_start": sent_start,
                "sent_end": sent_end,
                "n_sentences": event_sent_count,
                "type": event_type,
                "boundary": boundary,
                "summary": summary,
                "sentences": sentences,
            }
        )
        i = k

    return ReviewParseResult(
        story_id=story_id,
        n_sentences=n_sentences,
        n_words=n_words,
        n_events=n_events,
        method=method,
        events=events,
    )


def load_review_samples(review_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for md_path in sorted(review_dir.glob("*.md")):
        parsed = parse_review_sample(md_path)
        if not parsed:
            continue
        rows.append(
            {
                "story_id": parsed.story_id,
                "review_n_sentences": parsed.n_sentences,
                "review_n_words": parsed.n_words,
                "review_n_events": parsed.n_events,
                "review_method": parsed.method,
                "review_events_json": json.dumps(parsed.events, ensure_ascii=False),
                "review_md_path": str(md_path),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["story_id"])
    return pd.DataFrame(rows)


def ensure_story_id_str(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        s = str(int(value)).zfill(6)
        if re.fullmatch(r"\d{6}", s):
            return s
    except Exception:
        pass
    s2 = str(value).strip()
    if re.fullmatch(r"\d{6}", s2):
        return s2
    return None


def write_parquet_or_csv(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(out_path, index=False)
            return out_path
        except Exception:
            # fall back below
            pass
        out_path = out_path.with_suffix(".csv.gz")

    df.to_csv(out_path, index=False, compression="gzip" if out_path.suffix.endswith(".gz") else None)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Join Amazon + YouTube transcripts + event segments into a master table.")
    ap.add_argument(
        "--base",
        type=Path,
        default=Path("data/amazon_data/all_extracted_titles_with_amazon.xlsx"),
        help="Base Amazon/YouTube metadata table (xlsx or csv).",
    )
    ap.add_argument(
        "--base-xlsx",
        type=Path,
        default=None,
        help="Deprecated alias for --base (kept for backwards compatibility).",
    )
    ap.add_argument(
        "--videoid-storyid-map",
        type=Path,
        default=Path("data/amazon_data/unmatched_videos_with_youtube.csv"),
        help="Optional mapping table with columns like video_id, story_id (to link review_samples/transcripts_to_clean).",
    )
    ap.add_argument("--output-dir", type=Path, default=Path("output"))
    ap.add_argument("--transcripts-dir", type=Path, default=Path("data/transcripts_to_clean"))
    ap.add_argument("--review-samples-dir", type=Path, default=Path("review_samples"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/master_stories.parquet"))
    ap.add_argument("--out-index", type=Path, default=Path("data/processed/master_stories_index.csv"))
    args = ap.parse_args()

    base_path = args.base_xlsx if args.base_xlsx is not None else args.base
    if base_path.suffix.lower() in {".xlsx", ".xls"}:
        base = pd.read_excel(base_path)
    else:
        base = pd.read_csv(base_path, low_memory=False)
    base = base.copy()

    # Normalize ids
    if "video_id" in base.columns:
        base["video_id"] = base["video_id"].apply(parse_youtube_id)
    if "video_url" in base.columns:
        missing = base["video_id"].isna()
        base.loc[missing, "video_id"] = base.loc[missing, "video_url"].apply(parse_youtube_id)
    if "youtube_url" in base.columns:
        missing = base["video_id"].isna()
        base.loc[missing, "video_id"] = base.loc[missing, "youtube_url"].apply(parse_youtube_id)

    if "story_id" in base.columns:
        base["story_id"] = base["story_id"].apply(ensure_story_id_str)
    else:
        base["story_id"] = None

    # If base table lacks story_id, try to map via video_id.
    if args.videoid_storyid_map and args.videoid_storyid_map.exists() and "video_id" in base.columns:
        try:
            m = pd.read_csv(args.videoid_storyid_map, low_memory=False)
            if "video_id" in m.columns and "story_id" in m.columns:
                m = m[["video_id", "story_id"]].copy()
                m["video_id"] = m["video_id"].apply(parse_youtube_id)
                m["story_id"] = m["story_id"].apply(ensure_story_id_str)
                # Deduplicate video_id to avoid blowing up the base table.
                m = (
                    m.groupby("video_id", as_index=False)
                    .agg({"story_id": first_non_null})
                )
                base = base.merge(m, on="video_id", how="left", suffixes=("", "_map"))
                base["story_id"] = base["story_id"].fillna(base.get("story_id_map"))
                if "story_id_map" in base.columns:
                    base = base.drop(columns=["story_id_map"])
        except Exception:
            pass

    # Normalize age
    base["age_range"] = base.get("Reading Age", pd.Series([None] * len(base))).apply(parse_age_range)
    base["age_lo"] = base["age_range"].apply(lambda t: t[0] if isinstance(t, tuple) else None)
    base["age_hi"] = base["age_range"].apply(lambda t: t[1] if isinstance(t, tuple) else None)
    base["age_mid"] = base["age_range"].apply(lambda t: (t[0] + t[1]) / 2 if isinstance(t, tuple) else None)

    yt = load_youtube_outputs(args.output_dir)
    txt = load_story_transcripts(args.transcripts_dir)
    reviews = load_review_samples(args.review_samples_dir)

    merged = base.merge(yt, on="video_id", how="left")
    merged = merged.merge(txt, on="story_id", how="left")
    merged = merged.merge(reviews, on="story_id", how="left")

    yt_clean = merged["yt_cleaned_subtitles"] if "yt_cleaned_subtitles" in merged.columns else pd.Series([None] * len(merged))
    yt_src = merged["yt_transcript_source"] if "yt_transcript_source" in merged.columns else pd.Series([None] * len(merged))

    merged["transcript_best"] = yt_clean
    merged["transcript_source_best"] = yt_src.where(yt_clean.notna(), None)

    fallback_mask = merged["transcript_best"].isna() & merged["transcript_txt"].notna()
    merged.loc[fallback_mask, "transcript_best"] = merged.loc[fallback_mask, "transcript_txt"]
    merged.loc[fallback_mask, "transcript_source_best"] = "transcripts_to_clean"

    merged["has_transcript"] = merged["transcript_best"].notna()
    merged["has_events"] = merged["review_events_json"].notna()
    merged["transcript_num_chars"] = merged["transcript_best"].fillna("").map(len)
    merged["transcript_num_words"] = merged["transcript_best"].fillna("").map(lambda s: len(str(s).split()))

    out_written = write_parquet_or_csv(merged, args.out)

    index_cols = [
        "video_id",
        "story_id",
        "video_title",
        "Title",
        "Author",
        "Categories",
        "Num Reviews",
        "Average Rating",
        "Price",
        "Grade Level",
        "Book URL",
        "Reading Age",
        "age_lo",
        "age_hi",
        "transcript_source_best",
        "has_transcript",
        "has_events",
    ]
    index_cols = [c for c in index_cols if c in merged.columns]
    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    merged[index_cols].to_csv(args.out_index, index=False)

    print(f"Wrote master: {out_written}")
    print(f"Wrote index:  {args.out_index}")
    print(f"Rows: {len(merged)} | transcripts: {int(merged['has_transcript'].sum())} | events: {int(merged['has_events'].sum())}")


if __name__ == "__main__":
    main()
