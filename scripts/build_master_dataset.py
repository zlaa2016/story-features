#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


AGE_RE = re.compile(
    r"(?P<lo>\d+(?:\.\d+)?|baby)\s*(?:-|–|—|\bto\b)\s*(?P<hi>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
PLUS_RE = re.compile(r"(?P<lo>\d+(?:\.\d+)?)\s*(?:\+|(?:years?|yrs?)?\s+and\s+up)", re.IGNORECASE)
BABY_PLUS_RE = re.compile(r"\bbaby\s+and\s+up\b", re.IGNORECASE)
SINGLE_AGE_RE = re.compile(r"(?P<age>\d+(?:\.\d+)?)\s*(?:years?|yrs?)", re.IGNORECASE)
MONTH_RE = re.compile(
    r"(?P<lo>\d+(?:\.\d+)?)\s*(?:-|–|—|\bto\b)\s*(?P<hi>\d+(?:\.\d+)?)\s*months?",
    re.IGNORECASE,
)
YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})")
VALID_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,}$")


def parse_age_range(value: Any, and_up_max: float = 18.0) -> Optional[Tuple[float, float]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "n/a"}:
        return None

    month_match = MONTH_RE.search(text)
    if month_match:
        lo = float(month_match.group("lo")) / 12.0
        hi = float(month_match.group("hi")) / 12.0
        return (lo, hi) if lo <= hi else (hi, lo)

    match = AGE_RE.search(text)
    if match:
        lo_raw = match.group("lo")
        lo = 0.0 if lo_raw == "baby" else float(lo_raw)
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

    return None


def normalize_age_source(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower()
    if not s or s == "nan":
        return None
    if "customer" in s:
        return "customer"
    if "publisher" in s:
        return "publisher"
    if s in {"customer", "publisher"}:
        return s
    return None


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


def _normalize_story_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float)):
        try:
            return str(int(value)).zfill(6)
        except Exception:
            return None
    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"\d{1,6}", s):
        return s.zfill(6)
    return None


def load_cleaned_filter(meta_path: Path, cleaned_col: str) -> pd.DataFrame:
    meta = read_csv_any_encoding(meta_path)
    keep = pd.to_numeric(meta.get(cleaned_col), errors="coerce").fillna(0).astype(int) == 1
    meta = meta[keep].copy()
    if "story_id" in meta.columns:
        meta["story_id"] = meta["story_id"].apply(_normalize_story_id)
    else:
        meta["story_id"] = None
    if "video_id" in meta.columns:
        meta["video_id"] = meta["video_id"].apply(parse_youtube_id)
    else:
        meta["video_id"] = None
    cols = [c for c in ["story_id", "video_id", cleaned_col] if c in meta.columns]
    out = meta[cols].copy()
    if cleaned_col in out.columns and cleaned_col != "cleaned":
        out = out.rename(columns={cleaned_col: "cleaned"})
    elif cleaned_col not in out.columns:
        out["cleaned"] = 1
    return out.drop_duplicates(subset=[c for c in ["story_id", "video_id"] if c in out.columns], keep="first")


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


def load_playlists_from_extracted_titles(titles_dir: Path) -> pd.DataFrame:
    """
    Aggregate the playlist(s) a video_id appeared in from data/extracted_titles/*_titles.csv.
    Returns columns: video_id, source_playlists
    """
    if not titles_dir.exists():
        return pd.DataFrame(columns=["video_id", "source_playlists"])

    rows: List[pd.DataFrame] = []
    for p in sorted(titles_dir.glob("*_titles.csv")):
        try:
            df = read_csv_any_encoding(p)
        except Exception:
            continue
        if "video_id" not in df.columns:
            continue
        playlist = p.name.replace("_titles.csv", "")
        tmp = df[["video_id"]].copy()
        tmp["video_id"] = tmp["video_id"].astype(str).str.strip()
        tmp = tmp[tmp["video_id"].str.match(VALID_VIDEO_ID_RE, na=False)]
        tmp["source_playlist"] = playlist
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["video_id", "source_playlists"])

    all_df = pd.concat(rows, ignore_index=True).drop_duplicates()
    playlists = (
        all_df.groupby("video_id", as_index=False)["source_playlist"]
        .agg(lambda s: "|".join(sorted(set(s))))
        .rename(columns={"source_playlist": "source_playlists"})
    )
    return playlists


def _decode_text_bytes(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").strip()


def load_story_transcripts(transcripts_dir: Path, transcripts_zip: Optional[Path] = None) -> pd.DataFrame:
    story_ids: List[str] = []
    texts: List[str] = []
    paths: List[str] = []

    if transcripts_zip is not None:
        with zipfile.ZipFile(transcripts_zip, "r") as zf:
            for info in sorted(zf.infolist(), key=lambda item: item.filename):
                name = Path(info.filename).name
                if info.is_dir() or not name.lower().endswith(".txt"):
                    continue
                match = re.match(r"(\d{6})_", name)
                if not match:
                    continue
                story_ids.append(match.group(1))
                texts.append(_decode_text_bytes(zf.read(info.filename)))
                paths.append(f"{transcripts_zip}:{info.filename}")
    else:
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

    if not story_ids:
        return pd.DataFrame(columns=["story_id", "transcript_txt", "transcript_txt_path"])

    # Keep one transcript per story id. If duplicates exist, prefer the longest
    # non-empty cleaned text.
    rows = pd.DataFrame({"story_id": story_ids, "transcript_txt": texts, "transcript_txt_path": paths})
    rows["_n_chars"] = rows["transcript_txt"].fillna("").map(len)
    rows = (
        rows.sort_values(["story_id", "_n_chars", "transcript_txt_path"], ascending=[True, False, True])
        .drop_duplicates("story_id", keep="first")
        .drop(columns=["_n_chars"])
    )
    return pd.DataFrame(
        {
            "story_id": rows["story_id"],
            "transcript_txt": rows["transcript_txt"],
            "transcript_txt_path": rows["transcript_txt_path"],
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
    out = pd.DataFrame(rows)
    # Some reviewed samples exist as near-duplicate markdown files with the same
    # six-digit story id. Keep one row per story so review joins cannot inflate
    # the master table.
    out["_review_n_events_sort"] = pd.to_numeric(out["review_n_events"], errors="coerce").fillna(-1)
    out["_review_n_words_sort"] = pd.to_numeric(out["review_n_words"], errors="coerce").fillna(-1)
    out = (
        out.sort_values(
            ["story_id", "_review_n_events_sort", "_review_n_words_sort", "review_md_path"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("story_id", keep="first")
        .drop(columns=["_review_n_events_sort", "_review_n_words_sort"])
    )
    return out


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
        default=Path("story_wAmazonData_042626/amazon_youtube_batch1_wid.csv"),
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
        default=Path("data/amazon_data/all_story_nosub_manual_amazon.csv"),
        help="Optional mapping table with columns like video_id, story_id (to link review_samples/transcripts_to_clean).",
    )
    ap.add_argument("--output-dir", type=Path, default=Path("output"))
    ap.add_argument("--titles-dir", type=Path, default=Path("data/extracted_titles"))
    ap.add_argument("--transcripts-dir", type=Path, default=Path("story_wAmazonData_042626/cleaned_idchecked"))
    ap.add_argument(
        "--transcripts-zip",
        type=Path,
        default=None,
        help="Optional zip archive containing cleaned story .txt files. Overrides --transcripts-dir.",
    )
    ap.add_argument(
        "--transcripts-source-name",
        type=str,
        default=None,
        help="Label to store in transcript_source_best for text loaded from --transcripts-dir.",
    )
    ap.add_argument("--review-samples-dir", type=Path, default=Path("review_samples"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/master_stories.parquet"))
    ap.add_argument("--out-csv", type=Path, default=Path("data/processed/master_stories.csv.gz"))
    ap.add_argument("--out-index", type=Path, default=Path("data/processed/master_stories_index.csv"))
    ap.add_argument(
        "--cleaned-meta",
        type=Path,
        default=None,
        help="Optional metadata CSV with story_id/video_id and a cleaned column to merge/filter against.",
    )
    ap.add_argument("--cleaned-col", type=str, default="cleaned")
    ap.add_argument(
        "--cleaned-only",
        action="store_true",
        help="Keep only rows whose story_id/video_id are marked cleaned=1 in --cleaned-meta.",
    )
    args = ap.parse_args()

    base_path = args.base_xlsx if args.base_xlsx is not None else args.base
    if base_path.suffix.lower() in {".xlsx", ".xls"}:
        base = pd.read_excel(base_path)
    else:
        base = read_csv_any_encoding(base_path)
    base = base.copy()

    # Normalize ids
    if "video_id" not in base.columns:
        base["video_id"] = None
    if "video_id" in base.columns:
        base["video_id"] = base["video_id"].apply(parse_youtube_id)
    if "video_url" in base.columns:
        missing = base["video_id"].isna()
        base.loc[missing, "video_id"] = base.loc[missing, "video_url"].apply(parse_youtube_id)
    if "youtube_url" in base.columns:
        missing = base["video_id"].isna()
        base.loc[missing, "video_id"] = base.loc[missing, "youtube_url"].apply(parse_youtube_id)

    # Drop rows we can't link to YouTube (these are usually unmatched Amazon rows / scrape artifacts).
    base = base[base["video_id"].notna()].copy()

    # Ensure we always have a usable YouTube watch URL for inspection/exports.
    if "youtube_url" not in base.columns:
        base["youtube_url"] = None
    base["youtube_url"] = base["youtube_url"].where(
        base["youtube_url"].notna(),
        "https://www.youtube.com/watch?v=" + base["video_id"].astype(str),
    )

    if "story_id" in base.columns:
        base["story_id"] = base["story_id"].apply(ensure_story_id_str)
    else:
        base["story_id"] = None

    # If base table lacks story_id, try to map via video_id.
    if args.videoid_storyid_map and args.videoid_storyid_map.exists() and "video_id" in base.columns:
        try:
            m = read_csv_any_encoding(args.videoid_storyid_map)
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
    if "Reading Age Source" in base.columns:
        base["amazon_age_source"] = base["Reading Age Source"].apply(normalize_age_source)
    elif "amazon_age_source" in base.columns:
        base["amazon_age_source"] = base["amazon_age_source"].apply(normalize_age_source)
    else:
        base["amazon_age_source"] = None
    base["has_amazon_age"] = base.get("Reading Age", pd.Series([None] * len(base))).notna()

    yt = load_youtube_outputs(args.output_dir)
    playlists = load_playlists_from_extracted_titles(args.titles_dir)
    txt = load_story_transcripts(args.transcripts_dir, args.transcripts_zip)
    reviews = load_review_samples(args.review_samples_dir)

    merged = base.merge(yt, on="video_id", how="left")
    merged = merged.merge(playlists, on="video_id", how="left")
    merged = merged.merge(txt, on="story_id", how="left")
    merged = merged.merge(reviews, on="story_id", how="left")

    if args.cleaned_meta is not None and args.cleaned_meta.exists():
        cleaned_meta = load_cleaned_filter(args.cleaned_meta, args.cleaned_col)
        if "story_id" in cleaned_meta.columns and cleaned_meta["story_id"].notna().any():
            merged = merged.merge(
                cleaned_meta[["story_id", "cleaned"]].dropna(subset=["story_id"]).drop_duplicates("story_id"),
                on="story_id",
                how="left",
                suffixes=("", "_cleaned"),
            )
        if "video_id" in cleaned_meta.columns and cleaned_meta["video_id"].notna().any():
            merged = merged.merge(
                cleaned_meta[["video_id", "cleaned"]].dropna(subset=["video_id"]).drop_duplicates("video_id"),
                on="video_id",
                how="left",
                suffixes=("", "_video"),
            )
        if "cleaned_video" in merged.columns:
            merged["cleaned"] = merged.get("cleaned").fillna(merged["cleaned_video"])
            merged = merged.drop(columns=["cleaned_video"])
        merged["cleaned"] = pd.to_numeric(merged.get("cleaned"), errors="coerce")
        if args.cleaned_only:
            merged = merged[merged["cleaned"].fillna(0).astype(int) == 1].copy()

    yt_clean = merged["yt_cleaned_subtitles"] if "yt_cleaned_subtitles" in merged.columns else pd.Series([None] * len(merged))
    yt_src = merged["yt_transcript_source"] if "yt_transcript_source" in merged.columns else pd.Series([None] * len(merged))
    txt_clean = merged["transcript_txt"] if "transcript_txt" in merged.columns else pd.Series([None] * len(merged))
    fallback_source = args.transcripts_source_name or (args.transcripts_zip.stem if args.transcripts_zip else args.transcripts_dir.name)

    merged["transcript_best"] = txt_clean
    merged["transcript_source_best"] = pd.Series([fallback_source] * len(merged), index=merged.index).where(txt_clean.notna(), None)

    fallback_mask = merged["transcript_best"].isna() & yt_clean.notna()
    merged.loc[fallback_mask, "transcript_best"] = yt_clean.loc[fallback_mask]
    merged.loc[fallback_mask, "transcript_source_best"] = yt_src.loc[fallback_mask]

    merged["has_transcript"] = merged["transcript_best"].notna()
    merged["has_events"] = merged["review_events_json"].notna()
    merged["transcript_num_chars"] = merged["transcript_best"].fillna("").map(len)
    merged["transcript_num_words"] = merged["transcript_best"].fillna("").map(lambda s: len(str(s).split()))

    out_written = write_parquet_or_csv(merged, args.out)
    csv_written: Optional[Path] = None
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.out_csv, index=False, compression="gzip" if str(args.out_csv).endswith(".gz") else None)
        csv_written = args.out_csv

    index_cols = [
        "video_id",
        "youtube_url",
        "source_playlists",
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
        "amazon_age_source",
        "has_amazon_age",
        "age_lo",
        "age_hi",
        "cleaned",
        "transcript_source_best",
        "has_transcript",
        "has_events",
    ]
    index_cols = [c for c in index_cols if c in merged.columns]
    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    merged[index_cols].to_csv(args.out_index, index=False)

    print(f"Wrote master: {out_written}")
    if csv_written is not None:
        print(f"Wrote CSV:    {csv_written}")
    print(f"Wrote index:  {args.out_index}")
    print(f"Rows: {len(merged)} | transcripts: {int(merged['has_transcript'].sum())} | events: {int(merged['has_events'].sum())}")


if __name__ == "__main__":
    main()
