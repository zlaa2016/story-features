#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from amazon_scraper_common import load_master_csv, save_master_csv, update_master_with_title_result


DEFAULT_MODEL = os.environ.get("OPENAI_TITLE_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You extract structured book metadata from YouTube video metadata.

Given only a YouTube video title and its description, identify:
- book_title: the overall book or collection title
- story_title: the specific story title if different from the book title
- author: the author name if reasonably inferable

Rules:
- Use only evidence from the title and description.
- Do not use outside knowledge.
- It is acceptable if the work is not a children's book, as long as it is plausibly a book, story, tale, anthology, audiobook, read-aloud, chapter reading, literary work, or other book-like text being presented in the video.
- If the title/description clearly indicates a specific story, chapter, episode, or segment from a larger book or collection, use the larger work as `book_title` when inferable and the specific segment as `story_title`.
- Prefer extracting a plausible book/work title over failing when the video appears to present a book or written story.
- If unknown, return "N/A".
- If the video appears to be non-book media only (for example only a TV episode, toy property, movie clip, song, gameplay, or unrelated video) and no book-like work can be inferred from the title/description, return "N/A" fields and extraction_status = "failed".
- Prefer conservative extraction over guessing.

Return valid JSON only with exactly these keys:
{
  "book_title": "...",
  "story_title": "...",
  "author": "...",
  "extraction_status": "success" or "failed"
}
"""


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


def normalize_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def is_truthy(value: Any) -> bool:
    text = normalize_string(value).lower()
    return text in {"1", "true", "t", "yes", "y"}


def load_api_key(key_path: Optional[Path]) -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    if key_path and key_path.exists():
        os.environ["OPENAI_API_KEY"] = key_path.read_text(encoding="utf-8").strip()


def build_user_prompt(video_title: str, description: str) -> str:
    return (
        "Return JSON only.\n\n"
        f"Video title:\n{video_title or 'N/A'}\n\n"
        f"Video description:\n{description or 'N/A'}\n"
    )


def call_openai(model: str, video_title: str, description: str) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=build_user_prompt(video_title, description),
        text={"format": {"type": "json_object"}},
        store=False,
    )
    payload = json.loads(response.output_text)
    return {
        "book_title": normalize_string(payload.get("book_title")) or "N/A",
        "story_title": normalize_string(payload.get("story_title")) or "N/A",
        "author": normalize_string(payload.get("author")) or "N/A",
        "extraction_status": normalize_string(payload.get("extraction_status")).lower() or "failed",
    }


def choose_source_csv(row: pd.Series) -> str:
    for col in ["source_csv", "transcript_source_csv", "playlist"]:
        value = normalize_string(row.get(col))
        if value:
            if col == "playlist" and not value.endswith(".csv"):
                return f"{value}.csv"
            return value
    return "unknown_source.csv"


def title_output_name(source_csv: str) -> str:
    stem = Path(source_csv).stem
    return f"{stem}_titles.csv"


def load_input_rows(queue_path: Path, status_column: Optional[str] = None) -> pd.DataFrame:
    df = read_csv_any_encoding(queue_path)
    if status_column and status_column in df.columns:
        df = df[df[status_column].map(is_truthy)].copy()
    needed = [c for c in ["video_id", "video_title", "description", "source_csv", "transcript_source_csv", "playlist"] if c in df.columns]
    df = df[needed].copy()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["video_title"] = df["video_title"].map(normalize_string)
    if "description" in df.columns:
        df["description"] = df["description"].map(normalize_string)
    else:
        df["description"] = ""
    df["source_csv"] = df.apply(choose_source_csv, axis=1)
    return df


def write_results(
    results: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    if not results:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    for source_csv, group in result_df.groupby("source_csv", sort=True):
        out_path = out_dir / title_output_name(source_csv)
        if out_path.exists():
            existing = read_csv_any_encoding(out_path)
        else:
            existing = pd.DataFrame(
                columns=[
                    "book_title",
                    "story_title",
                    "author",
                    "source_csv",
                    "video_id",
                    "video_title",
                    "description",
                    "extraction_status",
                ]
            )
        merged = pd.concat([existing, group], ignore_index=True)
        merged = merged.drop_duplicates(subset=["video_id"], keep="last")
        merged = merged[
            [
                "book_title",
                "story_title",
                "author",
                "source_csv",
                "video_id",
                "video_title",
                "description",
                "extraction_status",
            ]
        ]
        merged.to_csv(out_path, index=False)


def summarize_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "no rows"
    df = pd.DataFrame(results)
    status_counts = (
        df["extraction_status"].fillna("NA").astype(str).value_counts().to_dict()
        if "extraction_status" in df.columns
        else {}
    )
    source_counts = (
        df["source_csv"].fillna("NA").astype(str).value_counts().head(5).to_dict()
        if "source_csv" in df.columns
        else {}
    )
    return f"statuses={status_counts} top_sources={source_counts}"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description="Extract book/story titles from YouTube video title + description only.")
    ap.add_argument("--queue", type=Path, default=Path("data/processed/youtube_catalog.csv.gz"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/extracted_titles"))
    ap.add_argument("--master-csv", type=Path, default=None, help="Optional master status CSV to update in place.")
    ap.add_argument("--status-column", type=str, default="needs_title_extraction")
    ap.add_argument("--run-label", type=str, default=None, help="Optional run label; writes outputs into out-dir/<run-label>/")
    ap.add_argument("--key-path", type=Path, default=None, help="Optional path to an OpenAI API key file.")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--flush-every", type=int, default=25, help="Write intermediate results every N rows.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.run_label:
        args.out_dir = args.out_dir / args.run_label

    load_api_key(args.key_path)
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Pass --key-path or export OPENAI_API_KEY.")

    rows = load_input_rows(args.queue, status_column=args.status_column)
    master_df = load_master_csv(args.master_csv)
    queue_rows_before_slice = len(rows)
    if args.start:
        rows = rows.iloc[args.start:]
    if args.limit is not None:
        rows = rows.iloc[: args.limit]
    total_rows = len(rows)
    print(
        f"Starting title extraction | queue={args.queue} | total_queue_rows={queue_rows_before_slice} "
        f"| start={args.start} | limit={args.limit} | selected_rows={total_rows} | out_dir={args.out_dir}",
        flush=True,
    )
    if total_rows:
        top_sources = rows["source_csv"].fillna("NA").astype(str).value_counts().head(10).to_dict()
        print(f"Selected row source distribution (top 10): {top_sources}", flush=True)

    results: List[Dict[str, Any]] = []
    for index, row in enumerate(rows.itertuples(index=False), start=1):
        print(
            f"Processing row {index}/{total_rows} | video_id={row.video_id} | source_csv={row.source_csv} | "
            f"video_title={row.video_title[:100]}",
            flush=True,
        )
        if args.dry_run:
            payload = {
                "book_title": "N/A",
                "story_title": "N/A",
                "author": "N/A",
                "extraction_status": "dry_run",
            }
        else:
            payload = call_openai(args.model, row.video_title, row.description)
            time.sleep(args.sleep)
        results.append(
            {
                **payload,
                "source_csv": row.source_csv,
                "video_id": row.video_id,
                "video_title": row.video_title,
                "description": row.description,
            }
        )
        update_master_with_title_result(master_df, pd.Series(row._asdict()), payload)
        if args.flush_every > 0 and len(results) >= args.flush_every:
            write_results(results, args.out_dir)
            save_master_csv(master_df, args.master_csv)
            print(
                f"Checkpointed rows 1..{index} into {args.out_dir} | "
                f"batch_rows={len(results)} | {summarize_results(results)}",
                flush=True,
            )
            results = []

    remaining = len(results)
    write_results(results, args.out_dir)
    save_master_csv(master_df, args.master_csv)
    print(
        f"Finished run | selected_rows={total_rows} | remaining_batch_rows={remaining} "
        f"| {summarize_results(results)} | out_dir={args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
