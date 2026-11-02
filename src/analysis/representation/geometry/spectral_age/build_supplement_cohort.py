"""Cohort-format records (original stories + detail segments) for the batch videos, from the local story-level export.

``cohort.jsonl`` on the cluster was built with a stated-age filter (customer source, age 2-10) and an older story set, so many batch
videos have no record there.  This writes the same record layout the extractors read (``record_id, record_type, story_key, video_id,
text, segments, source_age`` ...) for every batch video, gzipped, so it can travel through git.  ``filter_cohort.py`` then adds only the
videos the cohort lacks.  Run locally (needs ``data/exports/``):

    python -m analysis.representation.geometry.spectral_age.build_supplement_cohort
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master", type=Path, default=Path("data/exports/story_level_export/story_level_export.jsonl"))
    parser.add_argument("--videos", type=Path, default=Path("data/processed/spectral_age/extraction_videos.csv"))
    parser.add_argument("--ranges", type=Path, default=Path("data/exports/age_ranges/video_age_ranges.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/spectral_age/supplement_cohort.jsonl.gz"))
    args = parser.parse_args(argv)
    with open(args.videos, newline="", encoding="utf-8") as handle:
        wanted = {row["video_id"] for row in csv.DictReader(handle)}
    ranges: dict[str, tuple[int, int, str]] = {}
    with open(args.ranges, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["video_id"] in wanted and row["range_lower"] and row["range_upper"]:
                ranges[row["video_id"]] = (int(float(row["range_lower"])), int(float(row["range_upper"])), row["range_text"])
    written = skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.master, encoding="utf-8") as source, gzip.open(args.output, "wt", encoding="utf-8", compresslevel=9) as target:
        for line in source:
            record = json.loads(line)
            video = record["video_id"]
            if video not in ranges:
                continue
            segments = (record.get("segmented_story") or {}).get("segments") or []
            text = " ".join(str(segment.get("text", "")).strip() for segment in segments).strip()
            if not segments or not text:
                skipped += 1
                continue
            lower, upper, age_text = ranges[video]
            target.write(json.dumps({
                "record_id": f"{record['story_key']}::original", "record_type": "original", "story_key": record["story_key"], "story_id": record.get("story_id"),
                "video_id": video, "condition": "original", "summary_id": 0, "source_age": (lower + upper) / 2, "source_age_range": age_text,
                "target_age": (lower + upper) / 2, "compression_ratio": 1.0, "word_count": len(text.split()), "text": text, "segments": segments,
                "origin": "spectral_age_supplement",
            }, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"batch_videos": len(ranges), "records_written": written, "skipped_without_segments": skipped, "output": str(args.output)}))


if __name__ == "__main__":
    main()
