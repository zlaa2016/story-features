"""Batch of the legacy event-segmented stories that cover ages 2-12, in the folder layout of spectral_age batches.

Writes ``subset_stories.csv`` (one row per story: story_key, story_id, video_id, view_count, age_lo, age_hi, event_file) and
``membership.csv`` (age, story_key: a story is listed once for every age 2-12 that its reading-age range covers), so the
deletion-graph extraction and analysis run on it unchanged.  ``story_key`` is ``<video_id>__story01``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from features.representation_extraction.causal_lm.detail_extraction.event_segments import event_files


def build_batch(master: pd.DataFrame, files: dict[int, Path], age_min: int = 2, age_max: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    stories = master[master["story_id"].isin(files)].drop_duplicates("story_id").dropna(subset=["age_lo", "age_hi"]).copy()
    stories = stories[(stories["age_lo"] <= age_max) & (stories["age_hi"] >= age_min)]
    stories["story_id"] = stories["story_id"].astype(int)
    stories["story_key"] = stories["video_id"].astype(str) + "__story01"
    stories["view_count"] = pd.to_numeric(stories.get("yt_view_count"), errors="coerce").fillna(0).astype(int)
    stories["event_file"] = [files[i].name for i in stories["story_id"]]
    subset = stories[["story_key", "story_id", "video_id", "view_count", "age_lo", "age_hi", "event_file"]]
    subset = subset.sort_values(["view_count", "story_key"], ascending=[False, True]).reset_index(drop=True)
    rows = [
        {"age": age, "story_key": row.story_key}
        for row in subset.itertuples() for age in range(max(age_min, int(row.age_lo)), min(age_max, int(row.age_hi)) + 1)
    ]
    return subset, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=Path("data/processed/master_stories.csv.gz"))
    parser.add_argument("--event-dir", type=Path, default=Path("data/legacy/review_samples_event_segments"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/spectral_age/batch_events_ages2-12"))
    parser.add_argument("--age-min", type=int, default=2)
    parser.add_argument("--age-max", type=int, default=12)
    args = parser.parse_args()
    master = pd.read_csv(args.master, low_memory=False, usecols=["story_id", "video_id", "age_lo", "age_hi", "yt_view_count"])
    subset, membership = build_batch(master, event_files(args.event_dir), args.age_min, args.age_max)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output_dir / "subset_stories.csv", index=False)
    membership.to_csv(args.output_dir / "membership.csv", index=False)
    print(f"{len(subset)} stories, {len(membership)} story-age rows -> {args.output_dir}")
    print(membership.groupby("age").size().to_string())


if __name__ == "__main__":
    main()
