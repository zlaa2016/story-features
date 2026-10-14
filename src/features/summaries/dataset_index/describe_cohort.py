"""Describe the analysis dataset: stories, summaries, details, words, ages, and how much has vectors.

    python describe_cohort.py [--vector-dir <folder with contextual_detail_metadata_*.csv>]
"""

from __future__ import annotations

import argparse
import glob
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from shared.io import read_jsonl, word_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("results/features/summaries/dataset_index/cohort.jsonl"))
    parser.add_argument("--vector-dir", type=Path, default=None, help="Optional: contextual shard folder, to report vector coverage")
    args = parser.parse_args()

    rows = []
    for r in read_jsonl(args.cohort):
        segments = r.get("segments") or []
        rows.append({
            "record_id": r["record_id"], "story_key": r["story_key"], "type": r["record_type"], "condition": r.get("condition") or "",
            "summary_id": r.get("summary_id"), "method": r.get("segmentation_method") or "", "source_age": r.get("source_age"),
            "target_age": r.get("target_age"), "n_details": len(segments), "words": r.get("word_count") or word_count(r.get("text", "")),
        })
    df = pd.DataFrame(rows)
    originals, summaries = df[df.type.eq("original")], df[df.type.eq("summary")]
    print(f"cohort: {args.cohort}")
    print(f"  stories (originals):  {len(originals):,}   original details: {originals.n_details.sum():,}   original words: {originals.words.sum():,}")
    print(f"  summaries:            {len(summaries):,}   summary details:  {summaries.n_details.sum():,}   summary words:  {summaries.words.sum():,}")
    print(f"  records in total:     {len(df):,}   (about {int((originals.words.sum() + summaries.words.sum()) * 1.35):,} tokens at 1.35 tokens/word)")
    per_story = originals.set_index("story_key")
    print(f"  words per story:      median {per_story.words.median():.0f}, mean {per_story.words.mean():.0f}, max {per_story.words.max():,}")
    print(f"  details per story:    median {per_story.n_details.median():.0f}, mean {per_story.n_details.mean():.0f}, max {per_story.n_details.max():,}")
    print("\nsummaries by condition:")
    print(summaries.groupby("condition").agg(summaries=("record_id", "size"), details=("n_details", "sum")).to_string())
    print("\nsummaries by segmentation method:", dict(Counter(summaries.method.replace("", "(not recorded)"))))
    conds = summaries.groupby("story_key").condition.agg(lambda s: tuple(sorted(set(s))))
    print("stories by set of conditions:", dict(Counter(conds).most_common(6)))
    print(f"stories with younger+actual+older: {sum(1 for c in conds if {'younger', 'actual', 'older'} <= set(c)):,}   stories with no summary: {len(originals) - len(conds):,}")
    ages = originals.source_age.dropna().astype(float)
    print("\nstories by source age (years):", dict(sorted(Counter(np.floor(ages)).items())))
    if args.vector_dir:
        have = set()
        for path in glob.glob(str(args.vector_dir / "contextual_detail_metadata_*.csv")):
            have |= set(pd.read_csv(path, usecols=["record_id"])["record_id"])
        covered = df[df.record_id.isin(have)]
        print(f"\nwith contextual vectors ({args.vector_dir.name}): {len(covered):,} of {len(df):,} records ({len(covered) / len(df):.0%}); "
              f"{covered.story_key.nunique():,} stories; {covered.n_details.sum():,} details")
        missing = df[~df.record_id.isin(have)]
        print(f"  records without vectors: {len(missing):,} ({missing.type.value_counts().to_dict()})")


if __name__ == "__main__":
    main()
