"""Describe the analysis dataset: stories, summaries, details, words, ages, and how much has vectors.

    python describe_cohort.py [--vector-dir <folder with contextual_detail_metadata_*.csv>]
"""

from __future__ import annotations

import argparse
import collections
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
    story_conditions = summaries.groupby("story_key").condition.agg(lambda s: set(s))
    print(f"  stories with a summary: {len(story_conditions):,} of {len(originals):,}   (summary details: {summaries.n_details.sum():,}, summary words: {summaries.words.sum():,})")
    print(f"  records in total:     {len(df):,}   (about {int((originals.words.sum() + summaries.words.sum()) * 1.35):,} tokens at 1.35 tokens/word)")
    per_story = originals.set_index("story_key")
    print(f"  words per story:      median {per_story.words.median():.0f}, mean {per_story.words.mean():.0f}, max {per_story.words.max():,}")
    print(f"  details per story:    median {per_story.n_details.median():.0f}, mean {per_story.n_details.mean():.0f}, max {per_story.n_details.max():,}")
    total = len(originals)
    print("\nstories with a summary, by condition (summaries in brackets):")
    for condition in ("actual", "older", "younger"):
        n_stories = sum(condition in c for c in story_conditions)
        n_summaries = int(summaries.condition.eq(condition).sum())
        print(f"  {condition:8s} {n_stories:6,} stories ({n_stories / total:5.1%})   [{n_summaries:,} summaries]")
    print(f"  any      {len(story_conditions):6,} stories ({len(story_conditions) / total:5.1%});  none: {total - len(story_conditions):,}")
    print(f"  two or more conditions: {sum(len(c) >= 2 for c in story_conditions):,}   all three: {sum({'younger', 'actual', 'older'} <= c for c in story_conditions):,}")
    print("  stories by set of conditions:", dict(Counter(tuple(sorted(c)) for c in story_conditions).most_common(7)))
    print("  summaries by segmentation method:", dict(Counter(summaries.method.replace("", "(not recorded)"))))
    ages = originals.source_age.dropna().astype(float)
    print("\nstories by source age (years):", dict(sorted(Counter(np.floor(ages)).items())))
    if args.vector_dir:
        counts = collections.Counter()
        for path in glob.glob(str(args.vector_dir / "contextual_detail_metadata_*.csv")):
            counts.update(pd.read_csv(path, usecols=["record_id"])["record_id"].value_counts().to_dict())
        have = set(counts)
        covered = df[df.record_id.isin(have)].copy()
        print(f"\nwith contextual vectors ({args.vector_dir.name}): {len(covered):,} of {len(df):,} records ({len(covered) / len(df):.0%}); "
              f"{covered.story_key.nunique():,} stories; {int(sum(counts.values())):,} vector rows (details)")
        missing = df[~df.record_id.isin(have)]
        print(f"  records without vectors: {len(missing):,} ({missing.type.value_counts().to_dict()})")
        covered["vector_rows"] = covered.record_id.map(counts)
        print("\ndo the vectors use the same details as the cohort? (vector rows vs cohort details, per record)")
        for kind, group in covered.groupby("type"):
            same = int((group.vector_rows == group.n_details).sum())
            ratio = (group.vector_rows / group.n_details.clip(lower=1))
            print(f"  {kind:9s} records {len(group):6,}   same count {same:6,} ({same / len(group):.0%})   "
                  f"vector rows {int(group.vector_rows.sum()):,} vs cohort details {int(group.n_details.sum()):,}   median ratio {ratio.median():.2f}")
        print("  a median ratio near 0.5 for summaries means the vectors were extracted from coarser (sentence-level) summary segments")

if __name__ == "__main__":
    main()
