"""Pick stories that have younger, actual and older summaries, flat across source age.

Writes one row per story with the record id of the lowest-numbered summary in each
condition. Every compression-search method reads this same file, so they are compared
on identical stories.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared.io import write_csv

CONDITIONS = ("younger", "actual", "older")


def method_audit(metadata: pd.DataFrame) -> pd.DataFrame:
    """How many summaries were segmented by each method (empty = the method was not recorded)."""
    summaries = metadata.loc[metadata["record_type"].eq("summary")]
    if "segmentation_method" not in summaries:
        return pd.DataFrame({"segmentation_method": ["(column missing)"], "summaries": [len(summaries)]})
    return (summaries["segmentation_method"].fillna("").replace("", "(not recorded)").value_counts()
            .rename_axis("segmentation_method").reset_index(name="summaries"))


def complete_stories(metadata: pd.DataFrame, conditions: tuple[str, ...] = CONDITIONS, require_method: str = "") -> pd.DataFrame:
    originals = metadata.loc[metadata["record_type"].eq("original"), ["story_key", "source_age"]].drop_duplicates("story_key")
    summaries = metadata.loc[metadata["record_type"].eq("summary") & metadata["condition"].isin(conditions)].copy()
    if require_method:
        # Summary details from different segmenters are different units (local_sentence gives about half as many
        # segments as api_detail), so compression ratios are only comparable within one method.
        summaries = summaries.loc[summaries["segmentation_method"].eq(require_method)]
    summaries["summary_id"] = pd.to_numeric(summaries["summary_id"], errors="coerce").fillna(0).astype(int)
    first = summaries.sort_values(["story_key", "condition", "summary_id"]).drop_duplicates(["story_key", "condition"])
    wide = first.pivot(index="story_key", columns="condition", values="record_id")
    wide = wide.dropna(subset=list(conditions)).rename(columns={c: f"{c}_record_id" for c in conditions})
    out = originals.merge(wide.reset_index(), on="story_key", how="inner")
    out["original_record_id"] = out["story_key"].astype(str) + "::original"
    out["source_age"] = pd.to_numeric(out["source_age"], errors="coerce")
    return out.dropna(subset=["source_age"]).reset_index(drop=True)


def flatten_by_age(stories: pd.DataFrame, bin_width: float, per_bin: int, min_bin: int, seed: int) -> pd.DataFrame:
    stories = stories.copy()
    stories["age_bin"] = (np.floor(stories["source_age"] / bin_width) * bin_width).astype(float)
    counts = stories["age_bin"].value_counts()
    if per_bin <= 0:
        eligible = counts[counts >= min_bin]
        per_bin = int(eligible.min()) if len(eligible) else int(counts.max())
    rng = np.random.default_rng(seed)
    kept = []
    for _, group in stories.groupby("age_bin"):
        if len(group) > per_bin:
            group = group.iloc[np.sort(rng.choice(len(group), size=per_bin, replace=False))]
        kept.append(group)
    return pd.concat(kept).sort_values(["age_bin", "story_key"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-metadata", type=Path, default=Path("results/features/summaries/dataset_index/cohort_metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/features/summaries/dataset_index/flat_subset.csv"))
    parser.add_argument("--age-bin-width", type=float, default=1.0)
    parser.add_argument("--per-bin", type=int, default=0, help="Stories per age bin; 0 = size of the smallest bin with at least --min-bin stories.")
    parser.add_argument("--min-bin", type=int, default=5)
    parser.add_argument("--no-flatten", action="store_true", help="Keep every story that has all three conditions.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--require-method", default="", help="Keep only summaries segmented by this method (for example api_detail); default keeps all.")
    args = parser.parse_args()

    metadata = pd.read_csv(args.cohort_metadata, low_memory=False)
    print("Summary segmentation methods in the cohort:")
    print(method_audit(metadata).to_string(index=False))
    stories = complete_stories(metadata, require_method=args.require_method)
    if args.require_method:
        everyone = complete_stories(metadata)
        print(f"--require-method {args.require_method}: {len(stories)} stories with all three summaries, against {len(everyone)} without the filter")
    if stories.empty:
        raise SystemExit("No story has younger, actual and older summaries")
    stories["age_bin"] = (np.floor(stories["source_age"] / args.age_bin_width) * args.age_bin_width).astype(float)
    before = stories["age_bin"].value_counts().sort_index()
    subset = stories if args.no_flatten else flatten_by_age(stories, args.age_bin_width, args.per_bin, args.min_bin, args.seed)
    after = subset["age_bin"].value_counts().sort_index()
    write_csv(args.output, subset)
    table = pd.DataFrame({"complete_stories": before, "selected": after}).fillna(0).astype(int)
    write_csv(args.output.with_name(args.output.stem + "_by_age.csv"), table.reset_index(names="age_bin"))
    print(f"{len(stories)} stories have all three conditions; selected {len(subset)}")
    print(table.to_string())


if __name__ == "__main__":
    main()
