"""Story subsets for the spectral-age analysis: a narrow-range baseline set and flat per-age batches.

The unit is the **story** (``story_key``, e.g. ``abc123__story02``): a video can hold several stories, each with its own
text, details and vectors.  A story inherits its video's stated age range and view count.

``baseline`` keeps every story whose stated age range is narrow (``upper - lower <= max_span``, 6 = under 7 ages).
``select`` then builds a flat batch from it: for each age the ``per_age`` most-viewed stories whose range covers that age
(a wide-range story sits in every age it covers).  With ``equal`` (the default) every age is cut to the smallest age's
size, so all ages are compared at the same sample size.  The batch folder holds everything a run needs:

* ``subset_stories.csv``  union of the selected stories with their range, views and the ages they serve
* ``membership.csv``      one row per (age, story) with the popularity rank inside that age
* ``summary.json``        candidates, selected size and shortfall per age

The pipeline can rerun ``select`` on the stories that actually have vectors (``--available``), writing the batch it really
ran on into the run folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SPAN = 6


def build_baseline(master_jsonl: Path, ranges_csv: Path, max_span: int = DEFAULT_SPAN) -> pd.DataFrame:
    """One row per story with a stated range at most ``max_span`` ages wide (upper - lower <= max_span)."""
    rows = []
    with open(master_jsonl, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            meta = record.get("metadata") or {}
            rows.append({"story_key": record["story_key"], "story_id": record.get("story_id") or "", "video_id": record["video_id"],
                         "substory_index": record.get("substory_index"), "n_substories": record.get("n_substories"),
                         "story_title": record.get("story_title_guess"), "view_count": meta.get("view_count"),
                         "like_count": meta.get("like_count"), "channel": meta.get("channel")})
    stories = pd.DataFrame(rows)
    ranges = pd.read_csv(ranges_csv, low_memory=False)[["video_id", "range_lower", "range_upper", "range_text", "range_source"]]
    frame = stories.merge(ranges, on="video_id", how="inner").dropna(subset=["range_lower", "range_upper"])
    frame["range_lower"], frame["range_upper"] = frame["range_lower"].astype(int), frame["range_upper"].astype(int)
    frame["span"] = frame["range_upper"] - frame["range_lower"]
    frame = frame[frame["span"] <= max_span].copy()
    frame["view_count"] = pd.to_numeric(frame["view_count"], errors="coerce")
    frame["log_views"] = np.log1p(frame["view_count"].fillna(0))
    return frame.sort_values("story_key").reset_index(drop=True)


def select_flat(baseline: pd.DataFrame, ages: list[int], per_age: int, available: set[str] | None = None, equal: bool = True
                ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pool = baseline if available is None else baseline[baseline["story_key"].astype(str).isin(available)]
    pool = pool.sort_values(["log_views", "story_key"], ascending=[False, True], kind="stable")
    candidates = {age: pool[(pool["range_lower"] <= age) & (pool["range_upper"] >= age)] for age in ages}
    size = min([per_age] + ([len(c) for c in candidates.values()] if equal else []))
    parts, summary_ages = [], {}
    for age, found in candidates.items():
        take = found.head(size if equal else min(per_age, len(found)))
        parts.append(pd.DataFrame({"age": age, "story_key": take["story_key"].to_numpy(), "rank": np.arange(1, len(take) + 1), "log_views": take["log_views"].to_numpy()}))
        summary_ages[int(age)] = {"candidates": int(len(found)), "selected": int(len(take)), "short_of_target": int(max(per_age - len(found), 0))}
    membership = pd.concat(parts, ignore_index=True)
    chosen = membership.groupby("story_key")["age"].agg(lambda s: ";".join(str(int(a)) for a in sorted(s))).rename("ages").reset_index()
    subset = chosen.merge(baseline.drop_duplicates("story_key"), on="story_key", how="left")
    summary = {"unit": "story", "target_per_age": per_age, "ages": [int(a) for a in ages], "equal_cells": equal, "cell_size": int(size), "by_age": summary_ages,
               "distinct_stories": int(len(subset)), "distinct_videos": int(subset["video_id"].nunique()), "restricted_to_available_vectors": available is not None}
    return membership, subset, summary


def write_batch(directory: Path, membership: pd.DataFrame, subset: pd.DataFrame, summary: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    membership.to_csv(directory / "membership.csv", index=False)
    columns = ["story_key", "story_id", "video_id", "story_title", "range_lower", "range_upper", "range_text", "log_views", "view_count", "ages", "n_substories", "channel"]
    subset[columns].to_csv(directory / "subset_stories.csv", index=False)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    base = sub.add_parser("baseline", help="narrow-range baseline from the local story-level export")
    base.add_argument("--master", type=Path, default=Path("data/exports/story_level_export/story_level_export.jsonl"))
    base.add_argument("--ranges", type=Path, default=Path("data/exports/age_ranges/video_age_ranges.csv"))
    base.add_argument("--max-span", type=int, default=DEFAULT_SPAN, help="largest upper - lower allowed (6 = under 7)")
    base.add_argument("--output", type=Path, default=Path("data/processed/spectral_age/baseline_stories_span_lt7.csv"))
    pick = sub.add_parser("select", help="flat per-age batch from a baseline")
    pick.add_argument("--baseline", type=Path, default=Path("data/processed/spectral_age/baseline_stories_span_lt7.csv"))
    pick.add_argument("--age-min", type=int, default=2)
    pick.add_argument("--age-max", type=int, default=12)
    pick.add_argument("--per-age", type=int, required=True)
    pick.add_argument("--keep-unequal", action="store_true", help="do not cut every age to the smallest age")
    pick.add_argument("--available", type=Path, default=None, help="text/csv of story_keys that have vectors (column story_key or one key per line)")
    pick.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def read_available(path: Path) -> set[str]:
    if path.suffix == ".csv":
        return set(pd.read_csv(path, usecols=["story_key"], dtype=str)["story_key"])
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "baseline":
        frame = build_baseline(args.master, args.ranges, args.max_span)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"{len(frame)} stories ({frame['video_id'].nunique()} videos) with stated range span <= {args.max_span} -> {args.output}")
        return
    baseline = pd.read_csv(args.baseline, low_memory=False, dtype={"story_key": str, "video_id": str})
    ages = list(range(args.age_min, args.age_max + 1))
    available = read_available(args.available) if args.available else None
    membership, subset, summary = select_flat(baseline, ages, args.per_age, available, equal=not args.keep_unequal)
    write_batch(args.output_dir, membership, subset, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_age"}))
    print({age: item["selected"] for age, item in summary["by_age"].items()})


if __name__ == "__main__":
    main()
