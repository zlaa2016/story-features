"""Story subsets for the spectral-age analysis: a narrow-range baseline set and flat per-age batches.

``baseline`` keeps every video whose stated age range is narrow (``upper - lower < max_span + 1``
by default, i.e. under 7).  ``select`` then builds a flat batch from it: for each age the
``per_age`` most-viewed videos whose range covers that age (a video with a wide range sits in
every age it covers).  With ``equal`` (the default) every age is cut to the smallest age's size, so
all ages are compared at the same sample size.  The batch folder holds everything a run needs:

* ``subset_videos.csv``  union of the selected videos with their range and log views
* ``membership.csv``     one row per (age, video) with the popularity rank inside that age
* ``summary.json``       candidates, selected size and shortfall per age

The pipeline can rerun ``select`` on the videos that actually have vectors (``--available``),
writing the batch it really ran on into the run folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SPAN = 6


def build_baseline(master_jsonl: Path, ranges_csv: Path, max_span: int = DEFAULT_SPAN) -> pd.DataFrame:
    """Videos with a stated range, at most ``max_span`` ages wide (upper - lower <= max_span)."""
    rows: dict[str, dict] = {}
    with open(master_jsonl, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            meta = record.get("metadata") or {}
            item = rows.setdefault(record["video_id"], {"video_id": record["video_id"], "view_count": meta.get("view_count"),
                                                         "like_count": meta.get("like_count"), "channel": meta.get("channel"), "story_ids": []})
            if record.get("story_id"):
                item["story_ids"].append(str(record["story_id"]))
    videos = pd.DataFrame(rows.values())
    videos["n_stories"] = videos["story_ids"].map(len)
    videos["story_ids"] = videos["story_ids"].map(";".join)
    ranges = pd.read_csv(ranges_csv, low_memory=False)[["video_id", "range_lower", "range_upper", "range_text", "range_source"]]
    frame = videos.merge(ranges, on="video_id", how="inner").dropna(subset=["range_lower", "range_upper"])
    frame["range_lower"], frame["range_upper"] = frame["range_lower"].astype(int), frame["range_upper"].astype(int)
    frame["span"] = frame["range_upper"] - frame["range_lower"]
    frame = frame[frame["span"] <= max_span].copy()
    frame["view_count"] = pd.to_numeric(frame["view_count"], errors="coerce")
    frame["log_views"] = np.log1p(frame["view_count"].fillna(0))
    return frame.sort_values("video_id").reset_index(drop=True)


def select_flat(baseline: pd.DataFrame, ages: list[int], per_age: int, available: set[str] | None = None, equal: bool = True
                ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pool = baseline if available is None else baseline[baseline["video_id"].astype(str).isin(available)]
    pool = pool.sort_values(["log_views", "video_id"], ascending=[False, True], kind="stable")
    candidates = {age: pool[(pool["range_lower"] <= age) & (pool["range_upper"] >= age)] for age in ages}
    size = min([per_age] + ([len(c) for c in candidates.values()] if equal else []))
    parts, summary_ages = [], {}
    for age, found in candidates.items():
        take = found.head(size if equal else min(per_age, len(found)))
        parts.append(pd.DataFrame({"age": age, "video_id": take["video_id"].to_numpy(), "rank": np.arange(1, len(take) + 1), "log_views": take["log_views"].to_numpy()}))
        summary_ages[int(age)] = {"candidates": int(len(found)), "selected": int(len(take)), "short_of_target": int(max(per_age - len(found), 0))}
    membership = pd.concat(parts, ignore_index=True)
    chosen = membership.groupby("video_id")["age"].agg(lambda s: ";".join(str(int(a)) for a in sorted(s))).rename("ages").reset_index()
    subset = chosen.merge(baseline.drop_duplicates("video_id"), on="video_id", how="left")
    summary = {"target_per_age": per_age, "ages": [int(a) for a in ages], "equal_cells": equal, "cell_size": int(size), "by_age": summary_ages,
               "distinct_videos": int(len(subset)), "restricted_to_available_vectors": available is not None}
    return membership, subset, summary


def write_batch(directory: Path, membership: pd.DataFrame, subset: pd.DataFrame, summary: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    membership.to_csv(directory / "membership.csv", index=False)
    columns = ["video_id", "range_lower", "range_upper", "range_text", "log_views", "view_count", "ages", "n_stories", "story_ids", "channel"]
    subset[columns].to_csv(directory / "subset_videos.csv", index=False)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    base = sub.add_parser("baseline", help="narrow-range baseline from the local story-level export")
    base.add_argument("--master", type=Path, default=Path("data/exports/story_level_export/story_level_export.jsonl"))
    base.add_argument("--ranges", type=Path, default=Path("data/exports/age_ranges/video_age_ranges.csv"))
    base.add_argument("--max-span", type=int, default=DEFAULT_SPAN, help="largest upper - lower allowed (6 = under 7)")
    base.add_argument("--output", type=Path, default=Path("data/processed/spectral_age/baseline_span_lt7.csv"))
    pick = sub.add_parser("select", help="flat per-age batch from a baseline")
    pick.add_argument("--baseline", type=Path, default=Path("data/processed/spectral_age/baseline_span_lt7.csv"))
    pick.add_argument("--age-min", type=int, default=2)
    pick.add_argument("--age-max", type=int, default=12)
    pick.add_argument("--per-age", type=int, required=True)
    pick.add_argument("--keep-unequal", action="store_true", help="do not cut every age to the smallest age")
    pick.add_argument("--available", type=Path, default=None, help="text/csv of video_ids that have vectors (column video_id or one id per line)")
    pick.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def read_available(path: Path) -> set[str]:
    if path.suffix == ".csv":
        return set(pd.read_csv(path, usecols=["video_id"], dtype=str)["video_id"])
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "baseline":
        frame = build_baseline(args.master, args.ranges, args.max_span)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"{len(frame)} videos with stated range span <= {args.max_span} -> {args.output}")
        return
    baseline = pd.read_csv(args.baseline, low_memory=False, dtype={"video_id": str})
    ages = list(range(args.age_min, args.age_max + 1))
    available = read_available(args.available) if args.available else None
    membership, subset, summary = select_flat(baseline, ages, args.per_age, available, equal=not args.keep_unequal)
    write_batch(args.output_dir, membership, subset, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_age"}))
    print({age: item["selected"] for age, item in summary["by_age"].items()})


if __name__ == "__main__":
    main()
