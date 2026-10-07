#!/usr/bin/env python3
"""Report planned probe size and any completed shards without loading model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from shared.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("results/features/summaries/dataset_index/cohort.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", default="isolated,true,shuffled,foreign")
    parser.add_argument("--permutations", type=int, default=1)
    parser.add_argument("--sample-stories", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=2026)
    args = parser.parse_args()

    originals = [row for row in read_jsonl(args.cohort) if row.get("record_type") == "original"]
    if 0 < args.sample_stories < len(originals):
        import numpy as np

        rng = np.random.default_rng(args.sample_seed)
        selected = np.sort(rng.choice(len(originals), size=args.sample_stories, replace=False))
        originals = [originals[int(index)] for index in selected]
    counts = [len([segment for segment in row.get("segments", []) if str(segment.get("text", "")).strip()]) for row in originals]
    conditions = {value.strip() for value in args.conditions.split(",") if value.strip()}
    repeated = sum(args.permutations if name in {"shuffled", "foreign"} else 1 for name in conditions - {"isolated"})
    native_rows = sum(counts) * (repeated + int("isolated" in conditions))
    all_rows = sum(n * (1 + (n + 1) * repeated) for n in counts) if "isolated" in conditions else sum(n * (n + 1) * repeated for n in counts)

    metadata_paths = sorted(args.output_dir.glob("probe_metadata_*.csv"))
    completed_rows = 0
    completed_stories: set[str] = set()
    shard_rows: list[dict] = []
    for metadata_path in metadata_paths:
        suffix = metadata_path.stem.removeprefix("probe_metadata_")
        feature = args.output_dir / f"probe_features_{suffix}.npz"
        frame = pd.read_csv(metadata_path, usecols=["story_key"])
        status = "complete" if feature.exists() else "metadata_only"
        if feature.exists():
            completed_rows += len(frame)
            completed_stories.update(frame["story_key"].astype(str))
        shard_rows.append({"shard": suffix, "status": status, "rows": len(frame), "stories": frame.story_key.nunique()})

    print(f"Original records: {len(originals):,}")
    print(f"Original details: {sum(counts):,}")
    if counts:
        print(f"Details/story: min={min(counts)} median={pd.Series(counts).median():.1f} max={max(counts)}")
    print(f"Estimated native probe rows: {native_rows:,}")
    print(f"Estimated all-prefix probe rows: {all_rows:,}")
    print(f"Completed shards in {args.output_dir}: {len(metadata_paths)}")
    print(f"Completed stories: {len(completed_stories):,}; completed rows: {completed_rows:,}")
    if shard_rows:
        print(pd.DataFrame(shard_rows).to_string(index=False))
    for config_path in sorted(args.output_dir.glob("probe_config_*.json")):
        config = json.loads(config_path.read_text())
        print(
            f"{config_path.name}: mode={config.get('prefix_mode')} start={config.get('start')} "
            f"limit={config.get('limit')} rows={config.get('probe_rows')} layers={config.get('selected_layers')}"
        )


if __name__ == "__main__":
    main()
