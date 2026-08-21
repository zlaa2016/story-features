from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mechanistic_interpretability.shared.io import write_csv


TABLES = [
    "original_detail_retention.csv",
    "summary_detail_alignment.csv",
    "story_alignment_metrics.csv",
    "window_length_scores.csv",
    "errors.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine structured-alignment array outputs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    for filename in TABLES:
        paths = sorted(args.input_dir.glob(f"shard_*/{filename}"))
        frames = [pd.read_csv(path, low_memory=False) for path in paths if path.stat().st_size > 1]
        output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        write_csv(args.input_dir / filename, output)
        print(f"{filename}: {len(output)} rows from {len(frames)} shards")


if __name__ == "__main__":
    main()
