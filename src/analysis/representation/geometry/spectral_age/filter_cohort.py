"""Cut the cohort file down to the records of the videos in the spectral-age batches.

The extractors (``extract_global.py``, ``extract_detail_controls.py``) read a cohort jsonl and shard it with
``--start/--limit``.  Running a new model on only the batch videos (original stories) is a few thousand
records instead of the whole cohort.  Videos of a batch that the cohort does not contain are listed in
``<output>.missing_videos.csv`` (they need the cohort rebuilt from the newer story export first).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("results/features/summaries/dataset_index/cohort.jsonl"))
    parser.add_argument("--batch-dirs", type=Path, nargs="+", default=[Path("data/processed/spectral_age/batch_top800_ages2-8"), Path("data/processed/spectral_age/batch_top400_ages2-12")])
    parser.add_argument("--record-types", default="original")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    wanted = set()
    for batch in args.batch_dirs:
        wanted |= set(pd.read_csv(batch / "subset_videos.csv", dtype={"video_id": str})["video_id"])
    types = {t.strip() for t in args.record_types.split(",")}
    found: set[str] = set()
    kept = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.cohort, encoding="utf-8") as source, open(args.output, "w", encoding="utf-8") as target:
        for line in source:
            record = json.loads(line)
            video = str(record.get("video_id", ""))
            if record.get("record_type") in types and video in wanted:
                target.write(line if line.endswith("\n") else line + "\n")
                found.add(video)
                kept += 1
    missing = sorted(wanted - found)
    pd.DataFrame({"video_id": missing}).to_csv(args.output.with_suffix(".missing_videos.csv"), index=False)
    print(json.dumps({"batch_videos": len(wanted), "videos_in_cohort": len(found), "missing_from_cohort": len(missing), "records_written": kept, "output": str(args.output)}))


if __name__ == "__main__":
    main()
