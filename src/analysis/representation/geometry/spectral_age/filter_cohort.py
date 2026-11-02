"""Cut the cohort file down to the records of the videos in the spectral-age batches.

The extractors (``extract_global.py``, ``extract_detail_controls.py``) read a cohort jsonl and shard it with
``--start/--limit``.  Running a model on only the batch videos (original stories) is a few thousand records instead
of the whole cohort.  Order: first the cohort's own records (in cohort order, so shard boundaries of an earlier run
stay valid), then, for batch videos the cohort lacks, the records of ``--supplement`` (built from the story-level
export by ``build_supplement_cohort.py``).  ``--exclude-dirs`` drops videos a model already has vectors for
(``global_metadata_*.csv`` in those folders).  Batch videos found in neither source go to ``<output>.missing_videos.csv``.
Standard library only, so it runs on the login node.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
from pathlib import Path


def batch_videos(batch_dirs: list[Path]) -> set[str]:
    wanted: set[str] = set()
    for batch in batch_dirs:
        with open(batch / "subset_videos.csv", newline="", encoding="utf-8") as handle:
            wanted |= {row["video_id"] for row in csv.DictReader(handle)}
    return wanted


def videos_with_vectors(directories: list[Path]) -> set[str]:
    done: set[str] = set()
    for directory in directories:
        for path in glob.glob(str(directory / "global_metadata_*.csv")):
            with open(path, newline="", encoding="utf-8") as handle:
                done |= {row["video_id"] for row in csv.DictReader(handle) if row.get("record_type", "original") == "original"}
    return done


def opened(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=Path, default=Path("results/features/summaries/dataset_index/cohort.jsonl"))
    parser.add_argument("--supplement", type=Path, default=Path("data/processed/spectral_age/supplement_cohort.jsonl.gz"))
    parser.add_argument("--batch-dirs", type=Path, nargs="+", default=[Path("data/processed/spectral_age/batch_top800_ages2-8"), Path("data/processed/spectral_age/batch_top400_ages2-12")])
    parser.add_argument("--exclude-dirs", type=Path, nargs="*", default=[], help="skip videos that already have global vectors in these folders")
    parser.add_argument("--record-types", default="original")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    wanted = batch_videos(args.batch_dirs) - videos_with_vectors(args.exclude_dirs)
    types = {t.strip() for t in args.record_types.split(",")}
    from_cohort: set[str] = set()
    counts = {"cohort": 0, "supplement": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as target:
        with opened(args.cohort) as source:
            for line in source:
                record = json.loads(line)
                video = str(record.get("video_id", ""))
                if record.get("record_type") in types and video in wanted:
                    target.write(line if line.endswith("\n") else line + "\n")
                    from_cohort.add(video)
                    counts["cohort"] += 1
        found = set(from_cohort)
        if args.supplement.exists():
            with opened(args.supplement) as source:
                for line in source:
                    record = json.loads(line)
                    video = str(record.get("video_id", ""))
                    if video in wanted and video not in from_cohort:
                        target.write(line if line.endswith("\n") else line + "\n")
                        found.add(video)
                        counts["supplement"] += 1
    missing = sorted(wanted - found)
    with open(args.output.with_suffix(".missing_videos.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id"])
        writer.writerows([[video] for video in missing])
    print(json.dumps({"wanted_videos": len(wanted), "videos_found": len(found), "missing": len(missing), "records_from_cohort": counts["cohort"],
                      "records_from_supplement": counts["supplement"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
