from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate summary-complete structured alignments.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args()

    summary = pd.read_csv(args.input_dir / "summary_detail_alignment.csv", low_memory=False)
    stories = pd.read_csv(args.input_dir / "story_alignment_metrics.csv", low_memory=False)
    errors_path = args.input_dir / "errors.csv"
    try:
        errors = pd.read_csv(errors_path, low_memory=False) if errors_path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        errors = pd.DataFrame()

    keys = ["story_key", "condition", "summary_id", "summary_variant"]
    required = keys + [
        "summary_index", "source_start", "source_end", "source_span_length",
        "alignment_cosine", "transport_null",
    ]
    missing = [column for column in required if column not in summary]
    if missing:
        raise ValueError(f"Missing summary-alignment columns: {missing}")
    duplicate_rows = int(summary.duplicated(keys + ["summary_index"]).sum())
    invalid_spans = int(
        (
            (summary["source_start"] > summary["source_end"])
            | ((summary["source_end"] - summary["source_start"] + 1) != summary["source_span_length"])
        ).sum()
    )
    incomplete_groups = 0
    for _, group in summary.groupby(keys, dropna=False):
        observed = sorted(group["summary_index"].astype(int).tolist())
        if observed != list(range(len(group))):
            incomplete_groups += 1
    counts = summary.groupby(keys, dropna=False).size().rename("observed_summary_details").reset_index()
    story_counts = stories[keys + ["n_summary_details", "summary_coverage_rate"]].merge(
        counts, on=keys, how="left"
    )
    count_mismatches = int(
        story_counts["n_summary_details"].ne(story_counts["observed_summary_details"]).sum()
    )
    nonfinite_cosines = int((~np.isfinite(pd.to_numeric(summary["alignment_cosine"], errors="coerce"))).sum())
    report = {
        "aligned_summary_records": int(len(stories)),
        "aligned_summary_details": int(len(summary)),
        "duplicate_summary_details": duplicate_rows,
        "invalid_source_spans": invalid_spans,
        "incomplete_summary_index_groups": incomplete_groups,
        "summary_count_mismatches": count_mismatches,
        "nonfinite_alignment_cosines": nonfinite_cosines,
        "mean_summary_coverage_rate": float(stories["summary_coverage_rate"].mean()),
        "null_assigned_summary_details": int(summary["transport_null"].astype(bool).sum()),
        "processing_errors": int(len(errors)),
    }
    write_json(args.input_dir / "validation.json", report)
    failures = duplicate_rows + invalid_spans + incomplete_groups + count_mismatches + nonfinite_cosines
    if len(errors) and not args.allow_errors:
        failures += len(errors)
    print(report)
    if failures:
        raise SystemExit(f"Alignment validation failed with {failures} violations")


if __name__ == "__main__":
    main()
