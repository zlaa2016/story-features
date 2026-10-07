from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.io import read_jsonl, write_json


TOKEN_PATTERN = re.compile(r"\w+(?:[-']\w+)?", flags=re.UNICODE)
VALID_RECORD_TYPES = {"original", "summary"}
VALID_CONDITIONS = {"original", "younger", "actual", "older"}


def lexical_tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(str(text))]


def validate_records(records: list[dict[str, Any]], minimum_age: float, maximum_age: float) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    record_ids = [str(row.get("record_id", "")) for row in records]
    duplicate_ids = [key for key, count in Counter(record_ids).items() if count > 1]
    if duplicate_ids:
        errors.append(f"duplicate record_id values: {len(duplicate_ids)}")
    originals = [row for row in records if row.get("record_type") == "original"]
    original_counts = Counter(str(row.get("story_key", "")) for row in originals)
    invalid_original_counts = [key for key, count in original_counts.items() if count != 1]
    if invalid_original_counts:
        errors.append(f"stories without exactly one original: {len(invalid_original_counts)}")
    original_stories = set(original_counts)
    segment_mismatches = []
    for row_number, row in enumerate(records, start=1):
        record_id = str(row.get("record_id", ""))
        record_type = str(row.get("record_type", ""))
        condition = str(row.get("condition", ""))
        story_key = str(row.get("story_key", ""))
        if not record_id or not story_key:
            errors.append(f"row {row_number} has an empty identifier")
        if record_type not in VALID_RECORD_TYPES:
            errors.append(f"{record_id}: invalid record_type {record_type!r}")
        if condition not in VALID_CONDITIONS:
            errors.append(f"{record_id}: invalid condition {condition!r}")
        if record_type == "original" and condition != "original":
            errors.append(f"{record_id}: original record has condition {condition!r}")
        if record_type == "summary" and story_key not in original_stories:
            errors.append(f"{record_id}: summary has no original")
        text_tokens = lexical_tokens(row.get("text", ""))
        segments = row.get("segments")
        if not text_tokens:
            errors.append(f"{record_id}: empty text")
        if not isinstance(segments, list) or not segments:
            errors.append(f"{record_id}: missing segments")
            continue
        segment_ids = [segment.get("segment_id") for segment in segments if isinstance(segment, dict)]
        if len(segment_ids) != len(segments) or segment_ids != list(range(1, len(segments) + 1)):
            errors.append(f"{record_id}: segment IDs are not contiguous from 1")
        segment_tokens = lexical_tokens(" ".join(str(segment.get("text", "")) for segment in segments))
        if not segment_tokens:
            errors.append(f"{record_id}: all segment text is empty")
        if text_tokens != segment_tokens:
            mismatch = abs(len(text_tokens) - len(segment_tokens)) / max(len(text_tokens), 1)
            segment_mismatches.append({"record_id": record_id, "relative_token_count_difference": mismatch})
        compression = pd.to_numeric(pd.Series([row.get("compression_ratio")]), errors="coerce").iloc[0]
        if not np.isfinite(compression) or compression <= 0:
            errors.append(f"{record_id}: invalid compression_ratio")
        if record_type == "summary":
            age = pd.to_numeric(pd.Series([row.get("target_age")]), errors="coerce").iloc[0]
            if not np.isfinite(age) or not minimum_age <= age <= maximum_age:
                errors.append(f"{record_id}: target_age outside [{minimum_age}, {maximum_age}]")
    if segment_mismatches:
        warnings.append(f"segment/text lexical sequence mismatch: {len(segment_mismatches)} records")
    return {
        "valid": not errors,
        "n_records": len(records),
        "n_stories": len(original_stories),
        "n_originals": len(originals),
        "n_summaries": len(records) - len(originals),
        "errors": errors,
        "warnings": warnings,
        "segment_mismatch_count": len(segment_mismatches),
        "segment_mismatch_examples": segment_mismatches[:20],
    }


def validate_condition_index(frame: pd.DataFrame, story_keys: set[str]) -> list[str]:
    errors: list[str] = []
    required = {"story_key", "condition", "condition_eligible", "has_segmented_summary"}
    missing = required - set(frame.columns)
    if missing:
        return [f"condition index missing columns: {sorted(missing)}"]
    if frame.duplicated(["story_key", "condition"]).any():
        errors.append("condition index contains duplicate story-condition rows")
    unknown = set(frame["story_key"].astype(str)) - story_keys
    if unknown:
        errors.append(f"condition index contains {len(unknown)} stories absent from cohort originals")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the canonical mechanistic cohort before GPU experiments.")
    parser.add_argument("--input-dir", type=Path, default=Path("results/features/summaries/dataset_index"))
    parser.add_argument("--minimum-age", type=float, default=2)
    parser.add_argument("--maximum-age", type=float, default=18)
    args = parser.parse_args()
    records = list(read_jsonl(args.input_dir / "cohort.jsonl"))
    report = validate_records(records, args.minimum_age, args.maximum_age)
    condition_errors = validate_condition_index(
        pd.read_csv(args.input_dir / "condition_index.csv", low_memory=False),
        {str(row["story_key"]) for row in records if row.get("record_type") == "original"},
    )
    report["errors"].extend(condition_errors)
    report["valid"] = not report["errors"]
    write_json(args.input_dir / "validation.json", report)
    if not report["valid"]:
        raise SystemExit("Cohort validation failed:\n- " + "\n- ".join(report["errors"][:30]))
    print(f"Validated {report['n_records']} records from {report['n_stories']} stories")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
