from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from shared.ages import AgeCondition, age_midpoint, relative_conditions
from shared.io import normalize_text, read_jsonl, text_hash, word_count, write_csv, write_jsonl


TITLE_FRAGMENT_RE = re.compile(r"^[\"'“”‘’]?(?:mr|mrs|ms|dr|prof|st|mir)\.?[\"'“”‘’]?$", re.IGNORECASE)


def repair_title_fragments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge isolated honorifics created by detail segmentation without dropping text."""
    repaired: list[dict[str, Any]] = []
    index = 0
    while index < len(segments):
        segment = dict(segments[index])
        text = normalize_text(segment.get("text"))
        if TITLE_FRAGMENT_RE.fullmatch(text) and index + 1 < len(segments):
            following = normalize_text(segments[index + 1].get("text"))
            if following:
                segment["text"] = f"{text} {following}"
                index += 1
        repaired.append(segment)
        index += 1

    for segment_id, segment in enumerate(repaired, start=1):
        if "segment_id" in segment:
            segment["segment_id"] = segment_id
    return repaired


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generation_provenance(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            response = row.get("response") or {}
            for index, summary in enumerate(response.get("summaries") or [], start=1):
                if not isinstance(summary, dict):
                    continue
                text = normalize_text(summary.get("summary"))
                if not text:
                    continue
                by_hash[text_hash(text)].append(
                    {
                        "generator_model": normalize_text(row.get("model")),
                        "generator_source": str(path),
                        "condition": normalize_text(row.get("target_age_label") or row.get("age_info_condition")),
                        "summary_id": int(summary.get("summary_id", index)),
                        "requested_target_words": row.get("target_word_count"),
                    }
                )
    return by_hash


def choose_provenance(
    candidates: list[dict[str, Any]],
    *,
    condition: str,
    summary_id: int,
) -> dict[str, Any]:
    if not candidates:
        return {}
    for candidate in candidates:
        if candidate["condition"] == condition and candidate["summary_id"] == summary_id:
            return candidate
    for candidate in candidates:
        if candidate["condition"] == condition:
            return candidate
    return candidates[0]


def load_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(path, low_memory=False)
    catalog["video_id"] = catalog["video_id"].fillna("").astype(str).str.strip()
    keep = ["video_id"]
    for column in [
        "best_Reading Age",
        "best_Reading Age Source",
        "best_Title",
        "best_Author",
    ]:
        if column in catalog.columns:
            keep.append(column)
    return catalog[keep].drop_duplicates("video_id")


def original_record(
    row: dict[str, Any],
    catalog_row: dict[str, Any],
    *,
    repair_fragments: bool = False,
) -> dict[str, Any] | None:
    story_key = normalize_text(row.get("story_key"))
    video_id = normalize_text(row.get("video_id"))
    raw_text = normalize_text(row.get("punctuated_story") or row.get("cleaned_story_verbatim"))
    segments = row.get("segments") or []
    if repair_fragments:
        segments = repair_title_fragments(segments)
    segmented_text = normalize_text(" ".join(normalize_text(segment.get("text")) for segment in segments))
    if not story_key or not raw_text or not segmented_text or not segments:
        return None
    raw_words = word_count(raw_text)
    segment_words = word_count(segmented_text)
    segment_coverage = min(raw_words, segment_words) / max(raw_words, segment_words, 1)
    age_range = normalize_text(catalog_row.get("best_Reading Age"))
    age_source = normalize_text(catalog_row.get("best_Reading Age Source")).lower()
    return {
        "record_id": f"{story_key}::original",
        "record_type": "original",
        "story_key": story_key,
        "video_id": video_id,
        "video_title": normalize_text(row.get("video_title")),
        "condition": "original",
        "condition_eligible": True,
        "condition_skip_reason": "",
        "summary_id": 0,
        "source_age_range": age_range,
        "source_age": age_midpoint(age_range),
        "target_age_range": age_range,
        "target_age": age_midpoint(age_range),
        "age_source": age_source,
        "book_title": normalize_text(catalog_row.get("best_Title")),
        "book_author": normalize_text(catalog_row.get("best_Author")),
        "compression_condition": "original",
        "generator_model": "",
        "generator_source": "",
        "requested_target_words": None,
        "word_count": segment_words,
        "raw_word_count": raw_words,
        "segment_word_coverage": segment_coverage,
        "compression_ratio": 1.0,
        "text": segmented_text,
        "segments": segments,
        "source_path": normalize_text(row.get("cleaned_path") or row.get("source_file")),
    }


def summary_record(
    row: dict[str, Any],
    original: dict[str, Any],
    provenance: dict[str, Any],
    condition_spec: AgeCondition,
    *,
    repair_fragments: bool = False,
) -> dict[str, Any] | None:
    raw_text = normalize_text(row.get("summary_text"))
    segments = row.get("segments") or []
    if repair_fragments:
        segments = repair_title_fragments(segments)
    segmented_text = normalize_text(" ".join(normalize_text(segment.get("text")) for segment in segments))
    if not raw_text or not segmented_text or not segments:
        return None
    condition = normalize_text(row.get("age_info_condition") or row.get("target_age_label"))
    summary_id = int(row.get("summary_id", 1))
    words = word_count(segmented_text)
    raw_words = word_count(raw_text)
    original_words = max(1, int(original["word_count"]))
    return {
        "record_id": f"{original['story_key']}::{condition}::{summary_id}",
        "record_type": "summary",
        "story_key": original["story_key"],
        "video_id": original["video_id"],
        "video_title": normalize_text(row.get("video_title")) or original["video_title"],
        "condition": condition,
        "condition_eligible": condition_spec.eligible,
        "condition_skip_reason": condition_spec.skip_reason,
        "summary_id": summary_id,
        "source_age_range": original["source_age_range"],
        "source_age": row.get("source_age") or original["source_age"],
        "target_age_range": condition_spec.target_spec,
        "target_age": condition_spec.target_age,
        "age_source": original["age_source"],
        "book_title": original["book_title"],
        "book_author": original["book_author"],
        "compression_condition": normalize_text(row.get("compression_condition")),
        "generator_model": provenance.get("generator_model", ""),
        "generator_source": provenance.get("generator_source", ""),
        "segmentation_method": normalize_text(row.get("segmentation_method")),
        "requested_target_words": provenance.get("requested_target_words"),
        "word_count": words,
        "raw_word_count": raw_words,
        "segment_word_coverage": min(words, raw_words) / max(words, raw_words, 1),
        "compression_ratio": words / original_words,
        "text": segmented_text,
        "segments": segments,
        "source_path": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical mechanistic-interpretability cohort.")
    parser.add_argument("--config", type=Path, default=Path("shared/config/default_paths.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/features/summaries/dataset_index"))
    parser.add_argument("--max-summaries", type=int, default=3)
    parser.add_argument("--age-source", default="customer")
    parser.add_argument("--source-age-min", type=float, default=2.0)
    parser.add_argument("--source-age-max", type=float, default=10.0)
    parser.add_argument("--minimum-age", type=int, default=2)
    parser.add_argument("--maximum-age", type=int, default=18)
    parser.add_argument("--minimum-segment-word-coverage", type=float, default=0.95)
    parser.add_argument(
        "--repair-title-fragments",
        action="store_true",
        help="Merge isolated honorific segments such as 'Mr.' into the following detail.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    catalog = load_catalog(Path(config["catalog_csv"]))
    catalog_map = catalog.set_index("video_id").to_dict("index")
    provenance = generation_provenance([Path(path) for path in config["generation_jsonl"]])

    originals: dict[str, dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []
    for path_text in config["original_detail_jsonl"]:
        path = Path(path_text)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            video_id = normalize_text(row.get("video_id"))
            record = original_record(
                row,
                catalog_map.get(video_id, {}),
                repair_fragments=args.repair_title_fragments,
            )
            if record is None:
                continue
            if float(record["segment_word_coverage"]) < args.minimum_segment_word_coverage:
                exclusions.append({"story_key": record["story_key"], "condition": "original", "summary_id": 0, "reason": "segment_word_coverage_below_threshold"})
                continue
            if args.age_source and record["age_source"] != args.age_source.lower():
                continue
            source_age = record.get("source_age")
            if source_age is None or not args.source_age_min <= float(source_age) <= args.source_age_max:
                continue
            originals.setdefault(record["story_key"], record)

    condition_specs = {
        story_key: {
            condition.name: condition
            for condition in relative_conditions(
                original["source_age_range"],
                minimum_age=args.minimum_age,
                maximum_age=args.maximum_age,
            )
        }
        for story_key, original in originals.items()
    }
    summaries: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path_text in config["summary_detail_jsonl"]:
        path = Path(path_text)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            story_key = normalize_text(row.get("story_key"))
            original = originals.get(story_key)
            if original is None:
                continue
            condition = normalize_text(row.get("age_info_condition") or row.get("target_age_label"))
            summary_id = int(row.get("summary_id", 1))
            if summary_id > args.max_summaries:
                exclusions.append({"story_key": story_key, "condition": condition, "summary_id": summary_id, "reason": "summary_id_above_maximum"})
                continue
            condition_spec = condition_specs[story_key].get(condition)
            if condition_spec is None:
                exclusions.append({"story_key": story_key, "condition": condition, "summary_id": summary_id, "reason": "unknown_condition"})
                continue
            if not condition_spec.eligible:
                exclusions.append({"story_key": story_key, "condition": condition, "summary_id": summary_id, "reason": condition_spec.skip_reason or "ineligible_condition"})
                continue
            text = normalize_text(row.get("summary_text"))
            source = choose_provenance(provenance.get(text_hash(text), []), condition=condition, summary_id=summary_id)
            record = summary_record(
                row,
                original,
                source,
                condition_spec,
                repair_fragments=args.repair_title_fragments,
            )
            if record is None:
                continue
            if float(record["segment_word_coverage"]) < args.minimum_segment_word_coverage:
                exclusions.append({"story_key": story_key, "condition": condition, "summary_id": summary_id, "reason": "segment_word_coverage_below_threshold"})
                continue
            summaries.setdefault((story_key, condition, summary_id), record)

    records = list(originals.values()) + list(summaries.values())
    records.sort(key=lambda row: (row["story_key"], row["record_type"] != "original", row["condition"], row["summary_id"]))

    condition_rows: list[dict[str, Any]] = []
    available = {(story, condition) for story, condition, _ in summaries}
    for story_key, original in originals.items():
        for condition in relative_conditions(
            original["source_age_range"],
            minimum_age=args.minimum_age,
            maximum_age=args.maximum_age,
        ):
            condition_rows.append(
                {
                    "story_key": story_key,
                    "video_id": original["video_id"],
                    "source_age_range": original["source_age_range"],
                    "source_age": original["source_age"],
                    "condition": condition.name,
                    "condition_eligible": condition.eligible,
                    "condition_skip_reason": condition.skip_reason,
                    "target_age_range": condition.target_spec,
                    "target_age": condition.target_age,
                    "has_segmented_summary": (story_key, condition.name) in available,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = args.output_dir / "cohort.jsonl"
    write_jsonl(cohort_path, records)
    write_jsonl(args.output_dir / "originals.jsonl", originals.values())
    write_jsonl(args.output_dir / "summaries.jsonl", summaries.values())
    metadata = pd.DataFrame([{key: value for key, value in row.items() if key not in {"text", "segments"}} for row in records])
    write_csv(args.output_dir / "cohort_metadata.csv", metadata)
    conditions = pd.DataFrame(condition_rows)
    write_csv(args.output_dir / "condition_index.csv", conditions)
    write_csv(args.output_dir / "cohort_exclusions.csv", pd.DataFrame(exclusions, columns=["story_key", "condition", "summary_id", "reason"]))

    print(f"Wrote {cohort_path}")
    print(f"Original stories: {len(originals)}")
    print(f"Segmented summaries: {len(summaries)}")
    print(f"Excluded records: {len(exclusions)}")
    print("Condition availability:")
    print(conditions.groupby(["condition", "condition_eligible", "has_segmented_summary"]).size().to_string())


if __name__ == "__main__":
    main()
