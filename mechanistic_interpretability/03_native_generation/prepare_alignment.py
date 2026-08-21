from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

from mechanistic_interpretability.shared.io import read_jsonl, write_jsonl


def sentence_segments(text: str) -> list[dict[str, object]]:
    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", str(text or "")) if piece.strip()]
    return [{"segment_id": index + 1, "text": piece} for index, piece in enumerate(pieces)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare native open-model summaries for structured detail alignment.")
    parser.add_argument("--cohort", type=Path, default=Path("mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl"))
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    originals = [row for row in read_jsonl(args.cohort) if row.get("record_type") == "original"]
    original_map = {row["story_key"]: row for row in originals}
    summaries = []
    seen = set()
    for path_text in sorted(glob.glob(str(args.generation_dir / "generations_*.jsonl"))):
        for row in read_jsonl(Path(path_text)):
            story_key = str(row.get("story_key", ""))
            condition = str(row.get("condition", ""))
            summary_id = int(row.get("summary_id", 1))
            text = str(row.get("summary_text", "")).strip()
            key = (story_key, condition, summary_id)
            if row.get("status") != "success" or story_key not in original_map or not text or key in seen:
                continue
            seen.add(key)
            original = original_map[story_key]
            summaries.append({
                "record_id": f"{story_key}::{condition}::{summary_id}",
                "record_type": "summary",
                "story_key": story_key,
                "video_id": original.get("video_id", ""),
                "condition": condition,
                "age_info_condition": condition,
                "summary_id": summary_id,
                "source_age": original.get("source_age"),
                "target_age": row.get("target_age"),
                "target_age_range": row.get("target_age_range", ""),
                "summary_text": text,
                "segments": sentence_segments(text),
                "compression_ratio": row.get("achieved_compression_ratio"),
                "generator_model": row.get("generator_model", ""),
                "segmentation_method": "local_sentence",
                "model_key": args.generation_dir.name,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "originals.jsonl", originals)
    write_jsonl(args.output_dir / "summaries.jsonl", summaries)
    print(f"Originals: {len(originals)}; native summaries: {len(summaries)}")


if __name__ == "__main__":
    main()
