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
    parser = argparse.ArgumentParser(description="Convert activation-steered generations to detail-alignment records.")
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
            text = str(row.get("summary_text", "")).strip()
            key = (story_key, str(row.get("control", "")), float(row.get("alpha", 0)))
            if story_key not in original_map or not text or key in seen:
                continue
            seen.add(key)
            original = original_map[story_key]
            summaries.append({
                "record_id": f"{story_key}::intervention::{row.get('control')}::{row.get('alpha')}",
                "record_type": "summary",
                "story_key": story_key,
                "video_id": original.get("video_id", ""),
                "condition": "intervention",
                "age_info_condition": "intervention",
                "summary_id": 1,
                "source_age": original.get("source_age"),
                "target_age": original.get("source_age"),
                "target_age_range": original.get("source_age_range", ""),
                "summary_text": text,
                "segments": sentence_segments(text),
                "compression_ratio": row.get("achieved_compression_ratio"),
                "generator_model": row.get("model_key", ""),
                "segmentation_method": "local_sentence",
                "control": row.get("control", ""),
                "alpha": row.get("alpha"),
                "intervention_layer": row.get("layer"),
                "direction_name": row.get("direction_name", ""),
                "model_key": row.get("model_key", ""),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "originals.jsonl", originals)
    write_jsonl(args.output_dir / "summaries.jsonl", summaries)
    print(f"Originals: {len(originals)}; intervention summaries: {len(summaries)}")


if __name__ == "__main__":
    main()
