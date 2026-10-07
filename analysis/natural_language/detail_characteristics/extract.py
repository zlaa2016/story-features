from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.detail_metrics import aggregate_norm, surface_characteristics, words
from shared.io import read_jsonl, write_csv
from shared.research import write_run_provenance


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_excel(path) if path.suffix.lower() in {".xls", ".xlsx"} else pd.read_csv(path, low_memory=False)


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {re.sub(r"[^a-z0-9]", "", str(column).lower()): str(column) for column in frame.columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    raise ValueError(f"None of {candidates} found in columns {frame.columns.tolist()}")


def load_norm(path: Path | None, value_candidates: list[str]) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    frame = read_table(path)
    word_column = find_column(frame, ["word", "term", "lemma"])
    value_column = find_column(frame, value_candidates)
    values = pd.to_numeric(frame[value_column], errors="coerce")
    return {
        str(word).casefold().strip(): float(value)
        for word, value in zip(frame[word_column], values)
        if str(word).strip() and pd.notna(value)
    }


def load_aoa(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    frame = read_table(path)
    word_column = find_column(frame, ["Word"])
    aoa_column = find_column(frame, ["AoA_Kup", "Rating.Mean", "AoA"])
    syllable_column = find_column(frame, ["Nsyll"])
    aoa = {}
    syllables = {}
    for word, age, syllable in zip(frame[word_column], frame[aoa_column], frame[syllable_column]):
        key = str(word).casefold().strip()
        if key and pd.notna(age):
            aoa[key] = float(age)
        if key and pd.notna(syllable):
            syllables[key] = float(syllable)
    return aoa, syllables


def detail_rows(cohort: Path) -> list[dict[str, Any]]:
    rows = []
    for record in read_jsonl(cohort):
        if record.get("record_type") != "original":
            continue
        segments = [item for item in record.get("segments") or [] if str(item.get("text", "")).strip()]
        for index, segment in enumerate(segments):
            rows.append({
                "story_key": record["story_key"],
                "video_id": record.get("video_id", ""),
                "source_age": record.get("source_age"),
                "source_age_range": record.get("source_age_range", ""),
                "detail_index": index,
                "detail_position": index / max(len(segments) - 1, 1),
                "detail_text": str(segment.get("text", "")).strip(),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure source-detail characteristics without generative annotation.")
    parser.add_argument("--cohort", type=Path, default=Path("results/features/summaries/dataset_index/cohort.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/natural_language/detail_characteristics"))
    parser.add_argument("--aoa", type=Path, default=Path("data/aoa/AoA_51715_words.xlsx"))
    parser.add_argument("--concreteness", type=Path, default=None)
    parser.add_argument("--vad", type=Path, default=None)
    parser.add_argument("--retention-csv", type=Path, default=None)
    args = parser.parse_args()

    aoa, syllables = load_aoa(args.aoa)
    concreteness = load_norm(args.concreteness, ["Conc.M", "concreteness", "mean"])
    valence = load_norm(args.vad, ["V.Mean.Sum", "valence"])
    arousal = load_norm(args.vad, ["A.Mean.Sum", "arousal"])
    dominance = load_norm(args.vad, ["D.Mean.Sum", "dominance"])
    output = []
    for row in detail_rows(args.cohort):
        token_list = words(row["detail_text"])
        output.append({
            **row,
            **surface_characteristics(row["detail_text"], syllables),
            **aggregate_norm(token_list, aoa, "aoa"),
            **aggregate_norm(token_list, concreteness, "concreteness"),
            **aggregate_norm(token_list, valence, "valence"),
            **aggregate_norm(token_list, arousal, "arousal"),
            **aggregate_norm(token_list, dominance, "dominance"),
        })
    frame = pd.DataFrame(output)
    if args.retention_csv and args.retention_csv.exists():
        retention = pd.read_csv(args.retention_csv, low_memory=False)
        retention = retention.rename(columns={"orig_idx": "detail_index"})
        columns = [column for column in ["story_key", "detail_index", "orig_centrality", "centrality_pct"] if column in retention]
        norms = retention[columns].drop_duplicates(["story_key", "detail_index"])
        frame = frame.merge(norms, on=["story_key", "detail_index"], how="left", validate="many_to_one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "detail_characteristics.csv", frame)
    coverage_columns = [column for column in frame if column.endswith("_coverage")]
    audit = pd.DataFrame([
        {"measure": column.removesuffix("_coverage"), "mean_coverage": float(frame[column].mean()), "n_details": len(frame)}
        for column in coverage_columns
    ])
    write_csv(args.output_dir / "measurement_coverage.csv", audit)
    write_run_provenance(args.output_dir, args, extra={"detail_rows": len(frame), "stories": int(frame["story_key"].nunique())})
    print(f"Wrote {len(frame)} details from {frame['story_key'].nunique()} stories")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
