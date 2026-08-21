from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mechanistic_interpretability.shared.io import parse_float_list, read_jsonl, write_csv
from mechanistic_interpretability.shared.research import select_work_shard, write_run_provenance


class Embedder:
    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ), dtype=np.float32)


def nonempty_subsets(indices: list[int], maximum_order: int) -> list[tuple[int, ...]]:
    return [
        subset
        for order in range(1, min(maximum_order, len(indices)) + 1)
        for subset in itertools.combinations(sorted(indices), order)
    ]


def mobius_coefficients(values: dict[tuple[int, ...], float]) -> dict[tuple[int, ...], float]:
    coefficients: dict[tuple[int, ...], float] = {}
    for subset in sorted(values, key=lambda value: (len(value), value)):
        lower = sum(
            coefficient
            for key, coefficient in coefficients.items()
            if set(key).issubset(subset)
        )
        coefficients[subset] = values[subset] - lower
    return coefficients


def segment_texts(record: dict[str, Any]) -> list[str]:
    return [str(item.get("text", "")).strip() for item in record.get("segments") or [] if str(item.get("text", "")).strip()]


def process_record(original: dict[str, Any], summary: dict[str, Any], embedder: Embedder, args: argparse.Namespace) -> list[dict[str, Any]]:
    source = segment_texts(original)
    target = segment_texts(summary)
    if not source or not target:
        return []
    atomic = embedder.encode(source + target)
    source_embeddings = atomic[:len(source)]
    target_embeddings = atomic[len(source):]
    atomic_similarity = target_embeddings @ source_embeddings.T
    specifications = []
    subset_texts = []
    for target_index, scores in enumerate(atomic_similarity):
        candidate_count = min(args.candidate_details, len(source))
        candidates = np.argsort(scores)[-candidate_count:].astype(int).tolist()
        for subset in nonempty_subsets(candidates, args.maximum_order):
            specifications.append((target_index, subset))
            subset_texts.append(" ".join(source[index] for index in subset))
    subset_embeddings = embedder.encode(subset_texts) if subset_texts else np.empty((0, source_embeddings.shape[1]))
    similarities: dict[int, dict[tuple[int, ...], float]] = {index: {} for index in range(len(target))}
    for embedding, (target_index, subset) in zip(subset_embeddings, specifications):
        similarities[target_index][subset] = float(target_embeddings[target_index] @ embedding)

    rows = []
    for target_index, values in similarities.items():
        for null_similarity in args.null_similarities:
            centered = {subset: score - null_similarity for subset, score in values.items()}
            coefficients = mobius_coefficients(centered)
            for subset, coefficient in coefficients.items():
                rows.append({
                    "story_key": original["story_key"],
                    "video_id": original.get("video_id", ""),
                    "condition": summary.get("condition", ""),
                    "summary_id": summary.get("summary_id", 1),
                    "source_age": original.get("source_age"),
                    "target_age": summary.get("target_age"),
                    "summary_detail_index": target_index,
                    "summary_detail_text": target[target_index],
                    "source_indices": json.dumps(subset),
                    "source_text": " ".join(source[index] for index in subset),
                    "interaction_order": len(subset),
                    "set_similarity": values[subset],
                    "null_similarity": null_similarity,
                    "mobius_coefficient": coefficient,
                    "absolute_mobius": abs(coefficient),
                    "positive_synergy": max(coefficient, 0.0) if len(subset) > 1 else 0.0,
                    "negative_redundancy": max(-coefficient, 0.0) if len(subset) > 1 else 0.0,
                    "candidate_policy": f"top_{args.candidate_details}_atomic_cosine",
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose one-to-many detail integration on the subset lattice.")
    parser.add_argument("--cohort", type=Path, default=Path("mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("mechanistic_interpretability/outputs/15_mobius_detail_integration"))
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidate-details", type=int, default=3)
    parser.add_argument("--maximum-order", type=int, default=3)
    parser.add_argument("--null-similarities", type=parse_float_list, default=parse_float_list("0.20,0.25,0.30"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-id", default="000")
    args = parser.parse_args()

    records = list(read_jsonl(args.cohort))
    originals = {row["story_key"]: row for row in records if row.get("record_type") == "original"}
    summaries = sorted(
        [row for row in records if row.get("record_type") == "summary" and row.get("story_key") in originals],
        key=lambda row: (str(row["story_key"]), str(row.get("condition", "")), int(row.get("summary_id", 1))),
    )
    summaries = select_work_shard(summaries, start=args.start, limit=args.limit, shard_index=args.shard_index, num_shards=args.num_shards)
    embedder = Embedder(args.model, args.device, args.batch_size)
    rows = []
    errors = []
    for index, summary in enumerate(summaries, start=1):
        try:
            rows.extend(process_record(originals[summary["story_key"]], summary, embedder, args))
        except Exception as error:
            errors.append({"record_id": summary.get("record_id", ""), "error": f"{type(error).__name__}: {error}"})
        print(f"[{index}/{len(summaries)}] {summary.get('record_id')} interactions={len(rows)} errors={len(errors)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"mobius_interactions_{args.shard_id}.csv", pd.DataFrame(rows))
    write_csv(args.output_dir / f"errors_{args.shard_id}.csv", pd.DataFrame(errors, columns=["record_id", "error"]))
    write_run_provenance(args.output_dir, args, extra={"summary_records": len(summaries), "interaction_rows": len(rows), "errors": len(errors)})


if __name__ == "__main__":
    main()
