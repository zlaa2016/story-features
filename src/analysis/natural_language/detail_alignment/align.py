from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from shared.detail_metrics import lexical_preservation
from shared.io import read_jsonl


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


class Embedder:
    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        values = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    length: int
    text: str


def build_spans(details: list[str], max_span: int) -> list[Span]:
    return [
        Span(start, start + length - 1, length, " ".join(details[start : start + length]))
        for length in range(1, min(max_span, len(details)) + 1)
        for start in range(len(details) - length + 1)
    ]


def off_diagonal_centrality(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) <= 1:
        return np.ones(len(embeddings), dtype=np.float32)
    similarities = embeddings @ embeddings.T
    return (similarities.sum(axis=1) - np.diag(similarities)) / (len(embeddings) - 1)


def normalized_position(index: float, size: int) -> float:
    return 0.0 if size <= 1 else float(index) / float(size - 1)


def condition_label(row: dict[str, Any]) -> str:
    return str(row.get("condition") or row.get("target_age_label") or row.get("age_info_condition") or "").strip()


def optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def metadata(row: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    source_age = row.get("source_age")
    if optional_float(source_age) is None:
        source_age = original.get("source_age")
    output = {
        "story_key": row["story_key"],
        "video_id": row.get("video_id") or original.get("video_id"),
        "video_title": row.get("video_title") or original.get("video_title"),
        "source_age": source_age,
        "target_age": row.get("target_age"),
        "target_age_range": row.get("target_age_range"),
        "condition": condition_label(row),
        "summary_id": row.get("summary_id", 1),
        "summary_variant": row.get("_summary_variant", 1),
        "summary_text_hash": row.get("_summary_text_hash", ""),
        "compression_ratio": row.get("compression_ratio"),
        "generator_model": row.get("generator_model", ""),
        "segmentation_method": row.get("segmentation_method", ""),
    }
    for key in ["control", "alpha", "intervention_layer", "direction_name", "model_key"]:
        if key in row:
            output[key] = row.get(key)
    return output


def unique_summary_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    variant_counts: dict[tuple[Any, ...], int] = {}
    for row in rows:
        text = str(row.get("summary_text") or " ".join(str(item.get("text", "")) for item in row.get("segments", []))).strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        intervention_identity = (
            str(row.get("control", "")),
            str(row.get("alpha", "")),
            str(row.get("intervention_layer", "")),
            str(row.get("direction_name", "")),
            str(row.get("model_key", "")),
        ) if "control" in row or "alpha" in row else ()
        base = (str(row.get("story_key", "")), condition_label(row), int(row.get("summary_id", 1)), *intervention_identity)
        key = (*base, digest)
        if key in seen:
            continue
        seen.add(key)
        variant_counts[base] = variant_counts.get(base, 0) + 1
        record = dict(row)
        record["_summary_variant"] = variant_counts[base]
        record["_summary_text_hash"] = digest
        unique.append(record)
    return unique


def candidate_indices(scores: np.ndarray, spans: list[Span], top_k: int) -> list[list[int]]:
    by_length: dict[int, list[int]] = {}
    for index, span in enumerate(spans):
        by_length.setdefault(span.length, []).append(index)
    output: list[list[int]] = []
    for row in scores:
        selected = set(np.argsort(row)[-min(top_k, len(row)) :].tolist())
        for indices in by_length.values():
            selected.add(max(indices, key=lambda idx: float(row[idx])))
        output.append(sorted(selected))
    return output


def monotonic_alignment(
    adjusted_scores: np.ndarray,
    spans: list[Span],
    *,
    top_k: int,
    overlap_penalty: float,
    reversal_penalty: float,
) -> list[int]:
    candidates = candidate_indices(adjusted_scores, spans, top_k)
    values: list[dict[int, float]] = []
    parents: list[dict[int, int | None]] = []
    for summary_index, current_candidates in enumerate(candidates):
        current_values: dict[int, float] = {}
        current_parents: dict[int, int | None] = {}
        for current in current_candidates:
            base = float(adjusted_scores[summary_index, current])
            if summary_index == 0:
                current_values[current] = base
                current_parents[current] = None
                continue
            best_value = -math.inf
            best_parent: int | None = None
            for previous, previous_value in values[-1].items():
                previous_span = spans[previous]
                current_span = spans[current]
                overlap = max(0, previous_span.end - current_span.start + 1)
                reversal = max(0, previous_span.start - current_span.start)
                value = previous_value + base - overlap_penalty * overlap - reversal_penalty * reversal
                if value > best_value:
                    best_value = value
                    best_parent = previous
            current_values[current] = best_value
            current_parents[current] = best_parent
        values.append(current_values)
        parents.append(current_parents)
    current = max(values[-1], key=values[-1].get)
    path = [current]
    for summary_index in range(len(values) - 1, 0, -1):
        parent = parents[summary_index][current]
        if parent is None:
            raise RuntimeError("Broken dynamic-programming alignment path")
        path.append(parent)
        current = parent
    return list(reversed(path))


def capacitated_transport(
    similarities: np.ndarray,
    *,
    null_similarity: float,
    source_capacity: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_source, n_summary = similarities.shape
    costs = np.concatenate(
        [-similarities.reshape(-1), np.full(n_summary, -null_similarity, dtype=float)]
    )
    n_variables = len(costs)
    equality = np.zeros((n_summary, n_variables), dtype=float)
    for summary_index in range(n_summary):
        for source_index in range(n_source):
            equality[summary_index, source_index * n_summary + summary_index] = 1.0
        equality[summary_index, n_source * n_summary + summary_index] = 1.0
    inequality = np.zeros((n_source, n_variables), dtype=float)
    for source_index in range(n_source):
        start = source_index * n_summary
        inequality[source_index, start : start + n_summary] = 1.0
    result = linprog(
        costs,
        A_ub=inequality,
        b_ub=np.full(n_source, source_capacity),
        A_eq=equality,
        b_eq=np.ones(n_summary),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Transport optimization failed: {result.message}")
    source_flow = result.x[: n_source * n_summary].reshape(n_source, n_summary)
    null_flow = result.x[n_source * n_summary :]
    return source_flow, null_flow


def inversion_rate(indices: list[int]) -> float:
    pairs = [(left, right) for pos, left in enumerate(indices) for right in indices[pos + 1 :]]
    return float(np.mean([left > right for left, right in pairs])) if pairs else 0.0


def process_pair(
    original: dict[str, Any],
    summary: dict[str, Any],
    embedder: Embedder,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    original_details = [str(item.get("text", "")).strip() for item in original.get("segments", [])]
    summary_details = [str(item.get("text", "")).strip() for item in summary.get("segments", [])]
    original_details = [text for text in original_details if text]
    summary_details = [text for text in summary_details if text]
    if not original_details or not summary_details:
        raise ValueError("Both records must contain nonempty detail segments")

    spans = build_spans(original_details, args.max_span)
    all_text = (
        original_details
        + summary_details
        + [span.text for span in spans]
        + [" ".join(original_details), " ".join(summary_details)]
    )
    encoded = embedder.encode(all_text)
    n_original = len(original_details)
    n_summary = len(summary_details)
    original_embeddings = encoded[:n_original]
    summary_embeddings = encoded[n_original : n_original + n_summary]
    span_embeddings = encoded[n_original + n_summary :]
    span_embeddings = span_embeddings[: len(spans)]
    whole_original_embedding = encoded[-2]
    whole_summary_embedding = encoded[-1]
    detail_similarity = original_embeddings @ summary_embeddings.T
    span_similarity = summary_embeddings @ span_embeddings.T
    length_penalties = np.asarray([args.length_penalty * math.log2(span.length) for span in spans])
    adjusted = span_similarity - length_penalties[None, :]
    chosen = monotonic_alignment(
        adjusted,
        spans,
        top_k=args.top_candidates,
        overlap_penalty=args.overlap_penalty,
        reversal_penalty=args.reversal_penalty,
    )
    nonconsecutive_indices: list[list[int]] = []
    nonconsecutive_texts: list[str] = []
    for summary_index, span_index in enumerate(chosen):
        count = spans[span_index].length
        indices = sorted(np.argsort(detail_similarity[:, summary_index])[-count:].tolist())
        nonconsecutive_indices.append(indices)
        nonconsecutive_texts.append(" ".join(original_details[index] for index in indices))
    nonconsecutive_embeddings = embedder.encode(nonconsecutive_texts)
    source_flow, null_flow = capacitated_transport(
        detail_similarity,
        null_similarity=args.null_similarity,
        source_capacity=args.source_capacity,
    )
    centrality = off_diagonal_centrality(original_embeddings)
    centrality_rank = pd.Series(centrality).rank(method="average", pct=True).to_numpy(float)
    summary_centrality = off_diagonal_centrality(summary_embeddings)
    summary_centrality_rank = pd.Series(summary_centrality).rank(method="average", pct=True).to_numpy(float)
    meta = metadata(summary, original)

    summary_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    covered = np.zeros(n_original, dtype=bool)
    selected_starts: list[int] = []
    selected_ends: list[int] = []
    for summary_index, span_index in enumerate(chosen):
        span = spans[span_index]
        raw_best_span_index = int(np.argmax(span_similarity[summary_index]))
        raw_best_span = spans[raw_best_span_index]
        single_best_cosine = float(np.max(span_similarity[summary_index, :n_original]))
        nonconsecutive_cosine = float(summary_embeddings[summary_index] @ nonconsecutive_embeddings[summary_index])
        covered[span.start : span.end + 1] = True
        selected_starts.append(span.start)
        selected_ends.append(span.end)
        atomic_source = int(np.argmax(source_flow[:, summary_index]))
        atomic_mass = float(source_flow[atomic_source, summary_index])
        is_null = float(null_flow[summary_index]) >= atomic_mass
        for length in range(1, min(args.max_span, n_original) + 1):
            valid = [idx for idx, candidate in enumerate(spans) if candidate.length == length]
            best = max(valid, key=lambda idx: float(span_similarity[summary_index, idx]))
            best_span = spans[best]
            window_rows.append(
                {
                    **meta,
                    "summary_index": summary_index,
                    "window_length": length,
                    "best_start": best_span.start,
                    "best_end": best_span.end,
                    "best_cosine": float(span_similarity[summary_index, best]),
                    "penalized_score": float(adjusted[summary_index, best]),
                }
            )
        summary_rows.append(
            {
                **meta,
                "summary_index": summary_index,
                "summary_text": summary_details[summary_index],
                "source_start": span.start,
                "source_end": span.end,
                "source_span_length": span.length,
                "summarizing_index": span.length,
                "raw_best_span_length": raw_best_span.length,
                "raw_summarizing_index": raw_best_span.length,
                "source_span_consecutive": True,
                "source_text": span.text,
                "alignment_cosine": float(span_similarity[summary_index, span_index]),
                "alignment_penalized_score": float(adjusted[summary_index, span_index]),
                "single_detail_cosine": single_best_cosine,
                "span_gain_over_single": float(span_similarity[summary_index, span_index]) - single_best_cosine,
                "nonconsecutive_source_indices": json.dumps(nonconsecutive_indices[summary_index]),
                "nonconsecutive_source_text": nonconsecutive_texts[summary_index],
                "nonconsecutive_cosine": nonconsecutive_cosine,
                "nonconsecutive_gain": nonconsecutive_cosine - float(span_similarity[summary_index, span_index]),
                "nonconsecutive_is_better": bool(nonconsecutive_cosine > float(span_similarity[summary_index, span_index])),
                "source_mean_centrality": float(np.mean(centrality[span.start : span.end + 1])),
                "source_mean_centrality_rank": float(np.mean(centrality_rank[span.start : span.end + 1])),
                "summary_centrality": float(summary_centrality[summary_index]),
                "summary_centrality_rank": float(summary_centrality_rank[summary_index]),
                "centrality_rank_shift": float(
                    summary_centrality_rank[summary_index]
                    - np.mean(centrality_rank[span.start : span.end + 1])
                ),
                "transport_source_index": -1 if is_null else atomic_source,
                "transport_source_text": "" if is_null else original_details[atomic_source],
                "transport_source_cosine": args.null_similarity if is_null else float(detail_similarity[atomic_source, summary_index]),
                "transport_null": bool(is_null),
                "transport_null_mass": float(null_flow[summary_index]),
                "atomic_grounded_at_0_30": bool(np.max(detail_similarity[:, summary_index]) >= 0.30),
                "atomic_grounded_at_0_40": bool(np.max(detail_similarity[:, summary_index]) >= 0.40),
                "atomic_grounded_at_0_50": bool(np.max(detail_similarity[:, summary_index]) >= 0.50),
                **lexical_preservation(span.text, summary_details[summary_index]),
            }
        )

    original_rows = []
    for original_index, text in enumerate(original_details):
        best_summary = int(np.argmax(detail_similarity[original_index]))
        transport_mass = float(source_flow[original_index].sum())
        original_rows.append(
            {
                **meta,
                "orig_idx": original_index,
                "orig_text": text,
                "orig_centrality": float(centrality[original_index]),
                "centrality_pct": float(centrality_rank[original_index]),
                "orig_position_norm": normalized_position(original_index, n_original),
                "best_summary_idx": best_summary,
                "best_cosine": float(detail_similarity[original_index, best_summary]),
                "best_summary_centrality": float(summary_centrality[best_summary]),
                "best_summary_centrality_rank": float(summary_centrality_rank[best_summary]),
                "best_match_centrality_rank_shift": float(
                    summary_centrality_rank[best_summary] - centrality_rank[original_index]
                ),
                "retained_structured": bool(covered[original_index]),
                "retained_cosine": bool(detail_similarity[original_index, best_summary] >= args.null_similarity),
                "transport_mass": transport_mass,
                "retained_transport": bool(transport_mass >= 0.5),
                **{
                    f"whole_summary_{key}": value
                    for key, value in lexical_preservation(text, " ".join(summary_details)).items()
                },
            }
        )

    overlap = sum(max(0, selected_ends[i - 1] - selected_starts[i] + 1) for i in range(1, len(selected_starts)))
    mean_alignment = float(np.mean([row["alignment_cosine"] for row in summary_rows]))
    central_rank_total = float(np.sum(centrality_rank))
    best_cosines = np.asarray([row["best_cosine"] for row in original_rows], dtype=float)
    retained_semantic = best_cosines[covered]
    omitted_semantic = best_cosines[~covered]
    centrality_semantic_correlation = (
        float(np.corrcoef(centrality_rank, best_cosines)[0, 1])
        if n_original > 1 and np.std(centrality_rank) > 0 and np.std(best_cosines) > 0
        else float("nan")
    )
    source_aligned_centrality_rank = np.asarray(
        [row["source_mean_centrality_rank"] for row in summary_rows], dtype=float
    )
    summary_aligned_centrality_rank = np.asarray(
        [row["summary_centrality_rank"] for row in summary_rows], dtype=float
    )
    centrality_rank_shift = summary_aligned_centrality_rank - source_aligned_centrality_rank
    centrality_rank_correlation = (
        float(np.corrcoef(source_aligned_centrality_rank, summary_aligned_centrality_rank)[0, 1])
        if n_summary > 1
        and np.std(source_aligned_centrality_rank) > 0
        and np.std(summary_aligned_centrality_rank) > 0
        else float("nan")
    )
    source_age = optional_float(meta["source_age"])
    target_age = optional_float(meta["target_age"])
    story_row = {
        **meta,
        "n_original_details": n_original,
        "n_summary_details": n_summary,
        "summary_coverage_rate": 1.0,
        "summary_grounded_rate": float(np.mean([not row["transport_null"] for row in summary_rows])),
        "whole_text_cosine": float(whole_original_embedding @ whole_summary_embedding),
        "mean_alignment_cosine": mean_alignment,
        "mean_span_gain_over_single": float(np.mean([row["span_gain_over_single"] for row in summary_rows])),
        "mean_source_span_length": float(np.mean([row["source_span_length"] for row in summary_rows])),
        "multi_detail_summary_rate": float(np.mean([row["source_span_length"] > 1 for row in summary_rows])),
        "nonconsecutive_fusion_rate": float(np.mean([row["nonconsecutive_is_better"] for row in summary_rows])),
        "source_detail_coverage_rate": float(covered.mean()),
        "source_centrality_coverage": float(np.sum(centrality_rank[covered]) / central_rank_total) if central_rank_total else float("nan"),
        "retained_minus_omitted_centrality": float(np.mean(centrality[covered]) - np.mean(centrality[~covered])) if covered.any() and (~covered).any() else float("nan"),
        "mean_semantic_match_retained": float(retained_semantic.mean()) if len(retained_semantic) else float("nan"),
        "mean_semantic_match_omitted": float(omitted_semantic.mean()) if len(omitted_semantic) else float("nan"),
        "retained_minus_omitted_semantic_match": float(retained_semantic.mean() - omitted_semantic.mean()) if len(retained_semantic) and len(omitted_semantic) else float("nan"),
        "centrality_weighted_semantic_preservation": float(np.dot(centrality_rank, best_cosines) / central_rank_total) if central_rank_total else float("nan"),
        "centrality_semantic_match_correlation": centrality_semantic_correlation,
        "mean_summary_centrality_rank": float(np.mean(summary_aligned_centrality_rank)),
        "mean_source_centrality_rank_of_summary": float(np.mean(source_aligned_centrality_rank)),
        "mean_centrality_rank_shift": float(np.mean(centrality_rank_shift)),
        "mean_absolute_centrality_rank_shift": float(np.mean(np.abs(centrality_rank_shift))),
        "source_summary_centrality_rank_correlation": centrality_rank_correlation,
        "order_inversion_rate": inversion_rate(selected_starts),
        "source_overlap_per_summary_detail": overlap / n_summary,
        "age_distance": abs(target_age - source_age) if target_age is not None and source_age is not None else float("nan"),
        "age_matched": meta["condition"] == "actual",
    }
    return original_rows, summary_rows, story_row, window_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Align every summary detail to contiguous source details.")
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/natural_language/detail_alignment"))
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-span", type=int, default=10)
    parser.add_argument("--top-candidates", type=int, default=32)
    parser.add_argument("--length-penalty", type=float, default=0.015)
    parser.add_argument("--overlap-penalty", type=float, default=0.03)
    parser.add_argument("--reversal-penalty", type=float, default=0.25)
    parser.add_argument("--null-similarity", type=float, default=0.35)
    parser.add_argument("--source-capacity", type=float, default=2.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    originals = {row["story_key"]: row for row in read_jsonl(args.originals)}
    summaries = unique_summary_records(
        row for row in read_jsonl(args.summaries) if row.get("story_key") in originals
    )
    stop = None if args.limit <= 0 else args.start + args.limit
    summaries = summaries[args.start:stop]
    embedder = Embedder(args.model, args.device, args.batch_size)
    original_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    story_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, summary in enumerate(summaries, start=1):
        try:
            outputs = process_pair(originals[summary["story_key"]], summary, embedder, args)
            original_rows.extend(outputs[0])
            summary_rows.extend(outputs[1])
            story_rows.append(outputs[2])
            window_rows.extend(outputs[3])
        except Exception as error:
            errors.append({"story_key": summary.get("story_key"), "condition": condition_label(summary), "summary_id": summary.get("summary_id", 1), "error": f"{type(error).__name__}: {error}"})
        if index % 25 == 0 or index == len(summaries):
            print(f"[{index}/{len(summaries)}] aligned={len(story_rows)} errors={len(errors)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    from shared.io import write_csv
    write_csv(args.output_dir / "original_detail_retention.csv", pd.DataFrame(original_rows))
    write_csv(args.output_dir / "summary_detail_alignment.csv", pd.DataFrame(summary_rows))
    write_csv(args.output_dir / "story_alignment_metrics.csv", pd.DataFrame(story_rows))
    write_csv(args.output_dir / "window_length_scores.csv", pd.DataFrame(window_rows))
    write_csv(args.output_dir / "errors.csv", pd.DataFrame(errors, columns=["story_key", "condition", "summary_id", "error"]))
    print(f"Wrote {args.output_dir} with {len(story_rows)} summary-complete alignments")


if __name__ == "__main__":
    main()
