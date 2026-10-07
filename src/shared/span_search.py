"""Span search shared by the natural-language and representation compression analyses.

Each summary detail is matched to the best *consecutive* run of source details. The
same search (candidate pruning + monotonic dynamic programme) is used whatever the
similarity comes from, so differences between methods come from the similarity, not
the search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    length: int
    text: str = ""


def span_list(n_details: int, max_span: int) -> list[Span]:
    """All consecutive spans of 1..max_span details, ordered by length then start."""
    return [
        Span(start, start + length - 1, length)
        for length in range(1, min(max_span, n_details) + 1)
        for start in range(n_details - length + 1)
    ]


def pooled_span_vectors(vectors: np.ndarray, weights: np.ndarray, spans: list[Span]) -> np.ndarray:
    """Weighted mean of per-detail vectors over every span, via prefix sums.

    With token-count weights and mean-pooled detail vectors this equals the mean over
    all tokens in the span, so no re-encoding is needed.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    prefix_v = np.vstack([np.zeros((1, vectors.shape[1])), np.cumsum(vectors * weights, axis=0)])
    prefix_w = np.vstack([np.zeros((1, 1)), np.cumsum(weights, axis=0)])
    starts = np.asarray([span.start for span in spans])
    ends = np.asarray([span.end + 1 for span in spans])
    total_v = prefix_v[ends] - prefix_v[starts]
    total_w = np.maximum(prefix_w[ends] - prefix_w[starts], 1e-12)
    return (total_v / total_w).astype(np.float32)


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def standardize(scores: np.ndarray) -> np.ndarray:
    """Z-score a similarity matrix so penalties are expressed in score-spread units."""
    spread = float(scores.std())
    return (scores - float(scores.mean())) / max(spread, 1e-12)


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


def search_story(
    similarity: np.ndarray,
    spans: list[Span],
    *,
    length_penalty: float,
    overlap_penalty: float,
    reversal_penalty: float,
    top_k: int,
    standardize_scores: bool = True,
) -> tuple[list[int], np.ndarray]:
    """Best consecutive span per summary detail.

    similarity: [n_summary_details, n_spans]. Returns the chosen span index for each
    summary detail and the (possibly standardized) similarity used for scoring.
    """
    scores = standardize(similarity) if standardize_scores else similarity
    lengths = np.asarray([span.length for span in spans], dtype=np.float64)
    adjusted = scores - length_penalty * np.log2(lengths)[None, :]
    chosen = monotonic_alignment(
        adjusted,
        spans,
        top_k=top_k,
        overlap_penalty=overlap_penalty,
        reversal_penalty=reversal_penalty,
    )
    return chosen, scores


def prefix_sums(values: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype=np.float64))])


def inversion_rate(indices: list[int]) -> float:
    pairs = [(left, right) for pos, left in enumerate(indices) for right in indices[pos + 1 :]]
    return float(np.mean([left > right for left, right in pairs])) if pairs else 0.0


def ratio_rows(
    chosen: list[int],
    spans: list[Span],
    scores: np.ndarray,
    *,
    source_words: np.ndarray,
    summary_words: np.ndarray,
    source_tokens: np.ndarray | None = None,
    summary_tokens: np.ndarray | None = None,
    null_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    """One row per summary detail with compression ratios (detail counts primary)."""
    n_source = len(source_words)
    n_summary = len(summary_words)
    word_prefix = prefix_sums(source_words)
    token_prefix = prefix_sums(source_tokens) if source_tokens is not None else None
    rows: list[dict[str, Any]] = []
    for index, span_index in enumerate(chosen):
        span = spans[span_index]
        span_words = word_prefix[span.end + 1] - word_prefix[span.start]
        row: dict[str, Any] = {
            "summary_detail_index": index,
            "span_start": span.start,
            "span_end": span.end,
            "span_len": span.length,
            "span_share": span.length / n_source,
            "relative_compression": span.length * n_summary / n_source,
            "span_words": float(span_words),
            "summary_words": float(summary_words[index]),
            "word_ratio": float(span_words) / max(float(summary_words[index]), 1.0),
            "score": float(scores[index, span_index]),
            "best_unconstrained_span_len": spans[int(np.argmax(scores[index]))].length,
            "span_position": (span.start + span.end) / 2.0 / max(n_source - 1, 1),
        }
        row["log2_word_ratio"] = math.log2(max(row["word_ratio"], 1e-9))
        if token_prefix is not None and summary_tokens is not None:
            span_tokens = token_prefix[span.end + 1] - token_prefix[span.start]
            row["span_tokens"] = float(span_tokens)
            row["summary_tokens"] = float(summary_tokens[index])
            row["token_ratio"] = float(span_tokens) / max(float(summary_tokens[index]), 1.0)
        if null_scores is not None:
            row["null_score"] = float(null_scores[index])
        rows.append(row)
    return pd.DataFrame(rows)


def summary_row(details: pd.DataFrame, chosen: list[int], spans: list[Span], n_source: int) -> dict[str, Any]:
    covered = np.zeros(n_source, dtype=bool)
    for span_index in chosen:
        covered[spans[span_index].start : spans[span_index].end + 1] = True
    starts = [spans[i].start for i in chosen]
    row = {
        "n_source_details": n_source,
        "n_summary_details": len(chosen),
        "detail_compression": n_source / max(len(chosen), 1),
        "mean_span_len": float(details["span_len"].mean()),
        "median_span_len": float(details["span_len"].median()),
        "mean_span_share": float(details["span_share"].mean()),
        "mean_relative_compression": float(details["relative_compression"].mean()),
        "mean_log2_word_ratio": float(details["log2_word_ratio"].mean()),
        "covered_fraction": float(covered.mean()),
        "inversion_rate": inversion_rate(starts),
        "mean_score": float(details["score"].mean()),
    }
    if "null_score" in details:
        row["mean_null_score"] = float(details["null_score"].mean())
    if "token_ratio" in details:
        row["mean_log2_token_ratio"] = float(np.log2(np.maximum(details["token_ratio"], 1e-9)).mean())
    return row
