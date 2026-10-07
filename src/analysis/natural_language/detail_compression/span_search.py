"""Natural-language span search: match each summary detail to a consecutive source span.

Details and candidate spans are embedded as text (sentence-transformer). For every story
in the subset, all three summaries (younger / actual / older) are searched and a per
summary-detail compression ratio is written. A null search against a different story's
source shows how much of the match is better than chance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.natural_language.detail_alignment.align import Embedder, build_spans
from shared.io import read_jsonl, word_count, write_csv
from shared.span_search import ratio_rows, search_story, span_list, summary_row

CONDITIONS = ("younger", "actual", "older")


def details_of(record: dict) -> list[str]:
    return [text for text in (str(item.get("text", "")).strip() for item in record.get("segments", [])) if text]


def embed_spans(embedder: Embedder, source_details: list[str], max_span: int) -> tuple[np.ndarray, list]:
    """Embed every candidate span of a source once; reused for all three summaries."""
    spans = build_spans(source_details, max_span)
    return embedder.encode([span.text for span in spans]), spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("results/features/summaries/dataset_index/cohort.jsonl"))
    parser.add_argument("--subset", type=Path, default=Path("results/features/summaries/dataset_index/flat_subset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/natural_language/detail_compression/span_search"))
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-span", type=int, default=10)
    parser.add_argument("--top-candidates", type=int, default=32)
    parser.add_argument("--length-penalties", default="0,0.1,0.2", help="Comma list, in score-spread units; the search is repeated for each.")
    parser.add_argument("--overlap-penalty", type=float, default=0.3)
    parser.add_argument("--reversal-penalty", type=float, default=2.5)
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N subset stories (debug).")
    parser.add_argument("--no-null", action="store_true")
    args = parser.parse_args()

    subset = pd.read_csv(args.subset)
    if args.limit > 0:
        subset = subset.head(args.limit)
    wanted = set(subset["original_record_id"]) | {r for c in CONDITIONS for r in subset[f"{c}_record_id"]}
    records = {r["record_id"]: r for r in read_jsonl(args.cohort) if r["record_id"] in wanted}
    penalties = [float(v) for v in args.length_penalties.split(",")]
    embedder = Embedder(args.model, args.device, args.batch_size)

    span_cache: dict[str, tuple[np.ndarray, list]] = {}

    def cached_spans(record_id: str, details: list[str]) -> tuple[np.ndarray, list]:
        # each story's source is also the next story's decoy, so embed it only once
        if record_id not in span_cache:
            span_cache[record_id] = embed_spans(embedder, details, args.max_span)
        return span_cache[record_id]

    story_keys = subset["story_key"].tolist()
    detail_frames, summary_rows = [], []
    for position, row in enumerate(subset.itertuples(index=False)):
        original = records.get(row.original_record_id)
        decoy_row = subset.iloc[(position + 1) % len(subset)]
        decoy = records.get(decoy_row["original_record_id"]) if not args.no_null and len(subset) > 1 else None
        if original is None:
            continue
        source = details_of(original)
        span_vectors, spans = cached_spans(original["record_id"], source)
        if decoy is not None:
            decoy_span_vectors, decoy_spans = cached_spans(decoy["record_id"], details_of(decoy))
        source_words = np.asarray([word_count(t) for t in source], dtype=float)
        for condition in CONDITIONS:
            summary = records.get(getattr(row, f"{condition}_record_id"))
            if summary is None:
                continue
            summary_details = details_of(summary)
            if not summary_details or not source:
                continue
            summary_vectors = embedder.encode(summary_details)
            sim = summary_vectors @ span_vectors.T
            if decoy is not None:
                null_sim = summary_vectors @ decoy_span_vectors.T
            summary_words = np.asarray([word_count(t) for t in summary_details], dtype=float)
            for penalty in penalties:
                chosen, scores = search_story(
                    sim, spans, length_penalty=penalty, overlap_penalty=args.overlap_penalty,
                    reversal_penalty=args.reversal_penalty, top_k=args.top_candidates,
                )
                null_scores = None
                if decoy is not None:
                    null_chosen, null_matrix = search_story(
                        null_sim, decoy_spans, length_penalty=penalty, overlap_penalty=args.overlap_penalty,
                        reversal_penalty=args.reversal_penalty, top_k=args.top_candidates,
                    )
                    null_scores = np.asarray([null_matrix[i, j] for i, j in enumerate(null_chosen)])
                details = ratio_rows(
                    chosen, spans, scores, source_words=source_words, summary_words=summary_words, null_scores=null_scores,
                )
                meta = {
                    "story_key": row.story_key, "condition": condition, "summary_id": summary.get("summary_id"),
                    "source_age": row.source_age, "target_age": summary.get("target_age"),
                    "record_id": summary["record_id"], "length_penalty": penalty,
                }
                details = pd.concat([pd.DataFrame([meta] * len(details)), details], axis=1)
                detail_frames.append(details)
                summary_rows.append({**meta, **summary_row(details, chosen, spans, len(source))})
        print(f"[{position + 1}/{len(story_keys)}] {row.story_key}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "details.csv", pd.concat(detail_frames, ignore_index=True))
    write_csv(args.output_dir / "summaries.csv", pd.DataFrame(summary_rows))
    (args.output_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
