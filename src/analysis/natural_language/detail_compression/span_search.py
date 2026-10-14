"""Natural-language span search: match each summary detail to a consecutive source span.

Details and candidate spans are embedded as text (sentence-transformer). Every story in the cohort that has a
summary in a confirmatory condition (younger / actual / older) is searched, with every one of its summaries: no
story is chosen, balanced across age, or required to have all three conditions. A per summary-detail compression
ratio is written. A null search against a different story's source shows how much of the match is better than chance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.natural_language.detail_alignment.align import Embedder, build_spans
from shared.cohort_units import NOT_DETAIL_LEVEL, excluded_record_ids, story_units
from shared.io import read_jsonl, word_count, write_csv
from shared.span_search import ratio_rows, search_story, span_list, summary_row


def details_of(record: dict) -> list[str]:
    return [text for text in (str(item.get("text", "")).strip() for item in record.get("segments", [])) if text]


def norm_id(value) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def load_summary_segments(path: Path) -> list[dict]:
    """Summary records from another segmentation (for example the sentence-level file), in cohort-record form."""
    records = []
    for row in read_jsonl(path):
        condition = row.get("age_info_condition") or row.get("condition")
        summary_id = row.get("summary_id")
        records.append({
            "record_id": f"{row['story_key']}::{condition}::{norm_id(summary_id)}", "record_type": "summary",
            "story_key": str(row["story_key"]), "condition": condition, "summary_id": summary_id,
            "target_age": row.get("target_age"), "segments": row.get("segments", []),
        })
    return records


def embed_spans(embedder: Embedder, source_details: list[str], max_span: int) -> tuple[np.ndarray, list]:
    """Embed every candidate span of a source once; reused for all three summaries."""
    spans = build_spans(source_details, max_span)
    return embedder.encode([span.text for span in spans]), spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("results/features/summaries/dataset_index/cohort.jsonl"))
    parser.add_argument("--cohort-metadata", type=Path, default=Path("results/features/summaries/dataset_index/cohort_metadata.csv"))
    parser.add_argument("--exclude-methods", default=",".join(NOT_DETAIL_LEVEL),
                        help="Comma list of summary segmentation methods to skip (default: sentence-level, a separate measurement). '' keeps everything.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/natural_language/detail_compression/span_search"))
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-span", type=int, default=10)
    parser.add_argument("--top-candidates", type=int, default=32)
    parser.add_argument("--length-penalties", default="0", help="Comma list, in score-spread units; the search is repeated for each.")
    parser.add_argument("--overlap-penalty", type=float, default=0.3)
    parser.add_argument("--reversal-penalty", type=float, default=2.5)
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N stories (debug only; a real run uses all).")
    parser.add_argument("--no-null", action="store_true")
    parser.add_argument("--summary-segments", type=Path, default=None,
                        help="JSONL with another segmentation of the summaries (for example the sentence-level file). Its summaries and "
                             "segments are used instead of the cohort's, so the ratio becomes source details per summary sentence. "
                             "Originals still come from the cohort. No vectors are needed.")
    args = parser.parse_args()

    if args.summary_segments:
        # Summaries come from the other segmentation file; only the original texts come from the cohort. The
        # segmentation-method exclusion does not apply: sentence-level summaries are the point of this arm.
        originals = (r for r in read_jsonl(args.cohort) if r.get("record_type") == "original")
        exclude = set()
        units = story_units(list(originals) + load_summary_segments(args.summary_segments), exclude)
    else:
        exclude = excluded_record_ids(args.cohort_metadata, [m for m in args.exclude_methods.split(",") if m])
        units = story_units(read_jsonl(args.cohort), exclude)
    if args.limit > 0:
        units = units[: args.limit]
    print(f"{len(units)} stories, {sum(len(u['summaries']) for u in units)} summaries; {len(exclude)} excluded by segmentation method")
    penalties = [float(v) for v in args.length_penalties.split(",")]
    embedder = Embedder(args.model, args.device, args.batch_size)

    span_cache: dict[str, tuple[np.ndarray, list]] = {}

    def cached_spans(record_id: str, details: list[str]) -> tuple[np.ndarray, list]:
        # each story's source is also the next story's decoy, so embed it only once
        if record_id not in span_cache:
            span_cache[record_id] = embed_spans(embedder, details, args.max_span)
        return span_cache[record_id]

    detail_frames, summary_rows = [], []
    for position, unit in enumerate(units):
        original = unit["original"]
        decoy = units[(position + 1) % len(units)]["original"] if not args.no_null and len(units) > 1 else None
        source = details_of(original)
        span_vectors, spans = cached_spans(original["record_id"], source)
        if decoy is not None:
            decoy_span_vectors, decoy_spans = cached_spans(decoy["record_id"], details_of(decoy))
        source_words = np.asarray([word_count(t) for t in source], dtype=float)
        for summary in unit["summaries"]:
            condition = summary["condition"]
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
                    "story_key": unit["story_key"], "condition": condition, "summary_id": summary.get("summary_id"),
                    "source_age": unit["source_age"], "target_age": summary.get("target_age"),
                    "record_id": summary["record_id"], "length_penalty": penalty,
                }
                details = pd.concat([pd.DataFrame([meta] * len(details)), details], axis=1)
                detail_frames.append(details)
                summary_rows.append({**meta, **summary_row(details, chosen, spans, len(source))})
        print(f"[{position + 1}/{len(units)}] {unit['story_key']}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "details.csv", pd.concat(detail_frames, ignore_index=True))
    write_csv(args.output_dir / "summaries.csv", pd.DataFrame(summary_rows))
    (args.output_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
