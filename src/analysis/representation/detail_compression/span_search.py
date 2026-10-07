"""Representation span search with GPT-OSS detail vectors (contextual or isolated).

Same search as the natural-language version, but similarity is cosine between hidden
states instead of sentence embeddings.

  contextual: every detail is pooled from a forward pass over its whole text, so a span
              vector is the token-weighted mean of the stored detail vectors (exact for
              mean pooling).
  isolated:   every detail was encoded alone, so a span vector is the token-weighted
              average of the isolated detail vectors (an approximation; encoding the span
              jointly would need a new GPU extraction).

Vectors are mean-centred per layer before cosine, because raw residual states share large
common directions that would otherwise dominate every similarity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shared.io import word_count, write_csv
from shared.span_search import pooled_span_vectors, ratio_rows, search_story, span_list, summary_row, unit_rows

CONDITIONS = ("younger", "actual", "older")
KINDS = {
    # kind: (metadata glob, feature prefix, mean array, primary array, token-count column)
    "contextual": ("contextual_detail_metadata_*.csv", "contextual_detail_features_", "mean_control_vectors", "last_token_vectors", "detail_token_count"),
    "isolated": ("isolated_detail_metadata_*.csv", "isolated_detail_features_", "mean_vectors", "eos_vectors", "token_count"),
}


def load_details(directory: Path, kind: str, record_ids: set[str], pooling: str, layers: list[int] | None):
    """Return {record_id: (detail_frame, vectors[n_details, n_layers, D])} and the layer ids."""
    pattern, prefix, mean_name, primary_name, token_column = KINDS[kind]
    array_name = mean_name if pooling == "mean" else primary_name
    records: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    layer_ids: np.ndarray | None = None
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {pattern} under {directory}")
    for metadata_path in paths:
        suffix = metadata_path.stem.split("_metadata_", 1)[1]
        feature_path = directory / f"{prefix}{suffix}.npz"
        if not feature_path.exists():
            continue
        metadata = pd.read_csv(metadata_path, low_memory=False).reset_index(drop=True)
        rows = np.flatnonzero(metadata["record_id"].isin(record_ids).to_numpy())
        if not len(rows):
            continue
        with np.load(feature_path, allow_pickle=False) as payload:
            shard_layers = payload["layers"]
            keep = np.arange(len(shard_layers)) if layers is None else np.flatnonzero(np.isin(shard_layers, layers))
            vectors = payload[array_name][rows][:, keep].astype(np.float32)
            layer_ids = shard_layers[keep] if layer_ids is None else layer_ids
        selected = metadata.iloc[rows].reset_index(drop=True)
        selected["tokens"] = pd.to_numeric(selected[token_column], errors="coerce").fillna(1.0)
        for record_id, group in selected.groupby("record_id", sort=False):
            order = group.sort_values("detail_index")
            records[str(record_id)] = (order.reset_index(drop=True), vectors[order.index.to_numpy()])
    if layer_ids is None:
        raise FileNotFoundError(f"No vectors for the requested records under {directory}")
    return records, layer_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(KINDS), required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--subset", type=Path, default=Path("results/features/summaries/dataset_index/flat_subset.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pooling", choices=["mean", "primary"], default="mean", help="mean = token-weighted mean vectors; primary = last-token (contextual) / EOS (isolated) vectors, averaged over the span.")
    parser.add_argument("--layers", default="", help="Comma list of stored layer ids; default all stored layers.")
    parser.add_argument("--max-span", type=int, default=10)
    parser.add_argument("--top-candidates", type=int, default=32)
    parser.add_argument("--length-penalties", default="0,0.1,0.2")
    parser.add_argument("--overlap-penalty", type=float, default=0.3)
    parser.add_argument("--reversal-penalty", type=float, default=2.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-null", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or Path(f"results/analysis/representation/detail_compression/span_search/{args.kind}_{args.pooling}")

    subset = pd.read_csv(args.subset)
    if args.limit > 0:
        subset = subset.head(args.limit)
    ids = {r for c in CONDITIONS for r in subset[f"{c}_record_id"]} | set(subset["original_record_id"])
    layers = [int(v) for v in args.layers.split(",") if v.strip()] or None
    records, layer_ids = load_details(args.vector_dir, args.kind, ids, args.pooling, layers)
    penalties = [float(v) for v in args.length_penalties.split(",")]

    # Per-layer mean over every loaded detail vector, used to centre before cosine.
    stacked = np.concatenate([vectors for _, vectors in records.values()], axis=0)
    layer_means = stacked.mean(axis=0)
    del stacked

    detail_frames, summary_rows, audit = [], [], []
    for position, row in enumerate(subset.itertuples(index=False)):
        original = records.get(row.original_record_id)
        decoy_row = subset.iloc[(position + 1) % len(subset)]
        decoy = records.get(decoy_row["original_record_id"]) if not args.no_null and len(subset) > 1 else None
        for condition in CONDITIONS:
            record_id = getattr(row, f"{condition}_record_id")
            summary = records.get(record_id)
            if original is None or summary is None:
                audit.append({"story_key": row.story_key, "condition": condition, "status": "missing_vectors"})
                continue
            source_meta, source_vectors = original
            summary_meta, summary_vectors = summary
            spans = span_list(len(source_meta), args.max_span)
            source_words = source_meta["detail_text"].map(word_count).to_numpy(float)
            summary_words = summary_meta["detail_text"].map(word_count).to_numpy(float)
            source_tokens = source_meta["tokens"].to_numpy(float)
            summary_tokens = summary_meta["tokens"].to_numpy(float)
            for layer_position, layer in enumerate(layer_ids):
                centre = layer_means[layer_position]
                span_vectors = unit_rows(pooled_span_vectors(source_vectors[:, layer_position] - centre, source_tokens, spans))
                detail_vectors = unit_rows(summary_vectors[:, layer_position] - centre)
                sim = detail_vectors @ span_vectors.T
                if decoy is not None:
                    decoy_meta, decoy_vectors = decoy
                    decoy_spans = span_list(len(decoy_meta), args.max_span)
                    decoy_span_vectors = unit_rows(pooled_span_vectors(decoy_vectors[:, layer_position] - centre, decoy_meta["tokens"].to_numpy(float), decoy_spans))
                    null_sim = detail_vectors @ decoy_span_vectors.T
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
                        chosen, spans, scores, source_words=source_words, summary_words=summary_words,
                        source_tokens=source_tokens, summary_tokens=summary_tokens, null_scores=null_scores,
                    )
                    meta = {
                        "story_key": row.story_key, "condition": condition, "record_id": record_id,
                        "source_age": row.source_age, "layer": int(layer), "length_penalty": penalty,
                        "kind": args.kind, "pooling": args.pooling,
                    }
                    details = pd.concat([pd.DataFrame([meta] * len(details)), details], axis=1)
                    detail_frames.append(details)
                    summary_rows.append({**meta, **summary_row(details, chosen, spans, len(source_meta))})
            audit.append({"story_key": row.story_key, "condition": condition, "status": "searched"})
        print(f"[{position + 1}/{len(subset)}] {row.story_key}", flush=True)

    if not detail_frames:
        raise SystemExit("No story had vectors for the original and a summary; check --vector-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "details.csv", pd.concat(detail_frames, ignore_index=True))
    write_csv(output_dir / "summaries.csv", pd.DataFrame(summary_rows))
    write_csv(output_dir / "audit.csv", pd.DataFrame(audit))
    (output_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
