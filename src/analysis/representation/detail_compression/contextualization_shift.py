"""Contextual-versus-isolated shift for every detail, at every stored layer.

shift = 1 - cos(contextual mean-pooled vector, isolated mean-pooled vector) for the same detail.
centrality = mean cosine between a detail's contextual vector and the other details' contextual
vectors in the same text (the definition used in the earlier contextualization notebook).
Large values mean the detail is represented very differently inside its text than alone.
Computed for original details and for the details of each younger/actual/older summary,
for every record that has both vectors (no story selection), so it can be set against source age, target age and
the span-search compression.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.representation.detail_compression.span_search import load_details
from shared.cohort_units import CONDITIONS, NOT_DETAIL_LEVEL, excluded_record_ids
from shared.io import write_csv

KEEP = ["story_key", "record_type", "condition", "summary_id", "source_age", "target_age", "detail_index"]


def shift_rows(record_id: str, context, isolated, layer_ids: np.ndarray) -> pd.DataFrame:
    ctx_meta, ctx_vectors = context
    iso_meta, iso_vectors = isolated
    ctx_vectors, iso_vectors = ctx_vectors.astype(np.float32), iso_vectors.astype(np.float32)
    ctx = ctx_vectors / np.maximum(np.linalg.norm(ctx_vectors, axis=-1, keepdims=True), 1e-12)
    iso = iso_vectors / np.maximum(np.linalg.norm(iso_vectors, axis=-1, keepdims=True), 1e-12)
    cosine = (ctx * iso).sum(axis=-1)                       # [n_details, n_layers]
    n = len(ctx)
    if n > 1:
        similarity = np.einsum("nld,mld->lnm", ctx, ctx)     # [n_layers, n, n]
        diagonal = np.einsum("lnn->ln", similarity)
        centrality = ((similarity.sum(axis=2) - diagonal) / (n - 1)).T   # [n_details, n_layers]
    else:
        centrality = np.full(cosine.shape, np.nan)
    base = ctx_meta[[c for c in KEEP if c in ctx_meta]].copy()
    base["record_id"] = record_id
    base["context_tokens"] = ctx_meta["tokens"].to_numpy()
    base["isolated_tokens"] = iso_meta["tokens"].to_numpy()
    long = base.loc[base.index.repeat(len(layer_ids))].reset_index(drop=True)
    long["layer"] = np.tile(layer_ids, len(base))
    long["cosine"] = cosine.reshape(-1)
    long["shift"] = 1.0 - long["cosine"]
    long["centrality"] = centrality.reshape(-1)
    return long


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--cohort-metadata", type=Path, default=Path("results/features/summaries/dataset_index/cohort_metadata.csv"))
    parser.add_argument("--exclude-methods", default=",".join(NOT_DETAIL_LEVEL),
                        help="Comma list of summary segmentation methods to skip (default: sentence-level, which needs no vectors). '' keeps everything.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/representation/detail_compression/contextualization_shift"))
    parser.add_argument("--layers", default="", help="Comma list of stored layer ids; default all stored layers.")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N stories (debug only; a real run uses all).")
    args = parser.parse_args()

    layers = [int(v) for v in args.layers.split(",") if v.strip()] or None
    context, context_layers = load_details(args.context_dir, "contextual", None, "mean", layers)
    isolated, isolated_layers = load_details(args.isolated_dir, "isolated", None, "mean", layers)
    if not np.array_equal(context_layers, isolated_layers):
        raise SystemExit(f"Layer mismatch: contextual {context_layers.tolist()} vs isolated {isolated_layers.tolist()}")

    skip = excluded_record_ids(args.cohort_metadata, [m for m in args.exclude_methods.split(",") if m])

    def wanted(frame: pd.DataFrame) -> bool:
        first = frame.iloc[0]
        return first["record_type"] == "original" or first.get("condition") in CONDITIONS

    ids = sorted(rid for rid, (frame, _) in context.items() if rid not in skip and wanted(frame))
    if args.limit > 0:
        keep = sorted({str(context[rid][0].iloc[0]["story_key"]) for rid in ids})[: args.limit]
        ids = [rid for rid in ids if str(context[rid][0].iloc[0]["story_key"]) in set(keep)]
    print(f"{len(ids)} records")
    frames, audit = [], []
    for record_id in ids:
        if record_id not in isolated:
            audit.append({"record_id": record_id, "status": "missing_vectors"})
            continue
        ctx_meta, iso_meta = context[record_id][0], isolated[record_id][0]
        if len(ctx_meta) != len(iso_meta) or not np.array_equal(ctx_meta["detail_index"].to_numpy(), iso_meta["detail_index"].to_numpy()):
            audit.append({"record_id": record_id, "status": "detail_mismatch"})
            continue
        frames.append(shift_rows(record_id, context[record_id], isolated[record_id], context_layers))
        audit.append({"record_id": record_id, "status": "ok", "details": len(ctx_meta)})
    if not frames:
        raise SystemExit("No record had both contextual and isolated vectors")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "details.csv", pd.concat(frames, ignore_index=True))
    write_csv(args.output_dir / "audit.csv", pd.DataFrame(audit))
    print(pd.DataFrame(audit).status.value_counts().to_string())
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
