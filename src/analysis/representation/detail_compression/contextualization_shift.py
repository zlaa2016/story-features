"""Contextual-versus-isolated shift for every detail, at every stored layer.

shift = 1 - cos(contextual mean-pooled vector, isolated mean-pooled vector) for the same detail.
Large values mean the detail is represented very differently inside its text than alone.
Computed for original details and for the details of each younger/actual/older summary,
for the stories in the shared subset, so it can be set against source age, target age and
the span-search compression.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.representation.detail_compression.span_search import CONDITIONS, load_details
from shared.io import write_csv

KEEP = ["story_key", "record_type", "condition", "summary_id", "source_age", "target_age", "detail_index"]


def shift_rows(record_id: str, context, isolated, layer_ids: np.ndarray) -> pd.DataFrame:
    ctx_meta, ctx_vectors = context
    iso_meta, iso_vectors = isolated
    ctx = ctx_vectors / np.maximum(np.linalg.norm(ctx_vectors, axis=-1, keepdims=True), 1e-12)
    iso = iso_vectors / np.maximum(np.linalg.norm(iso_vectors, axis=-1, keepdims=True), 1e-12)
    cosine = (ctx * iso).sum(axis=-1)                       # [n_details, n_layers]
    base = ctx_meta[[c for c in KEEP if c in ctx_meta]].copy()
    base["record_id"] = record_id
    base["context_tokens"] = ctx_meta["tokens"].to_numpy()
    base["isolated_tokens"] = iso_meta["tokens"].to_numpy()
    long = base.loc[base.index.repeat(len(layer_ids))].reset_index(drop=True)
    long["layer"] = np.tile(layer_ids, len(base))
    long["cosine"] = cosine.reshape(-1)
    long["shift"] = 1.0 - long["cosine"]
    return long


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--subset", type=Path, default=Path("results/features/summaries/dataset_index/flat_subset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/representation/detail_compression/contextualization_shift"))
    parser.add_argument("--layers", default="", help="Comma list of stored layer ids; default all stored layers.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    subset = pd.read_csv(args.subset)
    if args.limit > 0:
        subset = subset.head(args.limit)
    ids = set(subset["original_record_id"]) | {r for c in CONDITIONS for r in subset[f"{c}_record_id"]}
    layers = [int(v) for v in args.layers.split(",") if v.strip()] or None
    context, context_layers = load_details(args.context_dir, "contextual", ids, "mean", layers)
    isolated, isolated_layers = load_details(args.isolated_dir, "isolated", ids, "mean", layers)
    if not np.array_equal(context_layers, isolated_layers):
        raise SystemExit(f"Layer mismatch: contextual {context_layers.tolist()} vs isolated {isolated_layers.tolist()}")

    frames, audit = [], []
    for record_id in sorted(ids):
        if record_id not in context or record_id not in isolated:
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
