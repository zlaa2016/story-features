"""Directed deletion-coupling matrices: how deleting detail i moves the state of every other detail j.

For one story with n details, one full pass gives each detail's state (mean over its tokens, and its last token) at the selected
layers.  Then each detail i is deleted in turn, the story is run again, and every remaining detail j is read at its own tokens.
Edge i -> j is how far detail j's state moved: ``cos`` = 1 - cosine(full, deleted), ``rel`` = ||full - deleted|| / ||full||.

A causal language model cannot change a state because of something that comes later, so entries with j < i must be ~0 and
the matrix is upper triangular up to numerical noise.  That block is saved too: it is the noise floor and a check of the setup.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.io import read_jsonl, write_csv, write_npz
from shared.modeling import MODEL_SPECS, load_model, resolve_layers

global_extraction = importlib.import_module(
    "features.representation_extraction.causal_lm.detail_extraction.extract_global"
)
GlobalRepresentationExtractor = global_extraction.GlobalRepresentationExtractor
delete_span = global_extraction.delete_span
locate_segment_spans = global_extraction.locate_segment_spans
model_context_limit = global_extraction.model_context_limit
resolve_requested_layers = global_extraction.resolve_requested_layers
token_indices_for_span = global_extraction.token_indices_for_span
tokenize_without_truncation = global_extraction.tokenize_without_truncation

EPSILON = 1e-12


def coupling_matrices(full: np.ndarray, deleted: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``full`` is (n, layers, dim); ``deleted[i]`` is (n - 1, layers, dim): the states of every detail except i, in order.

    Returns ``cos`` and ``rel`` of shape (layers, n, n); entry [l, i, j] is the change of detail j caused by deleting i
    (the diagonal stays 0).
    """
    n, layers, _ = full.shape
    cos = np.zeros((layers, n, n), dtype=np.float32)
    rel = np.zeros((layers, n, n), dtype=np.float32)
    for i, rows in deleted.items():
        others = [j for j in range(n) if j != i]
        if rows.shape[0] != len(others):
            raise ValueError(f"deletion {i}: expected {len(others)} remaining details, got {rows.shape[0]}")
        base = full[others].astype(np.float64)
        moved = rows.astype(np.float64)
        base_norm = np.linalg.norm(base, axis=-1)
        moved_norm = np.linalg.norm(moved, axis=-1)
        cosine = (base * moved).sum(axis=-1) / np.maximum(base_norm * moved_norm, EPSILON)
        cos[:, i, others] = (1.0 - cosine).T
        rel[:, i, others] = (np.linalg.norm(moved - base, axis=-1) / np.maximum(base_norm, EPSILON)).T
    return cos, rel


def story_records(cohorts: list[Path], story_keys: list[str], record_type: str = "original") -> list[dict[str, Any]]:
    """The first record of the wanted type for every wanted story, in the order of ``story_keys`` (cohorts earlier in the list win)."""
    wanted = set(story_keys)
    found: dict[str, dict[str, Any]] = {}
    for path in cohorts:
        if len(found) == len(wanted):
            break
        for record in read_jsonl(path):
            key = record.get("story_key")
            if key in wanted and key not in found and record.get("record_type") == record_type:
                found[key] = record
    return [found[key] for key in story_keys if key in found]


def ordered_story_keys(stories_csv: Path, column: str, max_stories: int) -> list[str]:
    """Story keys of the csv, most viewed first when it has view counts; at most ``max_stories`` (0 keeps all)."""
    frame = pd.read_csv(stories_csv, low_memory=False).drop_duplicates(column)
    if "view_count" in frame.columns:
        frame = frame.sort_values(["view_count", column], ascending=[False, True], kind="stable")
    keys = [str(key) for key in frame[column]]
    return keys[:max_stories] if max_stories > 0 else keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=Path, nargs="+", required=True, help="cohort jsonl files holding the stories' original records")
    parser.add_argument("--stories", type=Path, required=True, help="csv with the wanted stories (e.g. a batch's subset_stories.csv)")
    parser.add_argument("--story-column", default="story_key")
    parser.add_argument("--max-stories", type=int, default=0, help="keep only the N most viewed stories of the csv (0 = all)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", default="qwen3_8b", choices=sorted(MODEL_SPECS))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--layers", default="0,8,16,-1")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-id", default="000")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--min-details", type=int, default=4, help="stories with fewer details have no graph worth measuring")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result_path = args.output_dir / f"deletion_graph_{args.shard_id}.npz"
    metadata_path = args.output_dir / f"deletion_graph_metadata_{args.shard_id}.csv"
    if result_path.exists() and metadata_path.exists() and not args.overwrite:
        print(f"Shard already complete: {result_path}")
        return

    keys = ordered_story_keys(args.stories, args.story_column, args.max_stories)
    stop = None if args.limit <= 0 else args.start + args.limit
    keys = keys[args.start:stop]
    records = story_records(args.cohort, keys)
    print(f"{len(keys)} wanted stories in this shard, {len(records)} found in the cohorts", flush=True)

    model, tokenizer, device, spec = load_model(
        args.model_key, model_name=args.model_name, dtype=args.dtype, attn_implementation=args.attn_implementation
    )
    selected_layers = resolve_requested_layers(len(resolve_layers(model, spec)), args.layers)
    extractor = GlobalRepresentationExtractor(model, tokenizer, selected_layers)
    configured_limit = model_context_limit(model, tokenizer)
    effective_limit = min(args.max_tokens, configured_limit) if configured_limit else args.max_tokens

    arrays: dict[str, np.ndarray] = {}
    metadata: list[dict[str, Any]] = []
    found_keys = {record["story_key"] for record in records}
    audit = [{"story_key": key, "status": "not_in_cohorts"} for key in keys if key not in found_keys]
    try:
        for index, record in enumerate(records, start=1):
            text = str(record["text"])
            segments = record.get("segments") or []
            base = {"story_key": record["story_key"], "record_id": record["record_id"], "video_id": record.get("video_id", "")}
            if len(segments) < args.min_details:
                audit.append({**base, "status": "too_few_details", "n_details": len(segments)})
                continue
            encoded = tokenize_without_truncation(text, tokenizer)
            token_count = int(encoded["input_ids"].shape[1])
            if token_count > effective_limit:
                audit.append({**base, "status": "over_context", "token_count": token_count})
                continue
            offsets = [tuple(map(int, pair)) for pair in encoded.pop("offset_mapping")[0].tolist()]
            spans = locate_segment_spans(text, segments)
            detail_indices = [token_indices_for_span(offsets, span) for span in spans]
            if any(not indices for indices in detail_indices):
                audit.append({**base, "status": "unmapped_details", "token_count": token_count})
                continue
            _, _, full_last, full_mean = extractor.encode(encoded, device, detail_indices)

            deleted_last: dict[int, np.ndarray] = {}
            deleted_mean: dict[int, np.ndarray] = {}
            failed = None
            for i, span in enumerate(spans):
                remaining = [segment for j, segment in enumerate(segments) if j != i]
                deleted_text = delete_span(text, span)
                deleted_encoded = tokenize_without_truncation(deleted_text, tokenizer)
                deleted_offsets = [tuple(map(int, pair)) for pair in deleted_encoded.pop("offset_mapping")[0].tolist()]
                remaining_spans = locate_segment_spans(deleted_text, remaining)
                remaining_indices = [token_indices_for_span(deleted_offsets, s) for s in remaining_spans]
                if any(not indices for indices in remaining_indices):
                    failed = i
                    break
                _, _, rows_last, rows_mean = extractor.encode(deleted_encoded, device, remaining_indices)
                deleted_last[i], deleted_mean[i] = rows_last, rows_mean
            if failed is not None:
                audit.append({**base, "status": "unmapped_after_deletion", "deleted_detail": failed, "token_count": token_count})
                continue

            cos_mean, rel_mean = coupling_matrices(full_mean, deleted_mean)
            cos_last, rel_last = coupling_matrices(full_last, deleted_last)
            row = len(metadata)
            arrays[f"s{row:04d}_cos_mean"] = cos_mean
            arrays[f"s{row:04d}_rel_mean"] = rel_mean
            arrays[f"s{row:04d}_cos_last"] = cos_last
            arrays[f"s{row:04d}_rel_last"] = rel_last
            below = np.tril_indices(len(segments), -1)
            metadata.append({
                "row": row, **base, "n_details": len(segments), "token_count": token_count,
                "detail_tokens": ";".join(str(len(indices)) for indices in detail_indices),
                "causal_floor_cos_mean": float(np.abs(cos_mean[:, below[0], below[1]]).max()),
                "causal_floor_rel_mean": float(np.abs(rel_mean[:, below[0], below[1]]).max()),
            })
            audit.append({**base, "status": "encoded_complete", "n_details": len(segments), "token_count": token_count})
            print(f"[{index}/{len(records)}] {record['story_key']} details={len(segments)} tokens={token_count}", flush=True)
    finally:
        extractor.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"deletion_graph_audit_{args.shard_id}.csv", pd.DataFrame(audit))
    if not metadata:
        print(f"No stories finished for shard {args.shard_id}; nothing to write")
        return
    write_npz(
        result_path,
        layers=np.asarray(selected_layers, dtype=np.int16),
        model_key=np.asarray(args.model_key),
        **arrays,
    )
    write_csv(metadata_path, pd.DataFrame(metadata))
    print(f"Wrote {result_path}: {len(metadata)} stories")


if __name__ == "__main__":
    main()
