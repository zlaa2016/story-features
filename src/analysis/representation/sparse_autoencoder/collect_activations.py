from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.io import read_jsonl, write_csv, write_npy
from shared.modeling import load_model


def detail_map(text: str, offsets: list[tuple[int, int]], segments: list[dict[str, Any]]) -> np.ndarray:
    result = np.full(len(offsets), -1, dtype=np.int32)
    cursor = 0
    lowered = text.casefold()
    for detail_index, segment in enumerate(segments):
        value = str(segment.get("text", "")).strip()
        start = text.find(value, cursor)
        if start < 0:
            start = lowered.find(value.casefold(), cursor)
        if start < 0:
            continue
        end = start + len(value)
        for token_index, (left, right) in enumerate(offsets):
            if right > start and left < end and right > left:
                result[token_index] = detail_index
        cursor = end
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect token-level residual activations for SAE training.")
    parser.add_argument("--cohort", type=Path, default=Path("results/features/summaries/dataset_index/cohort.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/representation/sparse_autoencoder/activations"))
    parser.add_argument("--model-key", default="qwen3_8b", choices=["qwen3_8b", "gpt_oss_20b", "qwen3_30b_a3b"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--tokens-per-record", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-id", required=True)
    args = parser.parse_args()

    import torch

    layer_dir = args.output_dir / f"layer_{args.layer:02d}"
    activation_path = layer_dir / f"activations_{args.shard_id}.npy"
    metadata_path = layer_dir / f"metadata_{args.shard_id}.csv"
    if activation_path.exists() and metadata_path.exists():
        print(f"Shard already complete: {activation_path}")
        return
    stop = None if args.limit <= 0 else args.start + args.limit
    records = list(islice(read_jsonl(args.cohort), args.start, stop))
    model, tokenizer, device, _spec = load_model(args.model_key, model_name=args.model_name, dtype="bfloat16")
    rng = np.random.default_rng(args.seed + args.start + args.layer)
    vectors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for record_number, record in enumerate(records, start=1):
        encoded_cpu = tokenizer(record["text"], return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=args.max_tokens)
        offsets = [tuple(map(int, pair)) for pair in encoded_cpu.pop("offset_mapping")[0].tolist()]
        encoded = {key: value.to(device) for key, value in encoded_cpu.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        if args.layer + 1 >= len(outputs.hidden_states):
            raise ValueError(f"Layer {args.layer} unavailable; hidden-state count={len(outputs.hidden_states)}")
        activation = outputs.hidden_states[args.layer + 1][0].float().cpu().numpy()
        details = detail_map(record["text"], offsets, record.get("segments") or [])
        candidates = np.flatnonzero(details >= 0)
        if len(candidates) > args.tokens_per_record:
            candidates = np.sort(rng.choice(candidates, size=args.tokens_per_record, replace=False))
        for token_index in candidates:
            vectors.append(activation[token_index].astype(np.float16))
            metadata.append({
                "activation_row": len(vectors) - 1, "record_id": record["record_id"], "story_key": record["story_key"],
                "record_type": record["record_type"], "condition": record.get("condition", ""), "summary_id": record.get("summary_id", 0),
                "source_age": record.get("source_age"), "target_age": record.get("target_age"), "detail_index": int(details[token_index]),
                "token_index": int(token_index), "token_id": int(encoded_cpu["input_ids"][0, token_index]),
                "token": tokenizer.convert_ids_to_tokens(int(encoded_cpu["input_ids"][0, token_index])),
            })
        print(f"[{record_number}/{len(records)}] {record['record_id']} sampled={len(candidates)}", flush=True)
        del outputs, activation, encoded
    if not vectors:
        raise RuntimeError("No token activations collected")
    layer_dir.mkdir(parents=True, exist_ok=True)
    write_npy(activation_path, np.stack(vectors))
    write_csv(metadata_path, pd.DataFrame(metadata))
    print(f"Wrote {activation_path}: {len(vectors)} tokens")


if __name__ == "__main__":
    main()
