from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mechanistic_interpretability.shared.io import parse_int_list, read_jsonl, write_csv, write_npz
from mechanistic_interpretability.shared.modeling import load_model


def locate_segment_spans(text: str, segments: list[dict[str, Any]]) -> list[tuple[int, int] | None]:
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    lowered = text.casefold()
    for segment in segments:
        value = str(segment.get("text", "")).strip()
        if not value:
            spans.append(None)
            continue
        start = text.find(value, cursor)
        if start < 0:
            start = lowered.find(value.casefold(), cursor)
        if start < 0:
            start = text.find(value)
        if start < 0:
            start = lowered.find(value.casefold())
        if start < 0:
            spans.append(None)
            continue
        end = start + len(value)
        spans.append((start, end))
        cursor = end
    return spans


def token_indices_for_span(offsets: list[tuple[int, int]], span: tuple[int, int] | None) -> list[int]:
    if span is None:
        return []
    start, end = span
    return [index for index, (left, right) in enumerate(offsets) if right > start and left < end and right > left]


def router_tensors(outputs: Any) -> list[Any]:
    value = getattr(outputs, "router_logits", None)
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        return [tensor for tensor in value if tensor is not None]
    return [value]


def aggregate_router(router: Any, token_indices: list[int]) -> tuple[float, int, float]:
    import torch

    if not token_indices:
        return float("nan"), -1, float("nan")
    tensor = router
    if tensor.ndim == 2:
        sequence = tensor
    elif tensor.ndim == 3:
        sequence = tensor[0]
    else:
        return float("nan"), -1, float("nan")
    valid = [index for index in token_indices if index < sequence.shape[0]]
    if not valid:
        return float("nan"), -1, float("nan")
    probabilities = torch.softmax(sequence[valid].float(), dim=-1).mean(dim=0)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    top_probability, top_expert = probabilities.max(dim=0)
    return float(entropy.cpu()), int(top_expert.cpu()), float(top_probability.cpu())


def resolve_hidden_layers(hidden_states: tuple[Any, ...], specification: str) -> list[int]:
    maximum = len(hidden_states) - 2
    if specification == "all":
        return list(range(maximum + 1))
    requested = parse_int_list(specification)
    resolved = []
    for layer in requested:
        value = maximum if layer == -1 else layer
        if value < 0 or value > maximum:
            raise ValueError(f"Layer {layer} is outside 0..{maximum}")
        if value not in resolved:
            resolved.append(value)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-detail hidden states and router summaries.")
    parser.add_argument("--cohort", type=Path, default=Path("mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("mechanistic_interpretability/outputs/02_representation_extraction"))
    parser.add_argument("--model-key", default="gpt_oss_20b", choices=["qwen3_8b", "gpt_oss_20b", "qwen3_30b_a3b"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layers", default="0,4,8,12,16,20,-1")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-id", default="000")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--record-types", default="original,summary")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--minimum-segment-coverage", type=float, default=0.95)
    args = parser.parse_args()

    import torch

    shard_path = args.output_dir / f"features_{args.shard_id}.npz"
    metadata_path = args.output_dir / f"metadata_{args.shard_id}.csv"
    if shard_path.exists() and metadata_path.exists() and not args.overwrite:
        print(f"Shard already complete: {shard_path}")
        return

    allowed_types = {item.strip() for item in args.record_types.split(",") if item.strip()}
    records = [row for row in read_jsonl(args.cohort) if row.get("record_type") in allowed_types]
    stop = None if args.limit <= 0 else args.start + args.limit
    records = records[args.start : stop]
    model, tokenizer, device, _spec = load_model(
        args.model_key,
        model_name=args.model_name,
        dtype=args.dtype,
        output_router_logits=True,
    )

    vectors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    selected_layers: list[int] | None = None
    for record_index, record in enumerate(records, start=1):
        text = str(record["text"])
        encoded_cpu = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=args.max_tokens,
        )
        offsets = [tuple(map(int, pair)) for pair in encoded_cpu.pop("offset_mapping")[0].tolist()]
        encoded = {key: value.to(device) for key, value in encoded_cpu.items()}
        forward_kwargs: dict[str, Any] = {
            "output_hidden_states": True,
            "use_cache": False,
        }
        if hasattr(model.config, "output_router_logits"):
            forward_kwargs["output_router_logits"] = True
        with torch.inference_mode():
            outputs = model(**encoded, **forward_kwargs)
        hidden_states = outputs.hidden_states
        if selected_layers is None:
            selected_layers = resolve_hidden_layers(hidden_states, args.layers)
        routers = router_tensors(outputs)
        spans = locate_segment_spans(text, record.get("segments") or [])
        extracted_details = 0
        for detail_index, (segment, span) in enumerate(zip(record.get("segments") or [], spans)):
            indices = token_indices_for_span(offsets, span)
            if not indices:
                continue
            extracted_details += 1
            pooled = []
            for layer in selected_layers:
                tensor = hidden_states[layer + 1][0, indices, :].float().mean(dim=0)
                pooled.append(tensor.cpu().numpy().astype(np.float16))
            router_entropy = []
            router_top_expert = []
            router_top_probability = []
            for router in routers:
                entropy, expert, probability = aggregate_router(router, indices)
                router_entropy.append(entropy)
                router_top_expert.append(expert)
                router_top_probability.append(probability)
            vectors.append(np.stack(pooled, axis=0))
            metadata.append(
                {
                    "feature_row": len(vectors) - 1,
                    "record_id": record["record_id"],
                    "record_type": record["record_type"],
                    "story_key": record["story_key"],
                    "video_id": record.get("video_id", ""),
                    "condition": record.get("condition", ""),
                    "summary_id": record.get("summary_id", 0),
                    "source_age": record.get("source_age"),
                    "target_age": record.get("target_age"),
                    "compression_ratio": record.get("compression_ratio"),
                    "detail_index": detail_index,
                    "detail_text": segment.get("text", ""),
                    "token_count": len(indices),
                    "router_entropy": json.dumps(router_entropy),
                    "router_top_expert": json.dumps(router_top_expert),
                    "router_top_probability": json.dumps(router_top_probability),
                }
            )
        expected_details = len(record.get("segments") or [])
        audit.append({
            "record_id": record["record_id"],
            "story_key": record["story_key"],
            "expected_details": expected_details,
            "located_details": sum(span is not None for span in spans),
            "extracted_details": extracted_details,
            "segment_coverage": extracted_details / max(expected_details, 1),
            "encoded_tokens": len(offsets),
            "max_tokens": args.max_tokens,
        })
        print(f"[{record_index}/{len(records)}] {record['record_id']} details={len(spans)}", flush=True)
        del outputs, hidden_states, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not vectors:
        raise RuntimeError("No detail vectors were extracted")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_frame = pd.DataFrame(audit)
    write_csv(args.output_dir / f"audit_{args.shard_id}.csv", audit_frame)
    low_coverage = audit_frame[audit_frame["segment_coverage"].lt(args.minimum_segment_coverage)]
    if len(low_coverage):
        raise RuntimeError(
            f"{len(low_coverage)} records fell below segment coverage {args.minimum_segment_coverage}; "
            f"inspect audit_{args.shard_id}.csv and increase context length or repair segmentation"
        )
    write_npz(
        shard_path,
        vectors=np.stack(vectors),
        layers=np.asarray(selected_layers, dtype=np.int16),
        model_key=np.asarray(args.model_key),
    )
    write_csv(metadata_path, pd.DataFrame(metadata))
    print(f"Wrote {shard_path}: {len(vectors)} detail vectors")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
