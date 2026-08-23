from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mechanistic_interpretability.shared.io import parse_int_list, read_jsonl, write_csv, write_npz
from mechanistic_interpretability.shared.modeling import MODEL_SPECS, layer_output_tensor, load_model, resolve_layers


def resolve_requested_layers(number_of_layers: int, specification: str) -> list[int]:
    if specification == "all":
        return list(range(number_of_layers))
    resolved: list[int] = []
    for requested in parse_int_list(specification):
        layer = number_of_layers - 1 if requested == -1 else requested
        if layer < 0 or layer >= number_of_layers:
            raise ValueError(f"Layer {requested} is outside 0..{number_of_layers - 1}")
        if layer not in resolved:
            resolved.append(layer)
    if not resolved:
        raise ValueError("At least one layer is required")
    return resolved


def locate_segment_spans(text: str, segments: list[dict[str, Any]]) -> list[tuple[int, int] | None]:
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    folded = text.casefold()
    for segment in segments:
        detail = str(segment.get("text", "")).strip()
        if not detail:
            spans.append(None)
            continue
        start = text.find(detail, cursor)
        if start < 0:
            start = folded.find(detail.casefold(), cursor)
        if start < 0:
            spans.append(None)
            continue
        spans.append((start, start + len(detail)))
        cursor = start + len(detail)
    return spans


def delete_span(text: str, span: tuple[int, int]) -> str:
    start, end = span
    left = text[:start].rstrip()
    right = text[end:].lstrip()
    if left and right:
        return f"{left} {right}"
    return left or right


def token_indices_for_span(offsets: list[tuple[int, int]], span: tuple[int, int] | None) -> list[int]:
    if span is None:
        return []
    start, end = span
    return [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start and left < end and right > left
    ]


def append_eos(encoded: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    import torch

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None or int(encoded["input_ids"][0, -1]) == int(eos_token_id):
        return encoded
    for key, value in list(encoded.items()):
        if key == "input_ids":
            suffix = torch.tensor([[eos_token_id]], dtype=value.dtype)
        elif key == "attention_mask":
            suffix = torch.ones((1, 1), dtype=value.dtype)
        elif key == "special_tokens_mask":
            suffix = torch.ones((1, 1), dtype=value.dtype)
        elif key == "offset_mapping":
            suffix = torch.zeros((1, 1, 2), dtype=value.dtype)
        elif value.ndim == 2 and value.shape[0] == 1:
            suffix = torch.zeros((1, 1), dtype=value.dtype)
        else:
            continue
        encoded[key] = torch.cat([value, suffix], dim=1)
    return encoded


def tokenize_without_truncation(text: str, tokenizer: Any) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
    )
    return append_eos(dict(encoded), tokenizer)


def model_context_limit(model: Any, tokenizer: Any) -> int | None:
    candidates = []
    for value in [
        getattr(model.config, "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ]:
        if isinstance(value, int) and 0 < value < 10**9:
            candidates.append(value)
    return min(candidates) if candidates else None


class GlobalRepresentationExtractor:
    def __init__(self, model: Any, tokenizer: Any, selected_layers: list[int]) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.transformer_layers = resolve_layers(model, _model_spec(model))
        self.selected_layers = selected_layers
        self.captured: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.captured_details: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.last_index = 0
        self.content_mask: Any = None
        self.detail_token_indices: list[list[int]] = []
        self.handles = [
            self.transformer_layers[layer].register_forward_hook(self._hook(layer))
            for layer in selected_layers
        ]

    def _hook(self, layer: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = layer_output_tensor(output)[0]
            mask = self.content_mask.to(tensor.device)
            eos = tensor[self.last_index].float().detach().cpu().numpy()
            mean = tensor[mask].float().mean(dim=0).detach().cpu().numpy()
            self.captured[layer] = (eos, mean)
            if self.detail_token_indices:
                detail_last = np.stack([
                    tensor[indices[-1]].float().detach().cpu().numpy()
                    for indices in self.detail_token_indices
                ])
                detail_mean = np.stack([
                    tensor[indices].float().mean(dim=0).detach().cpu().numpy()
                    for indices in self.detail_token_indices
                ])
                self.captured_details[layer] = (detail_last, detail_mean)

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def encode(
        self,
        encoded_cpu: dict[str, Any],
        device: Any,
        detail_token_indices: list[list[int]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        import torch

        special = encoded_cpu.pop("special_tokens_mask").bool()
        attention = encoded_cpu["attention_mask"].bool()
        content_mask = (attention & ~special)[0]
        if not bool(content_mask.any()):
            content_mask = attention[0]
        self.last_index = int(attention[0].sum().item()) - 1
        self.content_mask = content_mask
        self.detail_token_indices = detail_token_indices or []
        self.captured = {}
        self.captured_details = {}
        encoded = {key: value.to(device) for key, value in encoded_cpu.items()}
        backbone = getattr(self.model, "model", self.model)
        with torch.inference_mode():
            outputs = backbone(**encoded, use_cache=False, return_dict=True)
        final_state = outputs.last_hidden_state[0]
        final_layer = len(self.transformer_layers) - 1
        if final_layer in self.selected_layers:
            final_eos = final_state[self.last_index].float().cpu().numpy()
            final_mean = final_state[self.content_mask.to(final_state.device)].float().mean(dim=0).cpu().numpy()
            self.captured[final_layer] = (final_eos, final_mean)
            if self.detail_token_indices:
                final_detail_last = np.stack([
                    final_state[indices[-1]].float().cpu().numpy()
                    for indices in self.detail_token_indices
                ])
                final_detail_mean = np.stack([
                    final_state[indices].float().mean(dim=0).cpu().numpy()
                    for indices in self.detail_token_indices
                ])
                self.captured_details[final_layer] = (final_detail_last, final_detail_mean)
        missing = set(self.selected_layers) - set(self.captured)
        if missing:
            raise RuntimeError(f"Hooks did not capture layers {sorted(missing)}")
        eos = np.stack([self.captured[layer][0] for layer in self.selected_layers]).astype(np.float16)
        mean = np.stack([self.captured[layer][1] for layer in self.selected_layers]).astype(np.float16)
        detail_last = None
        detail_mean = None
        if self.detail_token_indices:
            missing_details = set(self.selected_layers) - set(self.captured_details)
            if missing_details:
                raise RuntimeError(f"Hooks did not capture detail states at layers {sorted(missing_details)}")
            detail_last = np.stack(
                [self.captured_details[layer][0] for layer in self.selected_layers],
                axis=1,
            ).astype(np.float16)
            detail_mean = np.stack(
                [self.captured_details[layer][1] for layer in self.selected_layers],
                axis=1,
            ).astype(np.float16)
        del outputs, final_state, encoded
        return eos, mean, detail_last, detail_mean


def _model_spec(model: Any) -> Any:
    from mechanistic_interpretability.shared.modeling import MODEL_SPECS

    model_name = str(getattr(model.config, "_name_or_path", ""))
    for spec in MODEL_SPECS.values():
        if spec.name == model_name:
            return spec
    for spec in MODEL_SPECS.values():
        try:
            resolve_layers(model, spec)
            return spec
        except AttributeError:
            continue
    raise AttributeError(f"Could not resolve transformer layers for {model_name}")


def record_metadata(record: dict[str, Any], vector_row: int, token_count: int) -> dict[str, Any]:
    return {
        "vector_row": vector_row,
        "record_id": record["record_id"],
        "record_type": record["record_type"],
        "story_key": record["story_key"],
        "video_id": record.get("video_id", ""),
        "condition": record.get("condition", ""),
        "summary_id": record.get("summary_id", 0),
        "source_age": record.get("source_age"),
        "target_age": record.get("target_age"),
        "compression_ratio": record.get("compression_ratio"),
        "token_count": token_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract untruncated whole-record representations and pooling controls.")
    parser.add_argument("--cohort", type=Path, default=Path("mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("mechanistic_interpretability/outputs/02_representation_extraction/gpt_oss_20b_global"))
    parser.add_argument("--model-key", default="gpt_oss_20b", choices=sorted(MODEL_SPECS))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--layers", default="0,4,8,12,16,20,-1")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-id", default="000")
    parser.add_argument("--max-tokens", type=int, default=131072)
    parser.add_argument("--record-types", default="original,summary")
    parser.add_argument("--overflow-policy", choices=["error", "skip"], default="error")
    parser.add_argument("--deletion-mode", choices=["none", "all"], default="none")
    parser.add_argument("--deletion-record-types", default="original")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    global_path = args.output_dir / f"global_features_{args.shard_id}.npz"
    metadata_path = args.output_dir / f"global_metadata_{args.shard_id}.csv"
    if global_path.exists() and metadata_path.exists() and not args.overwrite:
        print(f"Shard already complete: {global_path}")
        return

    allowed_types = {value.strip() for value in args.record_types.split(",") if value.strip()}
    deletion_types = {value.strip() for value in args.deletion_record_types.split(",") if value.strip()}
    records = [record for record in read_jsonl(args.cohort) if record.get("record_type") in allowed_types]
    stop = None if args.limit <= 0 else args.start + args.limit
    records = records[args.start:stop]
    model, tokenizer, device, spec = load_model(
        args.model_key,
        model_name=args.model_name,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    transformer_layers = resolve_layers(model, spec)
    selected_layers = resolve_requested_layers(len(transformer_layers), args.layers)
    extractor = GlobalRepresentationExtractor(model, tokenizer, selected_layers)
    configured_limit = model_context_limit(model, tokenizer)
    effective_limit = min(args.max_tokens, configured_limit) if configured_limit else args.max_tokens

    eos_vectors: list[np.ndarray] = []
    mean_vectors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    deletion_eos: list[np.ndarray] = []
    deletion_mean: list[np.ndarray] = []
    deletion_metadata: list[dict[str, Any]] = []
    detail_last_vectors: list[np.ndarray] = []
    detail_mean_vectors: list[np.ndarray] = []
    detail_metadata: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    try:
        for index, record in enumerate(records, start=1):
            encoded = tokenize_without_truncation(str(record["text"]), tokenizer)
            token_count = int(encoded["input_ids"].shape[1])
            if token_count > effective_limit:
                audit.append({
                    "record_id": record["record_id"],
                    "story_key": record["story_key"],
                    "status": "over_context",
                    "token_count": token_count,
                    "effective_context_limit": effective_limit,
                })
                if args.overflow_policy == "error":
                    raise ValueError(
                        f"{record['record_id']} has {token_count} tokens, exceeding {effective_limit}; "
                        "the experiment forbids truncation"
                    )
                continue
            offsets = [tuple(map(int, pair)) for pair in encoded.pop("offset_mapping")[0].tolist()]
            spans = locate_segment_spans(str(record["text"]), record.get("segments") or [])
            detail_indices = [token_indices_for_span(offsets, span) for span in spans]
            if any(not indices for indices in detail_indices):
                missing_count = sum(not indices for indices in detail_indices)
                raise ValueError(f"Could not map {missing_count} details to tokens in {record['record_id']}")
            eos, mean, record_detail_last, record_detail_mean = extractor.encode(encoded, device, detail_indices)
            vector_row = len(eos_vectors)
            eos_vectors.append(eos)
            mean_vectors.append(mean)
            metadata.append(record_metadata(record, vector_row, token_count))
            for detail_index, segment in enumerate(record.get("segments") or []):
                detail_last_vectors.append(record_detail_last[detail_index])
                detail_mean_vectors.append(record_detail_mean[detail_index])
                detail_metadata.append({
                    "detail_row": len(detail_last_vectors) - 1,
                    "record_id": record["record_id"],
                    "record_type": record["record_type"],
                    "story_key": record["story_key"],
                    "condition": record.get("condition", ""),
                    "summary_id": record.get("summary_id", 0),
                    "source_age": record.get("source_age"),
                    "target_age": record.get("target_age"),
                    "detail_index": detail_index,
                    "detail_text": segment.get("text", ""),
                    "detail_token_count": len(detail_indices[detail_index]),
                })
            audit.append({
                "record_id": record["record_id"],
                "story_key": record["story_key"],
                "status": "encoded_complete",
                "token_count": token_count,
                "effective_context_limit": effective_limit,
            })

            if args.deletion_mode == "all" and record.get("record_type") in deletion_types:
                if any(span is None for span in spans):
                    raise ValueError(f"Could not locate every detail in {record['record_id']}")
                for detail_index, span in enumerate(spans):
                    deleted_text = delete_span(str(record["text"]), span)
                    deleted_encoded = tokenize_without_truncation(deleted_text, tokenizer)
                    deleted_count = int(deleted_encoded["input_ids"].shape[1])
                    deleted_encoded.pop("offset_mapping")
                    deleted_eos_vector, deleted_mean_vector, _, _ = extractor.encode(deleted_encoded, device)
                    deletion_eos.append((eos.astype(np.float32) - deleted_eos_vector.astype(np.float32)).astype(np.float16))
                    deletion_mean.append((mean.astype(np.float32) - deleted_mean_vector.astype(np.float32)).astype(np.float16))
                    deletion_metadata.append({
                        "deletion_row": len(deletion_eos) - 1,
                        "record_id": record["record_id"],
                        "story_key": record["story_key"],
                        "detail_index": detail_index,
                        "detail_text": (record.get("segments") or [])[detail_index].get("text", ""),
                        "original_token_count": token_count,
                        "deleted_token_count": deleted_count,
                    })
            print(f"[{index}/{len(records)}] {record['record_id']} tokens={token_count}", flush=True)
    finally:
        extractor.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"global_audit_{args.shard_id}.csv", pd.DataFrame(audit))
    if not eos_vectors:
        print(f"No records selected for shard {args.shard_id}; nothing to write")
        return
    write_npz(
        global_path,
        eos_vectors=np.stack(eos_vectors),
        mean_vectors=np.stack(mean_vectors),
        layers=np.asarray(selected_layers, dtype=np.int16),
        model_key=np.asarray(args.model_key),
        context_limit=np.asarray(effective_limit, dtype=np.int32),
    )
    write_csv(metadata_path, pd.DataFrame(metadata))
    write_npz(
        args.output_dir / f"contextual_detail_features_{args.shard_id}.npz",
        last_token_vectors=np.stack(detail_last_vectors),
        mean_control_vectors=np.stack(detail_mean_vectors),
        layers=np.asarray(selected_layers, dtype=np.int16),
        model_key=np.asarray(args.model_key),
    )
    write_csv(args.output_dir / f"contextual_detail_metadata_{args.shard_id}.csv", pd.DataFrame(detail_metadata))
    if deletion_eos:
        write_npz(
            args.output_dir / f"deletion_features_{args.shard_id}.npz",
            eos_deltas=np.stack(deletion_eos),
            mean_deltas=np.stack(deletion_mean),
            layers=np.asarray(selected_layers, dtype=np.int16),
            model_key=np.asarray(args.model_key),
        )
        write_csv(args.output_dir / f"deletion_metadata_{args.shard_id}.csv", pd.DataFrame(deletion_metadata))
    print(f"Wrote {global_path}: {len(eos_vectors)} complete records")
    print(f"Wrote {len(detail_last_vectors)} contextual detail vectors")
    print(f"Mean pooling is stored only as a same-model control")
    if deletion_eos:
        print(f"Wrote {len(deletion_eos)} leave-one-detail-out influence vectors")


if __name__ == "__main__":
    main()
