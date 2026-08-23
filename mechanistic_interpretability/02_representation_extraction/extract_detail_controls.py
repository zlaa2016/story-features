from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mechanistic_interpretability.shared.io import read_jsonl, write_csv, write_npz
from mechanistic_interpretability.shared.modeling import layer_output_tensor, load_model, resolve_layers

global_extraction = importlib.import_module(
    "mechanistic_interpretability.02_representation_extraction.extract_global"
)
GlobalRepresentationExtractor = global_extraction.GlobalRepresentationExtractor
_model_spec = global_extraction._model_spec
delete_span = global_extraction.delete_span
locate_segment_spans = global_extraction.locate_segment_spans
model_context_limit = global_extraction.model_context_limit
resolve_requested_layers = global_extraction.resolve_requested_layers
tokenize_without_truncation = global_extraction.tokenize_without_truncation


def selected_detail_indices(number_of_details: int, maximum: int) -> list[int]:
    if number_of_details <= 0:
        return []
    if maximum <= 0 or number_of_details <= maximum:
        return list(range(number_of_details))
    return sorted(set(np.linspace(0, number_of_details - 1, maximum).round().astype(int).tolist()))


def detail_metadata(record: dict[str, Any], detail_index: int, detail_text: str) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "record_type": record["record_type"],
        "story_key": record["story_key"],
        "video_id": record.get("video_id", ""),
        "condition": record.get("condition", ""),
        "summary_id": record.get("summary_id", 0),
        "source_age": record.get("source_age"),
        "target_age": record.get("target_age"),
        "detail_index": detail_index,
        "detail_text": detail_text,
    }


def tokenize_detail_batch(texts: list[str], tokenizer: Any) -> dict[str, Any]:
    encoded_rows = []
    eos_token_id = tokenizer.eos_token_id
    for text in texts:
        row = tokenizer(text, truncation=False, return_special_tokens_mask=True)
        if eos_token_id is not None and (not row["input_ids"] or row["input_ids"][-1] != eos_token_id):
            row["input_ids"].append(eos_token_id)
            row["attention_mask"].append(1)
            row["special_tokens_mask"].append(1)
        encoded_rows.append(row)
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        return dict(tokenizer.pad(encoded_rows, padding=True, return_tensors="pt"))
    finally:
        tokenizer.padding_side = previous_padding_side


class BatchRepresentationExtractor:
    def __init__(self, model: Any, selected_layers: list[int]) -> None:
        self.model = model
        self.transformer_layers = resolve_layers(model, _model_spec(model))
        self.selected_layers = selected_layers
        self.last_indices: Any = None
        self.content_mask: Any = None
        self.captured: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.handles = [
            self.transformer_layers[layer].register_forward_hook(self._hook(layer))
            for layer in selected_layers
        ]

    def _vectors(self, tensor: Any) -> tuple[np.ndarray, np.ndarray]:
        import torch

        rows = torch.arange(tensor.shape[0], device=tensor.device)
        eos = tensor[rows, self.last_indices.to(tensor.device)].float()
        mask = self.content_mask.to(tensor.device)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (tensor.float() * mask.unsqueeze(-1)).sum(dim=1) / denominator
        return eos.detach().cpu().numpy(), mean.detach().cpu().numpy()

    def _hook(self, layer: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            self.captured[layer] = self._vectors(layer_output_tensor(output))

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def encode(self, encoded_cpu: dict[str, Any], device: Any) -> tuple[np.ndarray, np.ndarray]:
        import torch

        special = encoded_cpu.pop("special_tokens_mask").bool()
        attention = encoded_cpu["attention_mask"].bool()
        content = attention & ~special
        empty = ~content.any(dim=1)
        content[empty] = attention[empty]
        self.last_indices = attention.sum(dim=1) - 1
        self.content_mask = content
        self.captured = {}
        encoded = {key: value.to(device) for key, value in encoded_cpu.items()}
        backbone = getattr(self.model, "model", self.model)
        with torch.inference_mode():
            outputs = backbone(**encoded, use_cache=False, return_dict=True)
        final_layer = len(self.transformer_layers) - 1
        if final_layer in self.selected_layers:
            self.captured[final_layer] = self._vectors(outputs.last_hidden_state)
        missing = set(self.selected_layers) - set(self.captured)
        if missing:
            raise RuntimeError(f"Hooks did not capture layers {sorted(missing)}")
        eos = np.stack([self.captured[layer][0] for layer in self.selected_layers], axis=1).astype(np.float16)
        mean = np.stack([self.captured[layer][1] for layer in self.selected_layers], axis=1).astype(np.float16)
        del outputs, encoded
        return eos, mean


def load_global_reference(
    input_dir: Path,
    shard_id: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    feature_path = input_dir / f"global_features_{shard_id}.npz"
    metadata_path = input_dir / f"global_metadata_{shard_id}.csv"
    if not feature_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing global reference shard {shard_id} in {input_dir}")
    payload = np.load(feature_path, allow_pickle=False)
    metadata = pd.read_csv(metadata_path, low_memory=False)
    eos = payload["eos_vectors"]
    mean = payload["mean_vectors"]
    if len(metadata) != len(eos) or len(metadata) != len(mean):
        raise ValueError(f"Global reference shard {shard_id} has inconsistent row counts")
    lookup = {
        str(row.record_id): (eos[index].astype(np.float32), mean[index].astype(np.float32))
        for index, row in enumerate(metadata.itertuples(index=False))
    }
    return lookup, payload["layers"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract isolated-detail controls and leave-one-detail-out whole-record influence vectors."
    )
    parser.add_argument("--cohort", type=Path, default=Path("mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl"))
    parser.add_argument("--global-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", default="gpt_oss_20b", choices=["qwen3_8b", "gpt_oss_20b", "qwen3_30b_a3b"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--layers", default="0,4,8,12,16,20,-1")
    parser.add_argument("--mode", choices=["isolated", "deletion", "both"], default="isolated")
    parser.add_argument("--record-types", default="original,summary")
    parser.add_argument("--deletion-record-types", default="original")
    parser.add_argument("--max-deletions-per-record", type=int, default=0)
    parser.add_argument("--isolated-batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=131072)
    parser.add_argument("--overflow-policy", choices=["error", "skip"], default="error")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-id", default="000")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    isolated_feature_path = args.output_dir / f"isolated_detail_features_{args.shard_id}.npz"
    isolated_metadata_path = args.output_dir / f"isolated_detail_metadata_{args.shard_id}.csv"
    deletion_feature_path = args.output_dir / f"deletion_features_{args.shard_id}.npz"
    deletion_metadata_path = args.output_dir / f"deletion_metadata_{args.shard_id}.csv"
    want_isolated = args.mode in {"isolated", "both"} and (
        args.overwrite or not (isolated_feature_path.exists() and isolated_metadata_path.exists())
    )
    want_deletion = args.mode in {"deletion", "both"} and (
        args.overwrite or not (deletion_feature_path.exists() and deletion_metadata_path.exists())
    )
    if not want_isolated and not want_deletion:
        print(f"Requested controls already complete for shard {args.shard_id}")
        return

    allowed_types = {value.strip() for value in args.record_types.split(",") if value.strip()}
    deletion_types = {value.strip() for value in args.deletion_record_types.split(",") if value.strip()}
    records = [record for record in read_jsonl(args.cohort) if record.get("record_type") in allowed_types]
    stop = None if args.limit <= 0 else args.start + args.limit
    records = records[args.start:stop]
    global_lookup, reference_layers = load_global_reference(args.global_input_dir, args.shard_id)

    model, tokenizer, device, spec = load_model(
        args.model_key,
        model_name=args.model_name,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    transformer_layers = resolve_layers(model, spec)
    selected_layers = resolve_requested_layers(len(transformer_layers), args.layers)
    if not np.array_equal(reference_layers, np.asarray(selected_layers)):
        raise ValueError(f"Requested layers {selected_layers} do not match global reference {reference_layers.tolist()}")
    configured_limit = model_context_limit(model, tokenizer)
    effective_limit = min(args.max_tokens, configured_limit) if configured_limit else args.max_tokens

    isolated_eos: list[np.ndarray] = []
    isolated_mean: list[np.ndarray] = []
    isolated_rows: list[dict[str, Any]] = []
    deletion_eos: list[np.ndarray] = []
    deletion_mean: list[np.ndarray] = []
    deletion_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    batch_extractor = BatchRepresentationExtractor(model, selected_layers) if want_isolated else None
    single_extractor = GlobalRepresentationExtractor(model, tokenizer, selected_layers) if want_deletion else None
    try:
        if want_isolated and batch_extractor is not None:
            pending: list[tuple[dict[str, Any], int, str]] = []
            for record in records:
                for detail_index, segment in enumerate(record.get("segments") or []):
                    text = str(segment.get("text", "")).strip()
                    if text:
                        pending.append((record, detail_index, text))
            for batch_start in range(0, len(pending), args.isolated_batch_size):
                batch = pending[batch_start : batch_start + args.isolated_batch_size]
                encoded = tokenize_detail_batch([item[2] for item in batch], tokenizer)
                token_counts = encoded["attention_mask"].sum(dim=1).tolist()
                if max(token_counts, default=0) > effective_limit:
                    if args.overflow_policy == "error":
                        raise ValueError(f"Isolated detail exceeds context limit {effective_limit}")
                    keep = [index for index, count in enumerate(token_counts) if count <= effective_limit]
                    batch = [batch[index] for index in keep]
                    if not batch:
                        continue
                    encoded = tokenize_detail_batch([item[2] for item in batch], tokenizer)
                    token_counts = encoded["attention_mask"].sum(dim=1).tolist()
                eos, mean = batch_extractor.encode(encoded, device)
                for row_index, (record, detail_index, text) in enumerate(batch):
                    isolated_eos.append(eos[row_index])
                    isolated_mean.append(mean[row_index])
                    row = detail_metadata(record, detail_index, text)
                    row.update({"isolated_row": len(isolated_rows), "token_count": int(token_counts[row_index])})
                    isolated_rows.append(row)
                print(f"isolated details {min(batch_start + len(batch), len(pending))}/{len(pending)}", flush=True)

        if want_deletion and single_extractor is not None:
            for record_index, record in enumerate(records, start=1):
                if record.get("record_type") not in deletion_types:
                    continue
                reference = global_lookup.get(str(record["record_id"]))
                if reference is None:
                    raise KeyError(f"No global reference vector for {record['record_id']}")
                spans = locate_segment_spans(str(record["text"]), record.get("segments") or [])
                if any(span is None for span in spans):
                    raise ValueError(f"Could not locate every detail in {record['record_id']}")
                chosen = selected_detail_indices(len(spans), args.max_deletions_per_record)
                for detail_index in chosen:
                    deleted_text = delete_span(str(record["text"]), spans[detail_index])
                    encoded = tokenize_without_truncation(deleted_text, tokenizer)
                    token_count = int(encoded["input_ids"].shape[1])
                    if token_count > effective_limit:
                        if args.overflow_policy == "error":
                            raise ValueError(f"Deleted record exceeds context limit {effective_limit}")
                        continue
                    encoded.pop("offset_mapping")
                    deleted_eos, deleted_mean, _, _ = single_extractor.encode(encoded, device)
                    deletion_eos.append((reference[0] - deleted_eos.astype(np.float32)).astype(np.float16))
                    deletion_mean.append((reference[1] - deleted_mean.astype(np.float32)).astype(np.float16))
                    text = str((record.get("segments") or [])[detail_index].get("text", "")).strip()
                    row = detail_metadata(record, detail_index, text)
                    row.update({"deletion_row": len(deletion_rows), "deleted_token_count": token_count})
                    deletion_rows.append(row)
                audit_rows.append({
                    "record_id": record["record_id"],
                    "status": "deletion_complete",
                    "available_details": len(spans),
                    "deleted_details": len(chosen),
                })
                print(f"deletion records {record_index}/{len(records)} {record['record_id']} details={len(chosen)}", flush=True)
    finally:
        if batch_extractor is not None:
            batch_extractor.close()
        if single_extractor is not None:
            single_extractor.close()

    if want_isolated:
        if isolated_eos:
            write_npz(
                isolated_feature_path,
                eos_vectors=np.stack(isolated_eos),
                mean_vectors=np.stack(isolated_mean),
                layers=np.asarray(selected_layers, dtype=np.int16),
                model_key=np.asarray(args.model_key),
            )
            write_csv(isolated_metadata_path, pd.DataFrame(isolated_rows))
        else:
            write_csv(isolated_metadata_path, pd.DataFrame(columns=["isolated_row", "record_id"]))
        print(f"Wrote {len(isolated_rows)} isolated detail embeddings")
    if want_deletion:
        if deletion_eos:
            write_npz(
                deletion_feature_path,
                eos_deltas=np.stack(deletion_eos),
                mean_deltas=np.stack(deletion_mean),
                layers=np.asarray(selected_layers, dtype=np.int16),
                model_key=np.asarray(args.model_key),
            )
            write_csv(deletion_metadata_path, pd.DataFrame(deletion_rows))
        else:
            write_csv(deletion_metadata_path, pd.DataFrame(columns=["deletion_row", "record_id"]))
        write_csv(args.output_dir / f"deletion_audit_{args.shard_id}.csv", pd.DataFrame(audit_rows))
        print(f"Wrote {len(deletion_rows)} leave-one-detail-out influence embeddings")


if __name__ == "__main__":
    main()
