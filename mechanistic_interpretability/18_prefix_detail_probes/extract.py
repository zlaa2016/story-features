#!/usr/bin/env python3
"""Extract controlled causal-LM representations and distributions for story details.

For target detail D and prefix P, the extractor runs two token-controlled sequences:

    prefix probe:  BOS + P + DELIMITER + READOUT
    detail probe:  BOS + P + DELIMITER + D + DELIMITER + READOUT

It stores the first, last, and mean hidden state over D; the same fixed readout endpoint before
and after D; teacher-forced token NLL for D; tokenwise JS divergence from the isolated-detail
distribution; and JS divergence between the two readout next-token distributions.  The full
vocabulary distributions are reduced online and are never written to disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from mechanistic_interpretability.shared.io import parse_int_list, read_jsonl, write_csv, write_npz
from mechanistic_interpretability.shared.modeling import (
    MODEL_SPECS,
    layer_output_tensor,
    load_model,
    resolve_layers,
)


def resolve_requested_layers(number_of_layers: int, specification: str) -> list[int]:
    if specification == "all":
        return list(range(number_of_layers))
    resolved: list[int] = []
    for requested in parse_int_list(specification):
        layer = number_of_layers - 1 if requested == -1 else requested
        if not 0 <= layer < number_of_layers:
            raise ValueError(f"Layer {requested} is outside 0..{number_of_layers - 1}")
        if layer not in resolved:
            resolved.append(layer)
    if not resolved:
        raise ValueError("At least one layer is required")
    return resolved


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(value)


def join_token_segments(segments: Sequence[Sequence[int]], delimiter: Sequence[int]) -> list[int]:
    output: list[int] = []
    for segment in segments:
        if output:
            output.extend(delimiter)
        output.extend(segment)
    return output


def exact_length_foreign_prefix(
    streams: dict[str, list[int]], story_key: str, length: int, rng: np.random.Generator
) -> tuple[list[int], str]:
    if length <= 0:
        return [], ""
    donors = [key for key, stream in streams.items() if key != story_key and stream]
    if not donors:
        raise ValueError("A foreign-prefix condition requires at least two nonempty stories")
    pieces: list[int] = []
    used: list[str] = []
    while len(pieces) < length:
        donor = donors[int(rng.integers(len(donors)))]
        stream = streams[donor]
        take = min(length - len(pieces), len(stream))
        start = int(rng.integers(len(stream) - take + 1))
        pieces.extend(stream[start : start + take])
        used.append(donor)
    return pieces, "|".join(used)


def js_from_log_probabilities(log_p: Any, log_q: Any) -> Any:
    """Exact Jensen-Shannon divergence over the final vocabulary axis."""
    import torch

    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    return 0.5 * (
        (log_p.exp() * (log_p - log_m)).sum(dim=-1)
        + (log_q.exp() * (log_q - log_m)).sum(dim=-1)
    )


def entropy_from_log_probabilities(log_p: Any) -> Any:
    return -(log_p.exp() * log_p).sum(dim=-1)


@dataclass
class ProbeResult:
    first: np.ndarray
    last: np.ndarray
    mean: np.ndarray
    readout: np.ndarray
    token_nll: np.ndarray
    detail_log_probs: Any
    readout_log_probs: Any
    token_vectors: np.ndarray | None


class CausalProbeExtractor:
    """Layer-hook extractor that applies the LM head only at requested positions."""

    def __init__(self, model: Any, spec: Any, selected_layers: list[int]) -> None:
        self.model = model
        self.backbone = getattr(model, "model", model)
        self.output_head = model.get_output_embeddings()
        if self.output_head is None:
            raise RuntimeError("The causal LM has no output embedding / LM head")
        self.layers = resolve_layers(model, spec)
        self.selected_layers = selected_layers
        self.detail_start = 0
        self.detail_end = 0
        self.readout_index = 0
        self.keep_tokens = False
        self.captured: dict[int, tuple[Any, Any, Any, Any, Any | None]] = {}
        self.handles = [self.layers[layer].register_forward_hook(self._hook(layer)) for layer in selected_layers]

    def _summaries(self, tensor: Any) -> tuple[Any, Any, Any, Any, Any | None]:
        span = tensor[0, self.detail_start : self.detail_end]
        readout = tensor[0, self.readout_index]
        saved_span = span.float().detach().cpu() if self.keep_tokens else None
        return (
            span[0].float().detach().cpu(),
            span[-1].float().detach().cpu(),
            span.float().mean(dim=0).detach().cpu(),
            readout.float().detach().cpu(),
            saved_span,
        )

    def _hook(self, layer: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            self.captured[layer] = self._summaries(layer_output_tensor(output))

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def encode(
        self,
        input_ids: Sequence[int],
        detail_start: int,
        detail_end: int,
        readout_index: int,
        device: Any,
        *,
        save_token_states: bool = False,
    ) -> ProbeResult:
        import torch
        import torch.nn.functional as functional

        if not (0 < detail_start < detail_end <= readout_index < len(input_ids)):
            raise ValueError("Invalid detail/readout positions")
        self.detail_start = detail_start
        self.detail_end = detail_end
        self.readout_index = readout_index
        self.keep_tokens = save_token_states
        self.captured = {}
        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
        attention = torch.ones_like(ids)
        with torch.inference_mode():
            outputs = self.backbone(input_ids=ids, attention_mask=attention, use_cache=False, return_dict=True)
            final_state = outputs.last_hidden_state
            final_layer = len(self.layers) - 1
            if final_layer in self.selected_layers:
                self.captured[final_layer] = self._summaries(final_state)
            missing = set(self.selected_layers) - set(self.captured)
            if missing:
                raise RuntimeError(f"Layer hooks missed {sorted(missing)}")

            predictor = final_state[:, detail_start - 1 : detail_end - 1]
            detail_logits = self.output_head(predictor).float()[0]
            targets = ids[0, detail_start:detail_end]
            token_nll = functional.cross_entropy(detail_logits, targets, reduction="none")
            detail_log_probs = functional.log_softmax(detail_logits, dim=-1)
            readout_logits = self.output_head(final_state[:, readout_index : readout_index + 1]).float()[0, 0]
            readout_log_probs = functional.log_softmax(readout_logits, dim=-1)

        first = np.stack([self.captured[layer][0].numpy() for layer in self.selected_layers])
        last = np.stack([self.captured[layer][1].numpy() for layer in self.selected_layers])
        mean = np.stack([self.captured[layer][2].numpy() for layer in self.selected_layers])
        readout = np.stack([self.captured[layer][3].numpy() for layer in self.selected_layers])
        token_vectors = None
        if save_token_states:
            token_vectors = np.stack([
                self.captured[layer][4].numpy()
                for layer in self.selected_layers
            ], axis=1)
        result = ProbeResult(
            first=first.astype(np.float16),
            last=last.astype(np.float16),
            mean=mean.astype(np.float16),
            readout=readout.astype(np.float16),
            token_nll=token_nll.float().cpu().numpy(),
            detail_log_probs=detail_log_probs.detach(),
            readout_log_probs=readout_log_probs.detach(),
            token_vectors=None if token_vectors is None else token_vectors.astype(np.float16),
        )
        del outputs, final_state, detail_logits, readout_logits, ids, attention
        return result


def readout_only(
    extractor: CausalProbeExtractor,
    input_ids: Sequence[int],
    readout_start: int,
    device: Any,
) -> tuple[np.ndarray, Any]:
    """Use the readout sequence itself as a nonempty span and return its endpoint."""
    result = extractor.encode(
        input_ids,
        detail_start=readout_start,
        detail_end=len(input_ids),
        readout_index=len(input_ids) - 1,
        device=device,
    )
    return result.readout, result.readout_log_probs


def tokenize_text(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record.get("record_id", ""),
        "story_key": str(record.get("story_key", "")),
        "source_age": record.get("source_age"),
        "video_id": record.get("video_id", ""),
    }


def prefix_indices(boundary: int, target: int, exclude_target: bool) -> list[int]:
    indices = list(range(boundary))
    return [index for index in indices if not (exclude_target and index == target)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("mechanistic_interpretability/outputs/18_prefix_detail_probes/gpt_oss_20b"))
    parser.add_argument("--model-key", default="gpt_oss_20b", choices=sorted(MODEL_SPECS))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--layers", default="0,6,12,18,-1")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-stories", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--shard-id", default="000")
    parser.add_argument("--prefix-mode", choices=["native", "all"], default="native")
    parser.add_argument("--prefix-step", type=int, default=1)
    parser.add_argument("--conditions", default="isolated,true,shuffled,foreign")
    parser.add_argument("--delimiter-text", default="\n\n")
    parser.add_argument("--readout-text", default="\n")
    parser.add_argument("--max-tokens", type=int, default=131072)
    parser.add_argument("--max-details-per-story", type=int, default=0)
    parser.add_argument("--exclude-target-from-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-token-states", action="store_true")
    parser.add_argument("--overflow-policy", choices=["skip", "error"], default="skip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--permutations",
        type=int,
        default=1,
        help="Independent shuffled/foreign controls per target-prefix pair.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    feature_path = args.output_dir / f"probe_features_{args.shard_id}.npz"
    metadata_path = args.output_dir / f"probe_metadata_{args.shard_id}.csv"
    if feature_path.exists() and metadata_path.exists() and not args.overwrite:
        print(f"Shard already complete: {feature_path}")
        return
    if args.prefix_step < 1:
        raise ValueError("--prefix-step must be at least 1")
    if args.permutations < 1:
        raise ValueError("--permutations must be at least 1")

    allowed = [item.strip() for item in args.conditions.split(",") if item.strip()]
    unknown = set(allowed) - {"isolated", "true", "shuffled", "foreign"}
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")

    originals = [record for record in read_jsonl(args.cohort) if record.get("record_type") == "original"]
    if 0 < args.sample_stories < len(originals):
        rng = np.random.default_rng(args.sample_seed)
        selected = np.sort(rng.choice(len(originals), size=args.sample_stories, replace=False))
        originals = [originals[int(index)] for index in selected]
    stop = None if args.limit <= 0 else args.start + args.limit
    records = originals[args.start:stop]
    if not records:
        print("No original records selected")
        return

    model, tokenizer, device, spec = load_model(
        args.model_key,
        model_name=args.model_name,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    transformer_layers = resolve_layers(model, spec)
    selected_layers = resolve_requested_layers(len(transformer_layers), args.layers)
    extractor = CausalProbeExtractor(model, spec, selected_layers)

    delimiter_ids = tokenize_text(tokenizer, args.delimiter_text)
    readout_ids = tokenize_text(tokenizer, args.readout_text)
    if not readout_ids:
        raise ValueError("--readout-text must produce at least one token")
    bos_ids = [int(tokenizer.bos_token_id)] if tokenizer.bos_token_id is not None else []
    if not delimiter_ids and not bos_ids:
        raise ValueError("The delimiter must produce a token when the tokenizer has no BOS token")

    story_segments: dict[str, list[list[int]]] = {}
    for record in originals:
        key = str(record.get("story_key", ""))
        segments = [
            tokenize_text(tokenizer, str(segment.get("text", "")).strip())
            for segment in (record.get("segments") or [])
            if str(segment.get("text", "")).strip()
        ]
        story_segments[key] = [segment for segment in segments if segment]
    streams = {
        key: join_token_segments(segments, delimiter_ids)
        for key, segments in story_segments.items()
    }

    rows: list[dict[str, Any]] = []
    first_vectors: list[np.ndarray] = []
    last_vectors: list[np.ndarray] = []
    mean_vectors: list[np.ndarray] = []
    prefix_readouts: list[np.ndarray] = []
    post_readouts: list[np.ndarray] = []
    token_ids_flat: list[np.ndarray] = []
    token_nll_flat: list[np.ndarray] = []
    token_js_flat: list[np.ndarray] = []
    token_states_flat: list[np.ndarray] = []
    token_cursor = 0
    audit: list[dict[str, Any]] = []

    try:
        empty_prefix_sequence = bos_ids + delimiter_ids + readout_ids
        empty_prefix_readout, empty_prefix_distribution = readout_only(
            extractor,
            empty_prefix_sequence,
            len(empty_prefix_sequence) - len(readout_ids),
            device,
        )
        for record_number, record in enumerate(records, start=1):
            story_key = str(record.get("story_key", ""))
            detail_ids = story_segments.get(story_key, [])
            segments = [segment for segment in (record.get("segments") or []) if str(segment.get("text", "")).strip()]
            if args.max_details_per_story > 0:
                detail_ids = detail_ids[: args.max_details_per_story]
                segments = segments[: args.max_details_per_story]
            n_details = len(detail_ids)
            record_written = 0
            record_skipped = 0
            for target_index, target_ids in enumerate(detail_ids):
                # The isolated result is both a row (if requested) and the probability baseline for
                # every contextual condition of this exact target token sequence.
                isolated_prefix: list[int] = []
                isolated_base = bos_ids + delimiter_ids
                isolated_sequence = isolated_base + target_ids + delimiter_ids + readout_ids
                if len(isolated_sequence) > args.max_tokens:
                    record_skipped += 1
                    continue
                isolated_detail_start = len(isolated_base)
                isolated_detail_end = isolated_detail_start + len(target_ids)
                isolated_readout_start = isolated_detail_end + len(delimiter_ids)
                isolated_result = extractor.encode(
                    isolated_sequence,
                    isolated_detail_start,
                    isolated_detail_end,
                    len(isolated_sequence) - 1,
                    device,
                    save_token_states=args.save_token_states,
                )

                if args.prefix_mode == "native":
                    boundaries = [target_index]
                else:
                    boundaries = list(range(0, n_details + 1, args.prefix_step))
                    if boundaries[-1] != n_details:
                        boundaries.append(n_details)

                condition_boundaries: list[tuple[str, int, int]] = []
                if "isolated" in allowed:
                    condition_boundaries.append(("isolated", -1, 0))
                for boundary in boundaries:
                    for condition in allowed:
                        if condition == "isolated":
                            continue
                        repeats = args.permutations if condition in {"shuffled", "foreign"} else 1
                        condition_boundaries.extend((condition, boundary, permutation) for permutation in range(repeats))

                for condition, boundary, permutation in condition_boundaries:
                    donor = ""
                    removed = False
                    if condition == "isolated":
                        prefix = isolated_prefix
                    else:
                        indices = prefix_indices(boundary, target_index, args.exclude_target_from_prefix)
                        removed = args.exclude_target_from_prefix and target_index < boundary
                        true_prefix = join_token_segments([detail_ids[index] for index in indices], delimiter_ids)
                        if condition == "true":
                            prefix = true_prefix
                        elif condition == "shuffled":
                            rng = stable_rng(args.seed, story_key, target_index, boundary, condition, permutation)
                            shuffled = list(indices)
                            rng.shuffle(shuffled)
                            prefix = join_token_segments([detail_ids[index] for index in shuffled], delimiter_ids)
                            if len(prefix) != len(true_prefix):
                                raise AssertionError("Shuffling changed prefix token length")
                        elif condition == "foreign":
                            rng = stable_rng(args.seed, story_key, target_index, boundary, condition, permutation)
                            prefix, donor = exact_length_foreign_prefix(streams, story_key, len(true_prefix), rng)
                        else:  # guarded by CLI validation
                            raise AssertionError(condition)

                    base = bos_ids + prefix + delimiter_ids
                    detail_sequence = base + target_ids + delimiter_ids + readout_ids
                    prefix_sequence = base + readout_ids
                    if max(len(detail_sequence), len(prefix_sequence)) > args.max_tokens:
                        record_skipped += 1
                        if args.overflow_policy == "error":
                            raise ValueError(
                                f"{story_key} target={target_index} prefix={boundary} has "
                                f"{len(detail_sequence)} tokens > {args.max_tokens}"
                            )
                        continue

                    detail_start = len(base)
                    detail_end = detail_start + len(target_ids)
                    readout_start = detail_end + len(delimiter_ids)
                    if condition == "isolated":
                        result = isolated_result
                    else:
                        result = extractor.encode(
                            detail_sequence,
                            detail_start,
                            detail_end,
                            len(detail_sequence) - 1,
                            device,
                            save_token_states=args.save_token_states,
                        )
                    if not prefix:
                        prefix_readout = empty_prefix_readout
                        prefix_distribution = empty_prefix_distribution
                    else:
                        prefix_readout, prefix_distribution = readout_only(
                            extractor, prefix_sequence, len(prefix_sequence) - len(readout_ids), device
                        )

                    token_js = js_from_log_probabilities(
                        result.detail_log_probs, isolated_result.detail_log_probs
                    ).float().cpu().numpy()
                    readout_js = float(js_from_log_probabilities(
                        prefix_distribution, result.readout_log_probs
                    ).float().cpu())
                    prefix_entropy = float(entropy_from_log_probabilities(prefix_distribution).float().cpu())
                    post_entropy = float(entropy_from_log_probabilities(result.readout_log_probs).float().cpu())
                    token_offset = token_cursor
                    probe_row = len(rows)
                    metadata = record_metadata(record)
                    metadata.update({
                        "probe_row": probe_row,
                        "target_detail_index": target_index,
                        "target_detail_text": str(segments[target_index].get("text", "")),
                        "prefix_condition": condition,
                        "prefix_mode": args.prefix_mode,
                        "permutation_id": permutation,
                        "prefix_index": boundary,
                        "prefix_fraction": (boundary / n_details) if boundary >= 0 and n_details else np.nan,
                        "native_prefix": bool(condition == "true" and boundary == target_index),
                        "target_removed_from_prefix": removed,
                        "foreign_donor_story": donor,
                        "n_details_in_story": n_details,
                        "prefix_tokens": len(prefix),
                        "detail_tokens": len(target_ids),
                        "sequence_tokens": len(detail_sequence),
                        "token_offset": token_offset,
                        "mean_detail_nll": float(result.token_nll.mean()),
                        "sum_detail_nll": float(result.token_nll.sum()),
                        "isolated_mean_detail_nll": float(isolated_result.token_nll.mean()),
                        "detail_nll_delta_from_isolated": float(result.token_nll.mean() - isolated_result.token_nll.mean()),
                        "mean_token_js_from_isolated": float(token_js.mean()),
                        "readout_output_js": readout_js,
                        "prefix_readout_entropy": prefix_entropy,
                        "post_detail_readout_entropy": post_entropy,
                        "readout_entropy_change": post_entropy - prefix_entropy,
                    })
                    rows.append(metadata)
                    first_vectors.append(result.first)
                    last_vectors.append(result.last)
                    mean_vectors.append(result.mean)
                    prefix_readouts.append(prefix_readout)
                    post_readouts.append(result.readout)
                    token_ids_flat.append(np.asarray(target_ids, dtype=np.int32))
                    token_nll_flat.append(result.token_nll.astype(np.float32))
                    token_js_flat.append(token_js.astype(np.float32))
                    if args.save_token_states and result.token_vectors is not None:
                        token_states_flat.append(result.token_vectors)
                    token_cursor += len(target_ids)
                    record_written += 1
            audit.append({
                "record_id": record.get("record_id", ""),
                "story_key": story_key,
                "details": n_details,
                "probe_rows": record_written,
                "skipped_overflow": record_skipped,
                "status": "complete",
            })
            print(
                f"[{record_number}/{len(records)}] {story_key} details={n_details} "
                f"rows={record_written} skipped={record_skipped}",
                flush=True,
            )
    finally:
        extractor.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"probe_audit_{args.shard_id}.csv", pd.DataFrame(audit))
    if not rows:
        print("No probe rows were produced")
        return
    arrays: dict[str, Any] = {
        "first_detail_vectors": np.stack(first_vectors),
        "last_detail_vectors": np.stack(last_vectors),
        "mean_detail_vectors": np.stack(mean_vectors),
        "prefix_readout_vectors": np.stack(prefix_readouts),
        "post_detail_readout_vectors": np.stack(post_readouts),
        "detail_token_ids": np.concatenate(token_ids_flat),
        "detail_token_nll": np.concatenate(token_nll_flat),
        "detail_token_js_from_isolated": np.concatenate(token_js_flat),
        "layers": np.asarray(selected_layers, dtype=np.int16),
        "token_state_layers": np.asarray(selected_layers, dtype=np.int16),
    }
    if token_states_flat:
        arrays["detail_token_vectors"] = np.concatenate(token_states_flat, axis=0)
    write_npz(feature_path, **arrays)
    write_csv(metadata_path, pd.DataFrame(rows))
    config = vars(args).copy()
    config.update({
        "cohort": str(args.cohort),
        "output_dir": str(args.output_dir),
        "selected_layers": selected_layers,
        "delimiter_token_ids": delimiter_ids,
        "readout_token_ids": readout_ids,
        "records_selected": len(records),
        "probe_rows": len(rows),
        "note": "Full-vocabulary distributions were reduced online; token NLL and exact JS are saved.",
    })
    (args.output_dir / f"probe_config_{args.shard_id}.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote {metadata_path}: {len(rows):,} probes")
    print(f"Wrote {feature_path}")


if __name__ == "__main__":
    main()
