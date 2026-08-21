from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np

from mechanistic_interpretability.shared.io import append_jsonl, read_jsonl
from mechanistic_interpretability.shared.modeling import (
    layer_output_tensor,
    load_model,
    render_prompt,
    replace_layer_output,
    resolve_layers,
)
from mechanistic_interpretability.shared.prompts import strict_summary_prompt
from mechanistic_interpretability.shared.research import evaluation_story_keys, select_work_shard


def prompt_for(record: dict[str, Any], original: dict[str, Any], condition: str) -> str:
    target_words = max(20, round(int(original["word_count"]) * 0.30))
    return strict_summary_prompt(
        original["text"],
        target_words=target_words,
        target_age=str(record.get("target_age_range") or original.get("source_age_range", "")),
        condition=condition,
    )


def capture_prompt_state(model: Any, tokenizer: Any, device: Any, layer: Any, prompt: str) -> Any:
    import torch

    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["value"] = layer_output_tensor(output)[:, -1, :].detach().clone()

    handle = layer.register_forward_hook(hook)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        model(**encoded, use_cache=False)
    handle.remove()
    return captured["value"]


def sequence_nll(
    model: Any,
    tokenizer: Any,
    device: Any,
    layer: Any,
    prompt: str,
    summary: str,
    replacement: Any | None,
) -> tuple[float, list[float]]:
    import torch
    import torch.nn.functional as functional

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + summary, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt_length = int(prompt_ids.shape[1])
    full_ids = full_ids[:, :8192].to(device)
    attention_mask = torch.ones_like(full_ids)
    handle = None
    if replacement is not None and prompt_length - 1 < full_ids.shape[1]:
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = layer_output_tensor(output)
            modified = hidden.clone()
            modified[:, prompt_length - 1, :] = replacement.to(modified.device, modified.dtype)
            return replace_layer_output(output, modified)

        handle = layer.register_forward_hook(hook)
    with torch.inference_mode():
        logits = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False).logits
    if handle is not None:
        handle.remove()
    start = max(0, prompt_length - 1)
    prediction_logits = logits[:, start:-1, :]
    target_ids = full_ids[:, start + 1 :]
    losses = functional.cross_entropy(
        prediction_logits.reshape(-1, prediction_logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
    )
    values = losses.float().cpu().numpy().tolist()
    return float(np.mean(values)), values


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-forced causal patching of age-prompt representations.")
    parser.add_argument("--cohort", type=Path, default=Path("mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("mechanistic_interpretability/outputs/05_causal_patching/results.jsonl"))
    parser.add_argument("--model-key", default="gpt_oss_20b", choices=["qwen3_8b", "gpt_oss_20b", "qwen3_30b_a3b"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layers", default="8,12,16,20")
    parser.add_argument("--limit", type=int, default=0, help="Maximum summaries; 0 evaluates the full held-out cohort")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--story-split", type=Path, default=Path("mechanistic_interpretability/outputs/04_layerwise_probes/story_split.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    rows = list(read_jsonl(args.cohort))
    evaluation = evaluation_story_keys(args.story_split)
    originals = {row["story_key"]: row for row in rows if row.get("record_type") == "original" and str(row.get("story_key")) in evaluation}
    summaries = sorted(
        [row for row in rows if row.get("record_type") == "summary" and row.get("condition") in {"younger", "older"} and str(row.get("story_key")) in evaluation],
        key=lambda row: (str(row["story_key"]), str(row["condition"]), int(row.get("summary_id", 1))),
    )
    summaries = select_work_shard(summaries, start=args.start, limit=args.limit, shard_index=args.shard_index, num_shards=args.num_shards)
    if not summaries:
        print("No summaries selected for this shard")
        return
    if len(originals) < 2:
        raise ValueError("Causal patching requires at least two evaluation stories and one summary")
    completed = {
        (str(row.get("story_key")), str(row.get("condition")), int(row.get("summary_id", 1)), int(row.get("layer", -1)))
        for row in read_jsonl(args.output)
    } if args.output.exists() else set()
    model, tokenizer, device, spec = load_model(args.model_key, model_name=args.model_name, dtype=args.dtype)
    model_layers = resolve_layers(model, spec)
    requested_layers = [int(value) for value in args.layers.split(",") if value.strip()]
    rng = random.Random(args.seed)
    story_keys = sorted(originals)

    for row_index, summary in enumerate(summaries, start=1):
        original = originals[summary["story_key"]]
        source_descriptor = dict(summary)
        source_descriptor["target_age_range"] = original.get("source_age_range", "")
        source_prompt = render_prompt(tokenizer, prompt_for(source_descriptor, original, "actual"))
        target_prompt = render_prompt(tokenizer, prompt_for(summary, original, summary["condition"]))
        unrelated_keys = [key for key in story_keys if key != summary["story_key"]]
        random_original = originals[rng.choice(unrelated_keys)]
        random_descriptor = dict(source_descriptor)
        random_descriptor["target_age_range"] = random_original.get("source_age_range", "")
        random_prompt = render_prompt(tokenizer, prompt_for(random_descriptor, random_original, "actual"))
        for layer_index in requested_layers:
            completion_key = (str(summary["story_key"]), str(summary["condition"]), int(summary.get("summary_id", 1)), layer_index)
            if completion_key in completed:
                continue
            layer = model_layers[layer_index]
            actual_state = capture_prompt_state(model, tokenizer, device, layer, source_prompt)
            unrelated_state = capture_prompt_state(model, tokenizer, device, layer, random_prompt)
            baseline_nll, _ = sequence_nll(model, tokenizer, device, layer, target_prompt, summary["text"], None)
            patch_nll, patch_token_nll = sequence_nll(model, tokenizer, device, layer, target_prompt, summary["text"], actual_state)
            unrelated_nll, _ = sequence_nll(model, tokenizer, device, layer, target_prompt, summary["text"], unrelated_state)
            stable_seed = int.from_bytes(hashlib.sha256(f"{args.seed}::{summary['story_key']}::{layer_index}".encode()).digest()[:8], "big") % (2**63 - 1)
            generator = torch.Generator(device=actual_state.device).manual_seed(stable_seed)
            random_direction = torch.randn(actual_state.shape, generator=generator, device=actual_state.device, dtype=actual_state.dtype)
            random_direction = random_direction / random_direction.norm().clamp_min(1e-12) * actual_state.norm()
            random_nll, _ = sequence_nll(model, tokenizer, device, layer, target_prompt, summary["text"], random_direction)
            append_jsonl(
                args.output,
                [{
                    "story_key": summary["story_key"],
                    "condition": summary["condition"],
                    "summary_id": summary.get("summary_id", 1),
                    "layer": layer_index,
                    "baseline_nll": baseline_nll,
                    "actual_patch_nll": patch_nll,
                    "actual_patch_delta_nll": patch_nll - baseline_nll,
                    "unrelated_patch_nll": unrelated_nll,
                    "unrelated_patch_delta_nll": unrelated_nll - baseline_nll,
                    "random_patch_nll": random_nll,
                    "random_patch_delta_nll": random_nll - baseline_nll,
                    "actual_patch_token_nll": patch_token_nll,
                    "evaluation_split": True,
                    "seed": args.seed,
                }],
            )
        print(f"[{row_index}/{len(summaries)}] {summary['record_id']}", flush=True)


if __name__ == "__main__":
    main()
