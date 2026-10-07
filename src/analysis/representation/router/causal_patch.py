from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from shared.io import append_jsonl, read_jsonl
from shared.modeling import load_model, render_prompt, resolve_layers
from shared.prompts import strict_summary_prompt
from shared.research import evaluation_story_keys, select_work_shard


def find_gate(layer: Any) -> tuple[str, Any]:
    candidates = []
    for name, module in layer.named_modules():
        lower = name.lower()
        if name and (lower.endswith("gate") or lower.endswith("router")):
            candidates.append((name, module))
    if not candidates:
        raise AttributeError("No gate/router module found inside the selected transformer layer")
    return candidates[0]


def get_position(output: Any, position: int) -> Any:
    tensor = output[0] if isinstance(output, tuple) else output
    if tensor.ndim == 3:
        return tensor[:, position, :].detach().clone()
    if tensor.ndim == 2:
        return tensor[position : position + 1, :].detach().clone()
    raise ValueError(f"Unsupported router output shape {tuple(tensor.shape)}")


def replace_position(output: Any, position: int, value: Any) -> Any:
    tensor = output[0] if isinstance(output, tuple) else output
    modified = tensor.clone()
    if modified.ndim == 3:
        modified[:, position, :] = value.to(modified.device, modified.dtype)
    elif modified.ndim == 2:
        modified[position : position + 1, :] = value.to(modified.device, modified.dtype)
    else:
        raise ValueError(f"Unsupported router output shape {tuple(modified.shape)}")
    return (modified, *output[1:]) if isinstance(output, tuple) else modified


def prompt_for(original: dict[str, Any], target_age: str, condition: str) -> str:
    return strict_summary_prompt(
        original["text"],
        target_words=max(20, round(int(original["word_count"]) * 0.30)),
        target_age=target_age,
        condition=condition,
    )


def capture_router(model: Any, tokenizer: Any, device: Any, gate: Any, prompt: str) -> Any:
    import torch

    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        position = tensor.shape[1] - 1 if tensor.ndim == 3 else tensor.shape[0] - 1
        captured["value"] = get_position(output, position)

    handle = gate.register_forward_hook(hook)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        model(**encoded, use_cache=False)
    handle.remove()
    return captured["value"]


def nll(model: Any, tokenizer: Any, device: Any, gate: Any, prompt: str, summary: str, patch: Any | None) -> float:
    import torch
    import torch.nn.functional as functional

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + summary, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :8192].to(device)
    prompt_length = int(prompt_ids.shape[1])
    handle = None
    if patch is not None:
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            return replace_position(output, prompt_length - 1, patch)
        handle = gate.register_forward_hook(hook)
    with torch.inference_mode():
        logits = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), use_cache=False).logits
    if handle is not None:
        handle.remove()
    start = max(0, prompt_length - 1)
    losses = functional.cross_entropy(
        logits[:, start:-1, :].reshape(-1, logits.shape[-1]),
        full_ids[:, start + 1 :].reshape(-1),
        reduction="mean",
    )
    return float(losses.cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description="Causally patch MoE router logits while holding residual activations unpatched.")
    parser.add_argument("--cohort", type=Path, default=Path("results/features/summaries/dataset_index/cohort.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/analysis/representation/router/causal_router_patch.jsonl"))
    parser.add_argument("--model-key", default="gpt_oss_20b", choices=["qwen3_8b", "gpt_oss_20b", "qwen3_30b_a3b"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layers", default="8,12,16,20")
    parser.add_argument("--limit", type=int, default=0, help="Maximum summaries; 0 evaluates the full held-out cohort")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--story-split", type=Path, default=Path("results/analysis/representation/probes/layerwise_detail/story_split.csv"))
    args = parser.parse_args()

    rows = list(read_jsonl(args.cohort))
    evaluation = evaluation_story_keys(args.story_split)
    originals = {row["story_key"]: row for row in rows if row.get("record_type") == "original" and str(row.get("story_key")) in evaluation}
    summaries = sorted(
        [row for row in rows if row.get("record_type") == "summary" and row.get("condition") in {"younger", "older"} and str(row.get("story_key")) in evaluation],
        key=lambda row: (str(row["story_key"]), str(row["condition"]), int(row.get("summary_id", 1))),
    )
    summaries = select_work_shard(summaries, start=args.start, limit=args.limit, shard_index=args.shard_index, num_shards=args.num_shards)
    if not summaries:
        print("No held-out summaries selected for this shard")
        return
    completed = {
        (str(row.get("story_key")), str(row.get("condition")), int(row.get("summary_id", 1)), int(row.get("layer", -1)))
        for row in read_jsonl(args.output)
    } if args.output.exists() else set()
    model, tokenizer, device, spec = load_model(args.model_key, model_name=args.model_name, dtype=args.dtype)
    layers = resolve_layers(model, spec)
    for row_index, summary in enumerate(summaries, start=1):
        original = originals[summary["story_key"]]
        actual_prompt = render_prompt(tokenizer, prompt_for(original, original.get("source_age_range", ""), "actual"))
        target_prompt = render_prompt(tokenizer, prompt_for(original, summary.get("target_age_range", ""), summary["condition"]))
        for layer_index in [int(value) for value in args.layers.split(",") if value.strip()]:
            completion_key = (str(summary["story_key"]), str(summary["condition"]), int(summary.get("summary_id", 1)), layer_index)
            if completion_key in completed:
                continue
            gate_name, gate = find_gate(layers[layer_index])
            actual_router = capture_router(model, tokenizer, device, gate, actual_prompt)
            baseline = nll(model, tokenizer, device, gate, target_prompt, summary["text"], None)
            patched = nll(model, tokenizer, device, gate, target_prompt, summary["text"], actual_router)
            append_jsonl(
                args.output,
                [{
                    "story_key": summary["story_key"],
                    "condition": summary["condition"],
                    "summary_id": summary.get("summary_id", 1),
                    "layer": layer_index,
                    "gate_module": gate_name,
                    "baseline_nll": baseline,
                    "router_patch_nll": patched,
                    "router_patch_delta_nll": patched - baseline,
                    "evaluation_split": True,
                }],
            )
        print(f"[{row_index}/{len(summaries)}] {summary['record_id']}", flush=True)


if __name__ == "__main__":
    main()
