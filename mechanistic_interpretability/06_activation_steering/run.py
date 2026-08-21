from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mechanistic_interpretability.shared.io import append_jsonl, parse_float_list, read_jsonl, word_count
from mechanistic_interpretability.shared.modeling import last_token_addition_hook, load_model, render_prompt, resolve_layers
from mechanistic_interpretability.shared.prompts import strict_summary_prompt
from mechanistic_interpretability.shared.research import evaluation_story_keys, select_work_shard


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summaries under matched age-direction and random-direction steering.")
    parser.add_argument("--cohort", type=Path, default=Path("mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl"))
    parser.add_argument("--directions", type=Path, default=Path("mechanistic_interpretability/outputs/04_layerwise_probes/age_directions.npz"))
    parser.add_argument("--output", type=Path, default=Path("mechanistic_interpretability/outputs/06_activation_steering/generations.jsonl"))
    parser.add_argument("--story-split", type=Path, default=Path("mechanistic_interpretability/outputs/04_layerwise_probes/story_split.csv"))
    parser.add_argument("--model-key", default="gpt_oss_20b", choices=["qwen3_8b", "gpt_oss_20b", "qwen3_30b_a3b"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--direction-name", default="older_minus_actual")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--alphas", default="-2,-1,-0.5,0,0.5,1,2")
    parser.add_argument("--limit", type=int, default=0, help="Maximum stories; 0 evaluates the full held-out cohort")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    payload = np.load(args.directions, allow_pickle=False)
    saved_layers = payload["layers"].astype(int).tolist()
    if args.direction_name not in payload:
        raise KeyError(f"Missing {args.direction_name}; available: {payload.files}")
    if args.layer not in saved_layers:
        raise ValueError(f"Layer {args.layer} not in saved layers {saved_layers}")
    direction = torch.from_numpy(payload[args.direction_name][saved_layers.index(args.layer)].astype(np.float32))
    evaluation = evaluation_story_keys(args.story_split)
    originals = sorted(
        (
            row for row in read_jsonl(args.cohort)
            if row.get("record_type") == "original" and str(row.get("story_key")) in evaluation
        ),
        key=lambda row: str(row["story_key"]),
    )
    originals = select_work_shard(originals, start=args.start, limit=args.limit, shard_index=args.shard_index, num_shards=args.num_shards)
    if not originals:
        print("No evaluation stories selected for this shard")
        return
    completed = {
        (str(row.get("story_key")), str(row.get("control")), float(row.get("alpha")))
        for row in read_jsonl(args.output)
    } if args.output.exists() else set()
    model, tokenizer, device, spec = load_model(args.model_key, model_name=args.model_name, dtype=args.dtype)
    layer = resolve_layers(model, spec)[args.layer]
    for story_index, original in enumerate(originals, start=1):
        stable_seed = int.from_bytes(
            hashlib.sha256(f"{args.seed}:{original['story_key']}:{args.layer}".encode()).digest()[:8],
            "big",
        ) % (2**63 - 1)
        generator = torch.Generator(device=device).manual_seed(stable_seed)
        random_direction = torch.randn(direction.shape, generator=generator, device=device)
        projection = torch.dot(random_direction, direction.to(device)) / direction.to(device).pow(2).sum().clamp_min(1e-12)
        random_direction = random_direction - projection * direction.to(device)
        random_direction = random_direction / random_direction.norm().clamp_min(1e-12) * direction.to(device).norm()
        target_words = max(20, round(int(original["word_count"]) * 0.30))
        prompt = render_prompt(
            tokenizer,
            strict_summary_prompt(
                original["text"],
                target_words=target_words,
                target_age=original.get("source_age_range", ""),
                condition="actual",
            ),
        )
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        for control_name, vector in [("age_direction", direction), ("random_direction", random_direction.cpu())]:
            for alpha in parse_float_list(args.alphas):
                if (str(original["story_key"]), control_name, float(alpha)) in completed:
                    continue
                handle = None
                if alpha != 0:
                    handle = layer.register_forward_hook(last_token_addition_hook(vector, alpha))
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                if handle is not None:
                    handle.remove()
                new_ids = generated[0, encoded["input_ids"].shape[1] :]
                text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                append_jsonl(
                    args.output,
                    [{
                        "story_key": original["story_key"],
                        "model_key": args.model_key,
                        "generator_model": args.model_name or spec.name,
                        "layer": args.layer,
                        "direction_name": args.direction_name,
                        "control": control_name,
                        "alpha": alpha,
                        "requested_target_words": target_words,
                        "summary_text": text,
                        "achieved_words": word_count(text),
                        "achieved_compression_ratio": word_count(text) / max(1, int(original["word_count"])),
                        "evaluation_split": True,
                        "seed": args.seed,
                        "control_seed": stable_seed,
                    }],
                )
        print(f"[{story_index}/{len(originals)}] {original['story_key']}", flush=True)


if __name__ == "__main__":
    main()
