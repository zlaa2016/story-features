from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mechanistic_interpretability.shared.io import append_jsonl, read_jsonl
from mechanistic_interpretability.shared.modeling import last_token_addition_hook, load_model, render_prompt, resolve_layers
from mechanistic_interpretability.shared.prompts import strict_summary_prompt
from mechanistic_interpretability.shared.sae import load_sae
from mechanistic_interpretability.shared.research import evaluation_story_keys, select_work_shard


def main() -> None:
    parser = argparse.ArgumentParser(description="Causally steer summaries with age-associated SAE decoder features.")
    parser.add_argument("--cohort", type=Path, default=Path("mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--story-split", type=Path, default=Path("mechanistic_interpretability/outputs/04_layerwise_probes/story_split.csv"))
    parser.add_argument("--model-key", default="qwen3_8b")
    parser.add_argument("--top-features", type=int, default=32)
    parser.add_argument("--alphas", default="-2,-1,0,1,2")
    parser.add_argument("--limit", type=int, default=0, help="Maximum stories; 0 evaluates the full held-out cohort")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    model, tokenizer, device, spec = load_model(args.model_key, dtype="bfloat16")
    layers = resolve_layers(model, spec); sae, payload = load_sae(str(args.checkpoint), "cpu"); layer = int(payload["config"]["layer"])
    stats = pd.read_csv(args.feature_statistics).sort_values("older_minus_younger")
    younger_ids = stats.head(args.top_features)["feature"].astype(int).to_list(); older_ids = stats.tail(args.top_features)["feature"].astype(int).to_list()
    decoder = sae.decoder.weight.detach().float()
    younger = decoder[:, younger_ids].mean(dim=1); older = decoder[:, older_ids].mean(dim=1)
    age_direction = (older - younger) * float(payload["scale"])
    generator = torch.Generator().manual_seed(args.seed); random_direction = torch.randn(age_direction.shape, generator=generator); random_direction *= age_direction.norm() / random_direction.norm().clamp_min(1e-8)
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
        (str(row.get("story_key")), str(row.get("direction")), float(row.get("alpha")))
        for row in read_jsonl(args.output)
    } if args.output.exists() else set()
    for index, original in enumerate(originals, start=1):
        target_words = max(20, round(original["word_count"] * 0.30)); prompt = render_prompt(tokenizer, strict_summary_prompt(original["text"], target_words=target_words, target_age=original.get("source_age_range", ""), condition="actual"))
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192); encoded = {key: value.to(device) for key, value in encoded.items()}
        for direction_name, direction in [("sae_age", age_direction), ("random_matched", random_direction)]:
            for alpha in [float(value) for value in args.alphas.split(",") if value.strip()]:
                if (str(original["story_key"]), direction_name, alpha) in completed:
                    continue
                handle = layers[layer].register_forward_hook(last_token_addition_hook(direction, alpha))
                with torch.inference_mode(): generated = model.generate(**encoded, max_new_tokens=max(32, target_words * 2), do_sample=False, pad_token_id=tokenizer.pad_token_id)
                handle.remove(); text = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                append_jsonl(args.output, [{"story_key": original["story_key"], "layer": layer, "direction": direction_name, "alpha": alpha, "target_words": target_words, "summary": text, "evaluation_split": True, "seed": args.seed}])
        print(f"[{index}/{len(originals)}] {original['story_key']}", flush=True)


if __name__ == "__main__":
    main()
