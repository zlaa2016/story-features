from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from mechanistic_interpretability.shared.io import append_jsonl, read_jsonl
from mechanistic_interpretability.shared.modeling import last_token_addition_hook, load_model, render_prompt, resolve_layers
from mechanistic_interpretability.shared.prompts import strict_summary_prompt
from mechanistic_interpretability.shared.research import evaluation_story_keys, select_work_shard


def directional_outputs(baseline_logits, negative_logits, positive_logits, epsilon: float):
    import torch

    baseline_probabilities = torch.softmax(baseline_logits, dim=-1)
    negative_probabilities = torch.softmax(negative_logits, dim=-1)
    positive_probabilities = torch.softmax(positive_logits, dim=-1)
    logit_derivative = (positive_logits - negative_logits) / (2.0 * epsilon)
    probability_derivative = (positive_probabilities - negative_probabilities) / (2.0 * epsilon)
    forward = (positive_logits - baseline_logits) / epsilon
    backward = (baseline_logits - negative_logits) / epsilon
    return {
        "baseline_probabilities": baseline_probabilities,
        "logit_derivative": logit_derivative,
        "probability_derivative": probability_derivative,
        "logit_derivative_l2": float(torch.linalg.vector_norm(logit_derivative).cpu()),
        "probability_derivative_l1": float(torch.linalg.vector_norm(probability_derivative, ord=1).cpu()),
        "probability_derivative_l2": float(torch.linalg.vector_norm(probability_derivative).cpu()),
        "directional_fisher": float(
            torch.sum(probability_derivative.square() / baseline_probabilities.clamp_min(1e-8)).cpu()
        ),
        "finite_difference_asymmetry": float(
            (torch.linalg.vector_norm(forward - backward) / torch.linalg.vector_norm(logit_derivative).clamp_min(1e-8)).cpu()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Central finite-difference Jacobian-vector products for next-token logits and probabilities.")
    parser.add_argument("--cohort", type=Path, default=Path("mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl"))
    parser.add_argument("--directions", type=Path, default=Path("mechanistic_interpretability/outputs/04_layerwise_probes/age_directions.npz"))
    parser.add_argument("--output", type=Path, default=Path("mechanistic_interpretability/outputs/09_directional_jacobian/results.jsonl"))
    parser.add_argument("--story-split", type=Path, default=Path("mechanistic_interpretability/outputs/04_layerwise_probes/story_split.csv"))
    parser.add_argument("--model-key", default="gpt_oss_20b", choices=["qwen3_8b", "gpt_oss_20b", "qwen3_30b_a3b"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--direction-name", default="older_minus_actual")
    parser.add_argument("--layers", default="8,12,16,20")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Maximum stories; 0 evaluates the full held-out cohort")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    args = parser.parse_args()

    import torch

    directions = np.load(args.directions, allow_pickle=False)
    saved_layers = directions["layers"].astype(int).tolist()
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
        (str(row.get("story_key")), int(row.get("layer", -1)), str(row.get("direction_name")), str(row.get("control")))
        for row in read_jsonl(args.output)
    } if args.output.exists() else set()
    model, tokenizer, device, spec = load_model(args.model_key, model_name=args.model_name, dtype=args.dtype)
    model_layers = resolve_layers(model, spec)

    for story_index, original in enumerate(originals, start=1):
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
        with torch.inference_mode():
            baseline_logits = model(**encoded, use_cache=False).logits[0, -1, :].float()
        for layer_index in [int(value) for value in args.layers.split(",") if value.strip()]:
            if layer_index not in saved_layers:
                continue
            direction = torch.from_numpy(
                directions[args.direction_name][saved_layers.index(layer_index)].astype(np.float32)
            )
            stable_seed = int.from_bytes(
                hashlib.sha256(f"{args.seed}:{original['story_key']}:{layer_index}".encode()).digest()[:8],
                "big",
            ) % (2**63 - 1)
            generator = torch.Generator(device=device).manual_seed(stable_seed)
            random_direction = torch.randn(direction.shape, generator=generator, device=device)
            direction_device = direction.to(device)
            projection = torch.dot(random_direction, direction_device) / direction_device.pow(2).sum().clamp_min(1e-12)
            random_direction = random_direction - projection * direction_device
            random_direction = random_direction / random_direction.norm().clamp_min(1e-12) * direction_device.norm()
            for control, vector in [("age_direction", direction), ("random_orthogonal", random_direction.cpu())]:
                key = (str(original["story_key"]), layer_index, args.direction_name, control)
                if key in completed:
                    continue
                perturbed = []
                for signed_epsilon in (-args.epsilon, args.epsilon):
                    handle = model_layers[layer_index].register_forward_hook(
                        last_token_addition_hook(vector, signed_epsilon)
                    )
                    with torch.inference_mode():
                        perturbed.append(model(**encoded, use_cache=False).logits[0, -1, :].float())
                    handle.remove()
                outputs = directional_outputs(baseline_logits, perturbed[0], perturbed[1], args.epsilon)
                logit_derivative = outputs.pop("logit_derivative")
                probability_derivative = outputs.pop("probability_derivative")
                baseline_probabilities = outputs.pop("baseline_probabilities")
                positive_values, positive_ids = torch.topk(logit_derivative, args.top_k)
                negative_values, negative_ids = torch.topk(-logit_derivative, args.top_k)
                probability_values, probability_ids = torch.topk(probability_derivative.abs(), args.top_k)
                baseline_values, baseline_ids = torch.topk(baseline_probabilities, args.top_k)
                append_jsonl(
                    args.output,
                    [{
                        "story_key": original["story_key"],
                        "layer": layer_index,
                        "direction_name": args.direction_name,
                        "control": control,
                        "control_seed": stable_seed,
                        "epsilon": args.epsilon,
                        "baseline_top_tokens": tokenizer.convert_ids_to_tokens(baseline_ids.cpu().tolist()),
                        "baseline_top_probabilities": baseline_values.cpu().tolist(),
                        "positive_tokens": tokenizer.convert_ids_to_tokens(positive_ids.cpu().tolist()),
                        "positive_logit_derivatives": positive_values.cpu().tolist(),
                        "negative_tokens": tokenizer.convert_ids_to_tokens(negative_ids.cpu().tolist()),
                        "negative_logit_derivatives": (-negative_values).cpu().tolist(),
                        "largest_probability_change_tokens": tokenizer.convert_ids_to_tokens(probability_ids.cpu().tolist()),
                        "largest_probability_derivatives": probability_derivative[probability_ids].cpu().tolist(),
                        "largest_probability_derivative_magnitudes": probability_values.cpu().tolist(),
                        **outputs,
                        "evaluation_split": True,
                    }],
                )
        print(f"[{story_index}/{len(originals)}] {original['story_key']}", flush=True)


if __name__ == "__main__":
    main()
