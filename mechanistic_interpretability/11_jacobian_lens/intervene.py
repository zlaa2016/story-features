from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from mechanistic_interpretability.shared.io import append_jsonl, read_jsonl
from mechanistic_interpretability.shared.modeling import layer_output_tensor, render_prompt, replace_layer_output
from mechanistic_interpretability.shared.prompts import strict_summary_prompt
from mechanistic_interpretability.shared.research import evaluation_story_keys, select_work_shard


def swap_hook(source: Any, target: Any):
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = layer_output_tensor(output)
        source_local = source.to(hidden.device, hidden.dtype)
        target_local = target.to(hidden.device, hidden.dtype)
        coefficient = hidden.float() @ source_local.float()
        modified = hidden + coefficient.to(hidden.dtype).unsqueeze(-1) * (target_local - source_local)
        return replace_layer_output(output, modified)
    return hook


def main() -> None:
    parser = argparse.ArgumentParser(description="Test causal J-lens coordinate swaps during summary generation.")
    parser.add_argument("--cohort", type=Path, default=Path("mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl"))
    parser.add_argument("--lens", type=Path, default=Path("mechanistic_interpretability/outputs/11_jacobian_lens/jacobian_lens.pt"))
    parser.add_argument("--token-effects", type=Path, default=Path("mechanistic_interpretability/outputs/11_jacobian_lens/analysis/jspace_token_condition_effects.csv"))
    parser.add_argument("--story-split", type=Path, default=Path("mechanistic_interpretability/outputs/04_layerwise_probes/story_split.csv"))
    parser.add_argument("--jlens-repo", type=Path, default=Path(os.environ.get("JLENS_REPO", f"/scratch/{os.environ.get('USER', 'user')}/tools/jacobian-lens")))
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0, help="Maximum stories; 0 evaluates the full held-out cohort")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("mechanistic_interpretability/outputs/11_jacobian_lens/coordinate_swaps.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(args.jlens_repo))
    import jlens

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    model = jlens.from_hf(hf_model, tokenizer); lens = jlens.JacobianLens.load(str(args.lens))
    effects = pd.read_csv(args.token_effects); effects = effects[effects["layer"] == args.layer].sort_values("older_minus_younger")
    if len(effects) < 2:
        raise ValueError(f"No younger/older token effects for layer {args.layer}")
    younger_token = int(effects.iloc[0]["token_id"]); older_token = int(effects.iloc[-1]["token_id"])
    unembedding = hf_model.get_output_embeddings().weight.detach().float().cpu()
    jacobian = lens.jacobians[args.layer].float()
    younger_direction = jacobian.T @ unembedding[younger_token]; younger_direction /= younger_direction.norm().clamp_min(1e-8)
    older_direction = jacobian.T @ unembedding[older_token]; older_direction /= older_direction.norm().clamp_min(1e-8)
    generator = torch.Generator().manual_seed(args.seed)
    random_source = torch.randn(younger_direction.shape, generator=generator); random_source /= random_source.norm().clamp_min(1e-8)
    random_target = torch.randn(younger_direction.shape, generator=generator); random_target -= (random_target @ random_source) * random_source; random_target /= random_target.norm().clamp_min(1e-8)
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
    completed = {(row.get("story_key"), row.get("intervention")) for row in read_jsonl(args.output)} if args.output.exists() else set()
    input_device = hf_model.get_input_embeddings().weight.device
    for index, original in enumerate(originals, start=1):
        target_words = max(20, round(original["word_count"] * 0.30)); prompt = render_prompt(tokenizer, strict_summary_prompt(original["text"], target_words=target_words, target_age=original.get("source_age_range", ""), condition="actual"))
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192); encoded = {key: value.to(input_device) for key, value in encoded.items()}
        for name, source, target in [("clean", None, None), ("younger_to_older", younger_direction, older_direction), ("older_to_younger", older_direction, younger_direction), ("random_matched", random_source, random_target)]:
            if (original["story_key"], name) in completed: continue
            handle = None if source is None else model.layers[args.layer].register_forward_hook(swap_hook(source, target))
            with torch.inference_mode(): generated = hf_model.generate(**encoded, max_new_tokens=min(args.max_new_tokens, max(32, target_words * 2)), do_sample=False, pad_token_id=tokenizer.pad_token_id)
            if handle is not None: handle.remove()
            summary = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            append_jsonl(args.output, [{"story_key": original["story_key"], "intervention": name, "layer": args.layer, "younger_token_id": younger_token, "younger_token": tokenizer.convert_ids_to_tokens(younger_token), "older_token_id": older_token, "older_token": tokenizer.convert_ids_to_tokens(older_token), "summary": summary, "evaluation_split": True, "seed": args.seed}])
        print(f"[{index}/{len(originals)}] {original['story_key']}", flush=True)


if __name__ == "__main__":
    main()
