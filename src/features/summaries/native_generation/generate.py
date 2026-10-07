from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from shared.ages import AgeCondition, relative_conditions
from shared.io import append_jsonl, read_jsonl, word_count
from shared.modeling import load_model, render_prompt
from shared.prompts import strict_summary_prompt


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    return {
        (str(row.get("story_key")), str(row.get("condition")), int(row.get("summary_id", 1)))
        for row in read_jsonl(path)
        if row.get("status") in {"success", "ineligible"}
    }


def clean_generation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    if "<|channel|>final<|message|>" in text:
        text = text.split("<|channel|>final<|message|>", 1)[-1]
    return text.strip()


def condition_rows(
    original: dict[str, Any],
    minimum_age: int,
    maximum_age: int,
    include_no_age: bool,
    include_shuffled: bool,
    rng: random.Random,
) -> list[AgeCondition]:
    rows = relative_conditions(
        original.get("source_age_range"),
        minimum_age=minimum_age,
        maximum_age=maximum_age,
    )
    if include_no_age:
        rows.append(AgeCondition("no_age", True, "", None, ""))
    if include_shuffled:
        age = rng.randint(minimum_age, maximum_age)
        rows.append(AgeCondition("shuffled", True, str(age), float(age), ""))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate budget-matched summaries with an observable open-weight model.")
    parser.add_argument("--cohort", type=Path, default=Path("results/features/summaries/dataset_index/cohort.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/features/summaries/native_generation/generations.jsonl"))
    parser.add_argument("--model-key", default="gpt_oss_20b", choices=["qwen3_8b", "gpt_oss_20b", "qwen3_30b_a3b"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n-summaries", type=int, default=1)
    parser.add_argument("--compression-ratio", type=float, default=0.30)
    parser.add_argument("--minimum-age", type=int, default=2)
    parser.add_argument("--maximum-age", type=int, default=18)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--include-no-age", action="store_true")
    parser.add_argument("--include-shuffled", action="store_true")
    parser.add_argument(
        "--available-conditions-only",
        action="store_true",
        help="Generate only story-condition pairs represented by existing GPT summaries in the cohort",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compression-tolerance", type=float, default=0.10, help="Maximum relative word-count deviation from the requested target")
    args = parser.parse_args()

    import torch

    cohort_rows = list(read_jsonl(args.cohort))
    originals = [row for row in cohort_rows if row.get("record_type") == "original"]
    available_conditions = {
        (str(row.get("story_key")), str(row.get("condition")))
        for row in cohort_rows
        if row.get("record_type") == "summary"
    }
    reference_target_words = {
        (str(row.get("story_key")), str(row.get("condition"))): int(row.get("word_count", 0))
        for row in cohort_rows
        if row.get("record_type") == "summary" and int(row.get("word_count", 0)) > 0
    }
    stop = None if args.limit <= 0 else args.start + args.limit
    originals = originals[args.start : stop]
    done = completed_keys(args.output)
    model, tokenizer, device, spec = load_model(args.model_key, model_name=args.model_name, dtype=args.dtype)
    rng = random.Random(args.seed)

    for story_index, original in enumerate(originals, start=1):
        for condition in condition_rows(
            original,
            args.minimum_age,
            args.maximum_age,
            args.include_no_age,
            args.include_shuffled,
            rng,
        ):
            if args.available_conditions_only and (str(original["story_key"]), condition.name) not in available_conditions:
                continue
            reference_words = reference_target_words.get((str(original["story_key"]), condition.name))
            target_words = reference_words or max(20, round(int(original["word_count"]) * args.compression_ratio))
            if not condition.eligible:
                key = (original["story_key"], condition.name, 0)
                if key not in done:
                    append_jsonl(
                        args.output,
                        [{
                            "story_key": original["story_key"],
                            "video_id": original.get("video_id", ""),
                            "condition": condition.name,
                            "condition_eligible": False,
                            "condition_skip_reason": condition.skip_reason,
                            "summary_id": 0,
                            "status": "ineligible",
                        }],
                    )
                continue
            for summary_id in range(1, args.n_summaries + 1):
                key = (original["story_key"], condition.name, summary_id)
                if key in done:
                    continue
                prompt_text = strict_summary_prompt(
                    original["text"],
                    target_words=target_words,
                    target_age=condition.target_spec,
                    condition=condition.name,
                )
                prompt = render_prompt(tokenizer, prompt_text, reasoning=args.reasoning)
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
                encoded = {name: value.to(device) for name, value in encoded.items()}
                generation_kwargs: dict[str, Any] = {
                    **encoded,
                    "max_new_tokens": args.max_new_tokens,
                    "pad_token_id": tokenizer.eos_token_id,
                    "use_cache": True,
                }
                if args.temperature > 0:
                    generation_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
                else:
                    generation_kwargs.update(do_sample=False)
                try:
                    generation_seed = int.from_bytes(
                        hashlib.sha256(f"{args.seed}::{original['story_key']}::{condition.name}::{summary_id}".encode()).digest()[:8],
                        "big",
                    ) % (2**63 - 1)
                    torch.manual_seed(generation_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(generation_seed)
                    with torch.inference_mode():
                        output_ids = model.generate(**generation_kwargs)
                    new_ids = output_ids[0, encoded["input_ids"].shape[1] :]
                    summary = clean_generation(tokenizer.decode(new_ids, skip_special_tokens=True))
                    achieved_words = word_count(summary)
                    record = {
                        "story_key": original["story_key"],
                        "video_id": original.get("video_id", ""),
                        "condition": condition.name,
                        "condition_eligible": True,
                        "condition_skip_reason": "",
                        "summary_id": summary_id,
                        "source_age_range": original.get("source_age_range", ""),
                        "source_age": original.get("source_age"),
                        "target_age_range": condition.target_spec,
                        "target_age": condition.target_age,
                        "generator_model": args.model_name or spec.name,
                        "requested_compression_ratio": target_words / max(1, int(original["word_count"])),
                        "fallback_compression_ratio": args.compression_ratio,
                        "target_budget_source": "matched_gpt_summary" if reference_words else "fixed_ratio",
                        "requested_target_words": target_words,
                        "achieved_words": achieved_words,
                        "achieved_compression_ratio": achieved_words / max(1, int(original["word_count"])),
                        "relative_target_word_error": abs(achieved_words - target_words) / max(target_words, 1),
                        "compression_within_tolerance": abs(achieved_words - target_words) / max(target_words, 1) <= args.compression_tolerance,
                        "summary_text": summary,
                        "status": "success" if summary else "empty",
                        "seed": generation_seed,
                    }
                except Exception as error:
                    record = {
                        "story_key": original["story_key"],
                        "video_id": original.get("video_id", ""),
                        "condition": condition.name,
                        "condition_eligible": True,
                        "summary_id": summary_id,
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                append_jsonl(args.output, [record])
                print(
                    f"[{story_index}/{len(originals)}] {original['story_key']} {condition.name} summary={summary_id} status={record['status']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
