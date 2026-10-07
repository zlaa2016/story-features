#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from generate_prompt_condition_summaries import build_prompt, load_story_rows


@dataclass(frozen=True)
class SummaryModelSpec:
    key: str
    model_name: str
    context_length: int
    hidden_size: int
    trust_remote_code: bool = False
    max_position_embeddings: Optional[int] = None
    rope_scaling: Optional[dict[str, Any]] = None
    notes: str = ""

    def model_config_overrides(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.max_position_embeddings is not None:
            out["max_position_embeddings"] = self.max_position_embeddings
        if self.rope_scaling is not None:
            out["rope_scaling"] = self.rope_scaling
        return out


QWEN3_YARN_128K = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}


SUMMARY_MODEL_SPECS: dict[str, SummaryModelSpec] = {
    "gpt_oss_20b": SummaryModelSpec(
        "gpt_oss_20b",
        "openai/gpt-oss-20b",
        context_length=131072,
        hidden_size=2880,
        notes="OpenAI open-weight GPT-style model; use Harmony/chat template.",
    ),
    "gpt_oss_120b": SummaryModelSpec(
        "gpt_oss_120b",
        "openai/gpt-oss-120b",
        context_length=131072,
        hidden_size=2880,
        notes="OpenAI open-weight GPT-style model; requires H100/80GB-class hardware.",
    ),
    "qwen3_8b": SummaryModelSpec(
        "qwen3_8b",
        "Qwen/Qwen3-8B",
        context_length=131072,
        hidden_size=4096,
        max_position_embeddings=131072,
        rope_scaling=QWEN3_YARN_128K,
        notes="Qwen3 dense; use YaRN/Transformers 4.51+ for 128K contexts.",
    ),
    "qwen3_14b": SummaryModelSpec(
        "qwen3_14b",
        "Qwen/Qwen3-14B",
        context_length=131072,
        hidden_size=5120,
        max_position_embeddings=131072,
        rope_scaling=QWEN3_YARN_128K,
        notes="Qwen3 dense; use YaRN/Transformers 4.51+ for 128K contexts.",
    ),
    "qwen3_30b_a3b_instruct_2507": SummaryModelSpec(
        "qwen3_30b_a3b_instruct_2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        context_length=262144,
        hidden_size=2048,
        notes="Qwen3 MoE 2507 refresh; 30.5B total, 3.3B activated; native 256K.",
    ),
    "qwen3_32b": SummaryModelSpec(
        "qwen3_32b",
        "Qwen/Qwen3-32B",
        context_length=131072,
        hidden_size=5120,
        max_position_embeddings=131072,
        rope_scaling=QWEN3_YARN_128K,
        notes="Qwen3 dense; use YaRN/Transformers 4.51+ for 128K contexts.",
    ),
    "gemma_3_1b_it": SummaryModelSpec(
        "gemma_3_1b_it",
        "google/gemma-3-1b-it",
        context_length=32768,
        hidden_size=1152,
        notes="Gemma 3 text-only 1B; larger Gemma 3 checkpoints are multimodal.",
    ),
    "olmo_2_7b_instruct": SummaryModelSpec(
        "olmo_2_7b_instruct",
        "allenai/OLMo-2-1124-7B-Instruct",
        context_length=4096,
        hidden_size=4096,
        notes="OLMo 2 paper-family control; many long stories may be skipped.",
    ),
    "olmo_2_13b_instruct": SummaryModelSpec(
        "olmo_2_13b_instruct",
        "allenai/OLMo-2-1124-13B-Instruct",
        context_length=4096,
        hidden_size=5120,
        notes="OLMo 2 paper-family control; many long stories may be skipped.",
    ),
    "olmo_2_32b_instruct": SummaryModelSpec(
        "olmo_2_32b_instruct",
        "allenai/OLMo-2-0325-32B-Instruct",
        context_length=4096,
        hidden_size=5120,
        notes="OLMo 2 paper-family control; many long stories may be skipped.",
    ),
}


def model_keys() -> list[str]:
    return sorted(SUMMARY_MODEL_SPECS)


def resolve_dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def load_model_and_tokenizer(
    model_name: str,
    dtype_name: str,
    device_map: str,
    trust_remote_code: bool,
    attn_implementation: Optional[str],
    config_overrides: Optional[dict[str, Any]] = None,
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(torch, dtype_name)
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }
    if config_overrides:
        kwargs.update(config_overrides)
    if dtype == "auto":
        kwargs["torch_dtype"] = "auto"
    else:
        kwargs["torch_dtype"] = dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    try:
        input_device = model.get_input_embeddings().weight.device
    except Exception:
        input_device = next(model.parameters()).device
    return model, tokenizer, input_device


def render_chat_prompt(tokenizer: Optional[Any], user_prompt: str, model_key: str, reasoning: str) -> str:
    system = "You are a careful story summarization assistant. Return only valid JSON."
    if model_key.startswith("gpt_oss"):
        system = f"Reasoning: {reasoning}\n{system}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system}\n\nUser:\n{user_prompt}\n\nAssistant:"


def extract_json_object(text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    cleaned = text.strip()
    try:
        value = json.loads(cleaned)
        return value, None if isinstance(value, dict) else "JSON root is not an object"
    except Exception as exc:
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first >= 0 and last > first:
            try:
                value = json.loads(cleaned[first : last + 1])
                return value, None if isinstance(value, dict) else "JSON root is not an object"
            except Exception as inner_exc:
                return None, f"{type(inner_exc).__name__}: {inner_exc}"
        return None, f"{type(exc).__name__}: {exc}"


def generate_text(
    model: Any,
    tokenizer: Any,
    input_device: Any,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, int]:
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(input_device)
    attention_mask = enc["attention_mask"].to(input_device)
    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        gen_kwargs.update({"do_sample": False})

    outputs = model.generate(**gen_kwargs)
    new_tokens = outputs[0, input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), int(input_ids.shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate age/no-age summaries with a local long-context open-weight model.")
    parser.add_argument("--zip-path", default="cleaned_idchecked_punctuated.zip")
    parser.add_argument("--master-path", default="data/processed/master_stories.parquet")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--model-key", default=os.environ.get("OPEN_WEIGHT_SUMMARY_MODEL_KEY", "gpt_oss_20b"), choices=model_keys())
    parser.add_argument("--model-name", default=None, help="Override the registered Hugging Face model name.")
    parser.add_argument("--dtype", default=os.environ.get("SUMMARY_DTYPE", "bfloat16"), choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--device-map", default=os.environ.get("SUMMARY_DEVICE_MAP", "auto"))
    parser.add_argument("--attn-implementation", default=os.environ.get("SUMMARY_ATTN_IMPLEMENTATION", ""))
    parser.add_argument("--max-context-tokens", type=int, default=0, help="0 uses the model registry context length.")
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reasoning", default="low", choices=["low", "medium", "high"], help="Used in the gpt-oss system prompt.")
    parser.add_argument("--conditions", nargs="+", default=["no_age", "age"], choices=["no_age", "age"])
    parser.add_argument("--overlength-policy", default="skip", choices=["skip", "error"])
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        print("key\tmodel_name\tcontext_length\thidden_size\tnotes")
        for key in model_keys():
            spec = SUMMARY_MODEL_SPECS[key]
            print(f"{spec.key}\t{spec.model_name}\t{spec.context_length}\t{spec.hidden_size}\t{spec.notes}")
        return

    spec = SUMMARY_MODEL_SPECS[args.model_key]
    model_name = args.model_name or spec.model_name
    max_context_tokens = args.max_context_tokens or spec.context_length
    max_input_tokens = max_context_tokens - args.max_new_tokens
    if max_input_tokens <= 0:
        raise SystemExit("--max-new-tokens must be smaller than the model context window")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    pipeline_dir = Path("results/features/summaries/runs") / f"summary_open_weight_{args.model_key}_{safe_model_name}_{run_timestamp}"
    prompt_dir = pipeline_dir / "prompts"
    generation_dir = pipeline_dir / "generations"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    generation_dir.mkdir(exist_ok=True)
    out_jsonl = pipeline_dir / "summary_outputs.jsonl"

    rows = load_story_rows(Path(args.zip_path), Path(args.master_path))
    rows = rows.iloc[args.start : args.start + args.limit].copy()
    print(f"Output directory: {pipeline_dir}")
    print(f"Loaded {len(rows)} stories for this run")
    print(f"Model: {model_name} key={args.model_key} context={max_context_tokens} hidden={spec.hidden_size}")

    model = tokenizer = input_device = None
    if not args.dry_run:
        model, tokenizer, input_device = load_model_and_tokenizer(
            model_name=model_name,
            dtype_name=args.dtype,
            device_map=args.device_map,
            trust_remote_code=spec.trust_remote_code,
            attn_implementation=args.attn_implementation or None,
            config_overrides=spec.model_config_overrides(),
        )
    else:
        tokenizer = None

    with out_jsonl.open("w", encoding="utf-8") as out:
        for _, row in rows.iterrows():
            story_id = str(row["story_id"])
            age_range = "" if pd.isna(row.get("age_range")) else str(row.get("age_range", ""))
            story_text = str(row["story_text"]).strip()

            for condition in args.conditions:
                prompt = build_prompt(condition, story_text, age_range)
                rendered_prompt = render_chat_prompt(tokenizer, prompt, args.model_key, args.reasoning)
                prompt_tokens: Optional[int] = None
                fits_context: Optional[bool] = None
                if tokenizer is not None:
                    enc = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)
                    prompt_tokens = int(enc["input_ids"].shape[-1])
                    fits_context = prompt_tokens <= max_input_tokens
                prompt_path = prompt_dir / f"{story_id}_{condition}.txt"
                prompt_path.write_text(rendered_prompt, encoding="utf-8")

                record: dict[str, Any] = {
                    "story_id": story_id,
                    "condition": condition,
                    "provider": "hf_local",
                    "model": model_name,
                    "model_key": args.model_key,
                    "age_range": age_range,
                    "prompt_path": str(prompt_path),
                    "prompt_tokens": prompt_tokens,
                    "max_context_tokens": max_context_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "fits_context": fits_context,
                    "response": None,
                    "parse_error": None,
                    "raw_generation_path": None,
                }

                if prompt_tokens is not None and prompt_tokens > max_input_tokens:
                    message = (
                        f"prompt has {prompt_tokens} tokens; max input tokens are {max_input_tokens} "
                        f"after reserving {args.max_new_tokens} generation tokens"
                    )
                    if args.overlength_policy == "error":
                        raise ValueError(message)
                    record["parse_error"] = f"skipped_overlength: {message}"
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"skipped {story_id} / {condition}: overlength")
                    continue

                if not args.dry_run:
                    assert model is not None and input_device is not None
                    raw_text, input_tokens = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        input_device=input_device,
                        prompt_text=rendered_prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    raw_path = generation_dir / f"{story_id}_{condition}.txt"
                    raw_path.write_text(raw_text, encoding="utf-8")
                    parsed, parse_error = extract_json_object(raw_text)
                    record["prompt_tokens"] = input_tokens
                    record["raw_generation_path"] = str(raw_path)
                    record["response"] = parsed
                    record["parse_error"] = parse_error

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(f"wrote {story_id} / {condition}")
                if args.sleep and math.isfinite(args.sleep):
                    time.sleep(args.sleep)

    print(f"Results: {out_jsonl}")


if __name__ == "__main__":
    main()
