#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    family: str
    variant: str
    context_length: int
    hidden_size: int
    dtype: str = "bfloat16"
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


# Long-context causal-LM registry for whole-story perplexity. Quantized variants
# are intentionally omitted because they failed in the related unsteerable runs.
QWEN3_YARN_128K = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}

MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "qwen3_8b_base",
        "Qwen/Qwen3-8B-Base",
        "qwen",
        "base",
        context_length=131072,
        hidden_size=4096,
        max_position_embeddings=131072,
        rope_scaling=QWEN3_YARN_128K,
        notes="Qwen3 dense; native 32K, validated to 128K with YaRN.",
    ),
    ModelSpec(
        "qwen3_8b",
        "Qwen/Qwen3-8B",
        "qwen",
        "posttrained",
        context_length=131072,
        hidden_size=4096,
        max_position_embeddings=131072,
        rope_scaling=QWEN3_YARN_128K,
        notes="Qwen3 dense; native 32K, validated to 128K with YaRN.",
    ),
    ModelSpec(
        "qwen3_14b_base",
        "Qwen/Qwen3-14B-Base",
        "qwen",
        "base",
        context_length=131072,
        hidden_size=5120,
        max_position_embeddings=131072,
        rope_scaling=QWEN3_YARN_128K,
        notes="Qwen3 dense; native 32K, validated to 128K with YaRN.",
    ),
    ModelSpec(
        "qwen3_14b",
        "Qwen/Qwen3-14B",
        "qwen",
        "posttrained",
        context_length=131072,
        hidden_size=5120,
        max_position_embeddings=131072,
        rope_scaling=QWEN3_YARN_128K,
        notes="Qwen3 dense; native 32K, validated to 128K with YaRN.",
    ),
    ModelSpec(
        "qwen3_30b_a3b_base",
        "Qwen/Qwen3-30B-A3B-Base",
        "qwen",
        "base_moe",
        context_length=131072,
        hidden_size=2048,
        max_position_embeddings=131072,
        rope_scaling=QWEN3_YARN_128K,
        notes="Qwen3 MoE; 30.5B total, 3.3B activated; native 32K, validated to 128K with YaRN.",
    ),
    ModelSpec(
        "qwen3_30b_a3b_instruct_2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "qwen",
        "instruct_moe",
        context_length=262144,
        hidden_size=2048,
        notes="Qwen3 MoE 2507 refresh; 30.5B total, 3.3B activated; native 256K.",
    ),
    ModelSpec(
        "qwen3_32b",
        "Qwen/Qwen3-32B",
        "qwen",
        "posttrained",
        context_length=131072,
        hidden_size=5120,
        max_position_embeddings=131072,
        rope_scaling=QWEN3_YARN_128K,
        notes="Qwen3 dense; native 32K, validated to 128K with YaRN.",
    ),
    ModelSpec(
        "gemma_3_1b_pt",
        "google/gemma-3-1b-pt",
        "gemma",
        "base",
        context_length=32768,
        hidden_size=1152,
        notes="Gemma 3 text-only 1B; larger Gemma 3 checkpoints are multimodal and need a separate scorer.",
    ),
    ModelSpec(
        "gemma_3_1b_it",
        "google/gemma-3-1b-it",
        "gemma",
        "instruct",
        context_length=32768,
        hidden_size=1152,
        notes="Gemma 3 text-only 1B; larger Gemma 3 checkpoints are multimodal and need a separate scorer.",
    ),
    ModelSpec("olmo_2_1b_base", "allenai/OLMo-2-0425-1B", "olmo", "base", context_length=4096, hidden_size=2048),
    ModelSpec("olmo_2_1b_instruct", "allenai/OLMo-2-0425-1B-Instruct", "olmo", "instruct", context_length=4096, hidden_size=2048),
    ModelSpec("olmo_2_7b_base", "allenai/OLMo-2-1124-7B", "olmo", "base", context_length=4096, hidden_size=4096),
    ModelSpec("olmo_2_7b_instruct", "allenai/OLMo-2-1124-7B-Instruct", "olmo", "instruct", context_length=4096, hidden_size=4096),
    ModelSpec("olmo_2_13b_base", "allenai/OLMo-2-1124-13B", "olmo", "base", context_length=4096, hidden_size=5120),
    ModelSpec("olmo_2_13b_instruct", "allenai/OLMo-2-1124-13B-Instruct", "olmo", "instruct", context_length=4096, hidden_size=5120),
    ModelSpec("olmo_2_32b_base", "allenai/OLMo-2-0325-32B", "olmo", "base", context_length=4096, hidden_size=5120),
    ModelSpec("olmo_2_32b_instruct", "allenai/OLMo-2-0325-32B-Instruct", "olmo", "instruct", context_length=4096, hidden_size=5120),
)


def all_specs() -> tuple[ModelSpec, ...]:
    return MODEL_SPECS


def get_model_spec(key: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.key == key:
            return spec
    keys = ", ".join(spec.key for spec in MODEL_SPECS)
    raise KeyError(f"Unknown model key: {key}. Valid keys: {keys}")


def iter_keys(families: Optional[set[str]] = None, min_context_length: int = 0, min_hidden_size: int = 0) -> Iterable[str]:
    for spec in MODEL_SPECS:
        if families and spec.family not in families:
            continue
        if spec.context_length < min_context_length:
            continue
        if spec.hidden_size < min_hidden_size:
            continue
        yield spec.key


def main() -> None:
    ap = argparse.ArgumentParser(description="List or index the story perplexity model registry.")
    ap.add_argument("--list-keys", action="store_true", help="Print one model key per line.")
    ap.add_argument("--count", action="store_true", help="Print number of registered models.")
    ap.add_argument("--key-at-index", type=int, default=None, help="Print model key at zero-based index.")
    ap.add_argument("--family", action="append", default=None, help="Optional family filter; repeatable.")
    ap.add_argument("--min-context-length", type=int, default=0, help="Only list models with at least this many tokens.")
    ap.add_argument("--min-hidden-size", type=int, default=0, help="Only list models with at least this hidden size.")
    ap.add_argument("--tsv", action="store_true", help="Print key/model/family/variant table.")
    args = ap.parse_args()

    families = set(args.family) if args.family else None
    keys = list(iter_keys(families=families, min_context_length=args.min_context_length, min_hidden_size=args.min_hidden_size))
    specs = [get_model_spec(key) for key in keys]

    if args.count:
        print(len(keys))
        return
    if args.key_at_index is not None:
        if args.key_at_index < 0 or args.key_at_index >= len(keys):
            raise SystemExit(f"Index out of range: {args.key_at_index}; registry has {len(keys)} models")
        print(keys[args.key_at_index])
        return
    if args.tsv:
        print("key\tmodel_name\tfamily\tvariant\tcontext_length\thidden_size\tnotes")
        for spec in specs:
            print(
                f"{spec.key}\t{spec.model_name}\t{spec.family}\t{spec.variant}\t"
                f"{spec.context_length}\t{spec.hidden_size}\t{spec.notes}"
            )
        return
    if args.list_keys or not any([args.count, args.tsv]):
        for key in keys:
            print(key)


if __name__ == "__main__":
    main()
