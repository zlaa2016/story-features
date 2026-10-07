#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from analysis.natural_language.perplexity.models import get_model_spec
    from analysis.natural_language.perplexity.table_io import read_table
except ModuleNotFoundError:
    from models import get_model_spec
    from table_io import read_table


FIELDNAMES = [
    "model_key",
    "model_name",
    "family",
    "variant",
    "quantization",
    "model_context_length",
    "model_hidden_size",
    "item_id",
    "story_id",
    "video_id",
    "reading_age",
    "age_lo",
    "age_hi",
    "age_median",
    "n_chars",
    "n_words",
    "n_tokens",
    "n_loss_tokens",
    "nll_sum",
    "mean_nll",
    "src/analysis/natural_language/perplexity",
    "log10_perplexity",
    "n_windows",
    "fits_single_window",
    "overlength_policy",
    "max_length",
    "stride",
]


def resolve_dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def load_tokenizer(model_name: str, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name: str,
    dtype_name: str,
    device_map: str,
    trust_remote_code: bool,
    config_overrides: Optional[dict[str, Any]] = None,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    dtype = resolve_dtype(torch, dtype_name)
    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if config_overrides:
        model_kwargs.update(config_overrides)

    if device_map != "none":
        model_kwargs["device_map"] = device_map

    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device_map == "none":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        try:
            device = model.get_input_embeddings().weight.device
        except Exception:
            device = next(model.parameters()).device
    model.eval()
    return model, device


def bounded_context_length(tokenizer: Any, model: Any, requested: int, registry_context_length: int = 0) -> int:
    if requested > 0:
        return requested
    if registry_context_length > 0:
        return registry_context_length
    candidates: list[int] = []
    tok_len = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_len, int) and 0 < tok_len < 1_000_000:
        candidates.append(tok_len)
    config = getattr(model, "config", None)
    for attr in ["max_position_embeddings", "n_positions", "seq_length"]:
        value = getattr(config, attr, None) if config is not None else None
        if isinstance(value, int) and value > 0:
            candidates.append(value)
    return min(candidates) if candidates else 2048


def score_text(
    text: str,
    tokenizer: Any,
    model: Any,
    input_device: Any,
    max_length: int,
    stride: int,
    overlength_policy: str,
) -> dict[str, float]:
    import torch

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids_all = enc["input_ids"]
    seq_len = int(input_ids_all.shape[1])
    if seq_len < 2:
        return {
            "n_tokens": float(seq_len),
            "n_loss_tokens": 0.0,
            "nll_sum": float("nan"),
            "mean_nll": float("nan"),
            "src/analysis/natural_language/perplexity": float("nan"),
            "log10_perplexity": float("nan"),
            "n_windows": 0.0,
            "fits_single_window": 1.0,
        }
    if seq_len > max_length and overlength_policy != "slide":
        if overlength_policy == "error":
            raise ValueError(f"Story has {seq_len} tokens, exceeding model window {max_length}.")
        if overlength_policy != "skip":
            raise ValueError(f"Unknown overlength policy: {overlength_policy}")
        return {
            "n_tokens": float(seq_len),
            "n_loss_tokens": 0.0,
            "nll_sum": float("nan"),
            "mean_nll": float("nan"),
            "src/analysis/natural_language/perplexity": float("nan"),
            "log10_perplexity": float("nan"),
            "n_windows": 0.0,
            "fits_single_window": 0.0,
        }

    nll_sum = 0.0
    n_loss_tokens = 0
    n_windows = 0
    prev_end = 0
    stride = min(stride, max_length)

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        input_ids = input_ids_all[:, begin:end].to(input_device)
        target_ids = input_ids.clone()
        n_windows += 1
        if trg_len < target_ids.shape[1]:
            target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=target_ids, use_cache=False)

        valid_labels = int((target_ids[:, 1:] != -100).sum().item())
        if valid_labels > 0:
            loss_value = float(outputs.loss.detach().cpu().item())
            nll_sum += loss_value * valid_labels
            n_loss_tokens += valid_labels

        prev_end = end
        if end == seq_len:
            break

    mean_nll = nll_sum / n_loss_tokens if n_loss_tokens else float("nan")
    ppl = math.exp(mean_nll) if math.isfinite(mean_nll) and mean_nll < 700 else float("inf")
    log10_ppl = mean_nll / math.log(10.0) if math.isfinite(mean_nll) else float("nan")
    return {
        "n_tokens": float(seq_len),
        "n_loss_tokens": float(n_loss_tokens),
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
        "src/analysis/natural_language/perplexity": ppl,
        "log10_perplexity": log10_ppl,
        "n_windows": float(n_windows),
        "fits_single_window": 1.0 if seq_len <= max_length else 0.0,
    }


def _str_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _float_or_empty(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return str(float(value))
    except Exception:
        return ""


def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        done = pd.read_csv(path, usecols=["item_id"], dtype={"item_id": str})
    except Exception:
        return set()
    return set(done["item_id"].dropna().astype(str))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute per-story causal-LM src/analysis/natural_language/perplexity.")
    ap.add_argument("--dataset", type=Path, default=Path("results/analysis/natural_language/perplexity/stories_for_perplexity.parquet"))
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--model-key", type=str, default="qwen3_8b")
    ap.add_argument("--model-name", type=str, default=None, help="Override registry model name.")
    ap.add_argument("--dtype", type=str, default=None, choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    ap.add_argument("--device-map", type=str, default="auto", help="Transformers device_map value, or 'none'.")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-length", type=int, default=0, help="Context tokens per scoring window; 0 uses registry/model limit.")
    ap.add_argument("--stride", type=int, default=1024, help="New tokens per scoring window.")
    ap.add_argument("--overlength-policy", type=str, default="skip", choices=["skip", "slide", "error"], help="What to do if a story is longer than the model window.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    spec = get_model_spec(args.model_key)
    model_name = args.model_name or spec.model_name
    dtype = args.dtype or spec.dtype
    trust_remote_code = bool(args.trust_remote_code or spec.trust_remote_code)
    out_path = args.output or Path("results/analysis/natural_language/perplexity/runs") / f"{args.model_key}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_table(args.dataset)
    if "text" not in df.columns:
        raise ValueError("Dataset must contain a `text` column. Run src/analysis/natural_language/perplexity/prepare_dataset.py first.")
    if "item_id" not in df.columns:
        raise ValueError("Dataset must contain an `item_id` column. Run src/analysis/natural_language/perplexity/prepare_dataset.py first.")

    if args.end and args.end > args.start:
        df = df.iloc[args.start : args.end].copy()
    elif args.start:
        df = df.iloc[args.start :].copy()
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    completed = read_completed_ids(out_path) if args.resume else set()
    if completed:
        df = df[~df["item_id"].astype(str).isin(completed)].copy()

    print(f"Loading tokenizer: {model_name}", flush=True)
    tokenizer = load_tokenizer(model_name, trust_remote_code=trust_remote_code)
    print(f"Loading model: {model_name} dtype={dtype}", flush=True)
    model, input_device = load_model(
        model_name,
        dtype,
        args.device_map,
        trust_remote_code,
        config_overrides=spec.model_config_overrides(),
    )
    max_length = bounded_context_length(tokenizer, model, args.max_length, spec.context_length)
    stride = args.stride if args.stride > 0 else max_length
    if stride > max_length:
        raise ValueError(f"stride ({stride}) must be <= max_length ({max_length})")
    print(
        f"Scoring {len(df)} rows with max_length={max_length} stride={stride} "
        f"overlength_policy={args.overlength_policy}",
        flush=True,
    )

    write_header = not out_path.exists() or not args.resume
    mode = "a" if args.resume else "w"
    with out_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for i, row in enumerate(df.itertuples(index=False), start=1):
            text = str(getattr(row, "text")).strip()
            scores = score_text(
                text,
                tokenizer,
                model,
                input_device,
                max_length=max_length,
                stride=stride,
                overlength_policy=args.overlength_policy,
            )
            result = {
                "model_key": spec.key,
                "model_name": model_name,
                "family": spec.family,
                "variant": spec.variant,
                "quantization": "none",
                "model_context_length": spec.context_length,
                "model_hidden_size": spec.hidden_size,
                "item_id": _str_or_empty(getattr(row, "item_id", "")),
                "story_id": _str_or_empty(getattr(row, "story_id", "")),
                "video_id": _str_or_empty(getattr(row, "video_id", "")),
                "reading_age": _str_or_empty(getattr(row, "reading_age", "")),
                "age_lo": _float_or_empty(getattr(row, "age_lo", "")),
                "age_hi": _float_or_empty(getattr(row, "age_hi", "")),
                "age_median": _float_or_empty(getattr(row, "age_median", "")),
                "n_chars": _str_or_empty(getattr(row, "n_chars", "")),
                "n_words": _str_or_empty(getattr(row, "n_words", "")),
                "n_tokens": int(scores["n_tokens"]),
                "n_loss_tokens": int(scores["n_loss_tokens"]),
                "nll_sum": scores["nll_sum"],
                "mean_nll": scores["mean_nll"],
                "src/analysis/natural_language/perplexity": scores["src/analysis/natural_language/perplexity"],
                "log10_perplexity": scores["log10_perplexity"],
                "n_windows": int(scores["n_windows"]),
                "fits_single_window": int(scores["fits_single_window"]),
                "overlength_policy": args.overlength_policy,
                "max_length": max_length,
                "stride": stride,
            }
            writer.writerow(result)
            if i % 10 == 0:
                f.flush()
                print(f"Processed {i}/{len(df)} rows", flush=True)

    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
