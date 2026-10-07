#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def mean_pool_hidden_states_encoder(hidden_states: tuple["torch.Tensor", ...], attention_mask: "torch.Tensor") -> "torch.Tensor":
    valid = attention_mask.bool()
    if valid.shape[1] >= 1:
        valid[:, 0] = False
    lengths = attention_mask.long().sum(dim=1).clamp(min=1)
    sep_idx = (lengths - 1).clamp(min=0)
    valid[torch.arange(valid.shape[0], device=valid.device), sep_idx] = False
    denom = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
    pooled = []
    for hs in hidden_states:
        pooled.append((hs * valid.unsqueeze(-1)).sum(dim=1) / denom)
    return torch.stack(pooled, dim=1)


def mean_pool_hidden_states_decoder(hidden_states: tuple["torch.Tensor", ...], attention_mask: "torch.Tensor") -> "torch.Tensor":
    valid = attention_mask.bool()
    if valid.shape[1] >= 1:
        valid[:, 0] = False
    denom = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
    pooled = []
    for hs in hidden_states:
        pooled.append((hs * valid.unsqueeze(-1)).sum(dim=1) / denom)
    return torch.stack(pooled, dim=1)


def build_messages(text: str, reasoning: str) -> List[Dict[str, str]]:
    system = f"Reasoning: {reasoning}\nYou are a helpful assistant."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract pooled per-layer AoA word vectors for encoder or decoder models.")
    ap.add_argument("--aoa-lexicon", type=Path, default=Path("data/aoa/AoA_ratings_Kuperman_et_al_BRM_with_PoS.xlsx"))
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--family", type=str, choices=["encoder", "decoder"], required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--word-col", type=str, default="Word")
    ap.add_argument("--aoa-col", type=str, default="Rating.Mean")
    ap.add_argument("--low", type=float, default=4.0)
    ap.add_argument("--high", type=float, default=9.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--reasoning", type=str, default="low", choices=["low", "medium", "high"])
    ap.add_argument("--device-map", type=str, choices=["auto", "none"], default="auto")
    args = ap.parse_args()

    global torch
    import torch  # type: ignore
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # type: ignore

    lex = pd.read_excel(args.aoa_lexicon)
    lex[args.aoa_col] = pd.to_numeric(lex[args.aoa_col], errors="coerce")
    lex[args.word_col] = lex[args.word_col].astype(str).str.strip()
    lex = lex[lex[args.aoa_col].notna() & lex[args.word_col].ne("")].copy()
    lex = lex.drop_duplicates(subset=[args.word_col]).reset_index(drop=True)
    lex["aoa_label"] = np.where(lex[args.aoa_col] < args.low, 0, np.where(lex[args.aoa_col] > args.high, 1, np.nan))
    lex = lex[lex["aoa_label"].notna()].copy()
    lex["aoa_label"] = lex["aoa_label"].astype(int)

    args.outdir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.family == "decoder" and tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.family == "encoder":
        model = AutoModel.from_pretrained(args.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_device = device
    else:
        if args.device_map == "auto":
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
            try:
                input_device = model.get_input_embeddings().weight.device
            except Exception:
                input_device = next(model.parameters()).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
            model.to(device)
            input_device = device
    model.eval()

    words = lex[args.word_col].tolist()
    vectors: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(words), args.batch_size):
            batch_words = words[start : start + args.batch_size]
            if args.family == "encoder":
                enc = tokenizer(
                    batch_words,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                )
                enc = {k: v.to(input_device) for k, v in enc.items()}
                out = model(**enc, output_hidden_states=True, return_dict=True)
                pooled = mean_pool_hidden_states_encoder(out.hidden_states, enc["attention_mask"])
            else:
                prompts = [
                    tokenizer.apply_chat_template(
                        build_messages(word, args.reasoning),
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    for word in batch_words
                ]
                enc = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                )
                enc = {k: v.to(input_device) for k, v in enc.items()}
                out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
                pooled = mean_pool_hidden_states_decoder(out.hidden_states, enc["attention_mask"])
            vectors.append(pooled.to(dtype=torch.float32).cpu().numpy())

    vec_arr = np.concatenate(vectors, axis=0)
    np.savez_compressed(
        args.outdir / "word_vectors.npz",
        vectors=vec_arr,
        word=np.array(words),
        aoa=np.array(lex[args.aoa_col].to_numpy(dtype=float), dtype=np.float32),
        aoa_label=np.array(lex["aoa_label"].to_numpy(dtype=int), dtype=np.int64),
        meta=json.dumps(
            {
                "model": args.model,
                "family": args.family,
                "word_col": args.word_col,
                "aoa_col": args.aoa_col,
                "low": args.low,
                "high": args.high,
                "reasoning": args.reasoning,
                "max_length": args.max_length,
            }
        ),
    )
    lex[[args.word_col, args.aoa_col, "aoa_label"]].to_csv(args.outdir / "aoa_word_split.csv", index=False)
    print(f"Wrote: {args.outdir / 'word_vectors.npz'} shape={vec_arr.shape}")


if __name__ == "__main__":
    main()
