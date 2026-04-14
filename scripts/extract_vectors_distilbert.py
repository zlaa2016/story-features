#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# NOTE: torch/transformers are imported lazily inside main() so `--help` works
# even outside the conda environment.


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists() and path.suffix.lower() == ".parquet":
        alt = path.with_suffix(".csv.gz")
        if alt.exists():
            path = alt
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError:
            alt = path.with_suffix(".csv.gz")
            if alt.exists():
                return pd.read_csv(alt, low_memory=False)
            raise
    if path.suffix.lower() in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported input: {path}")


def split_sentences(text: str) -> List[str]:
    # Rough sentence split; matches the notebook intent.
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s and s.strip()]


def build_token_chunks(
    sentences: List[str],
    tokenizer: DistilBertTokenizerFast,
    max_len: int,
    stride_tokens: int,
) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    buf_tok = 0

    def tok_len(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False)["input_ids"])

    for sent in sentences:
        t = tok_len(sent)
        # leave room for [CLS] and [SEP]
        if buf and buf_tok + t > max_len - 2:
            chunks.append(" ".join(buf))
            while buf and buf_tok > stride_tokens:
                buf_tok -= tok_len(buf.pop(0))
        buf.append(sent)
        buf_tok += t
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def mean_pool_hidden_states(
    hidden_states: Tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Returns pooled vectors with shape (batch, n_layers, hidden).
    Excludes padding tokens; also drops first ([CLS]) and last ([SEP]) real token when possible.
    """
    device = attention_mask.device
    batch, seq_len = attention_mask.shape
    valid = attention_mask.bool()

    # Drop [CLS]
    if seq_len >= 1:
        valid[:, 0] = False

    # Drop [SEP] (last real token index per row)
    lengths = attention_mask.long().sum(dim=1).clamp(min=1)  # includes specials
    sep_idx = (lengths - 1).clamp(min=0)
    valid[torch.arange(batch, device=device), sep_idx] = False

    denom = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (batch, 1)

    pooled_layers: List[torch.Tensor] = []
    for hs in hidden_states:
        # hs: (batch, seq_len, hidden)
        masked = hs * valid.unsqueeze(-1)
        pooled_layers.append(masked.sum(dim=1) / denom)

    return torch.stack(pooled_layers, dim=1)  # (batch, n_layers, hidden)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract DistilBERT story vectors (mean pooled per layer).")
    ap.add_argument("--master", type=Path, default=Path("data/processed/master_stories.parquet"))
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/distilbert_vectors"))
    ap.add_argument("--model", type=str, default="distilbert-base-uncased")
    ap.add_argument("--text-col", type=str, default="transcript_best")
    ap.add_argument("--id-col", type=str, default="video_id")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of stories (0 = no limit).")
    args = ap.parse_args()

    global torch
    import torch  # type: ignore
    from transformers import DistilBertModel, DistilBertTokenizerFast  # type: ignore

    df = read_table(args.master)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column: {args.text_col}")
    if args.id_col not in df.columns:
        raise ValueError(f"Missing id column: {args.id_col}")

    work = df[[args.id_col, "story_id", "age_lo", "age_hi", "age_mid", args.text_col]].copy()
    work = work.rename(columns={args.id_col: "item_id", args.text_col: "text"})
    work["item_id"] = work["item_id"].astype(str)
    work = work[work["text"].notna()].reset_index(drop=True)
    if args.limit and args.limit > 0:
        work = work.head(args.limit).reset_index(drop=True)

    args.outdir.mkdir(parents=True, exist_ok=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model)
    model = DistilBertModel.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    vectors: List[np.ndarray] = []
    item_ids: List[str] = []
    story_ids: List[Optional[str]] = []
    age_lo: List[Optional[float]] = []
    age_hi: List[Optional[float]] = []
    age_mid: List[Optional[float]] = []
    chunk_counts: List[int] = []

    for row in work.itertuples(index=False):
        text = str(row.text)
        sentences = split_sentences(text)
        chunks = build_token_chunks(sentences, tokenizer, max_len=args.max_len, stride_tokens=args.stride)
        if not chunks:
            continue

        pooled_chunks: List[torch.Tensor] = []
        for start in range(0, len(chunks), args.batch_size):
            batch_texts = chunks[start : start + args.batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            pooled = mean_pool_hidden_states(out.hidden_states, enc["attention_mask"])
            pooled_chunks.append(pooled.cpu())

        pooled_all = torch.cat(pooled_chunks, dim=0)  # (n_chunks, n_layers, hidden)
        story_vec = pooled_all.mean(dim=0)  # (n_layers, hidden)

        vectors.append(story_vec.numpy().astype(np.float32))
        item_ids.append(str(row.item_id))
        story_ids.append(row.story_id if isinstance(row.story_id, str) else (None if pd.isna(row.story_id) else str(row.story_id)))
        age_lo.append(None if pd.isna(row.age_lo) else float(row.age_lo))
        age_hi.append(None if pd.isna(row.age_hi) else float(row.age_hi))
        age_mid.append(None if pd.isna(row.age_mid) else float(row.age_mid))
        chunk_counts.append(int(pooled_all.shape[0]))

    if not vectors:
        raise SystemExit("No vectors extracted (no texts found).")

    vec_arr = np.stack(vectors, axis=0)  # (n_items, n_layers, hidden)

    np.savez_compressed(
        args.outdir / "story_vectors.npz",
        vectors=vec_arr,
        item_id=np.array(item_ids),
        story_id=np.array([s if s is not None else "" for s in story_ids]),
        age_lo=np.array([a if a is not None else np.nan for a in age_lo], dtype=np.float32),
        age_hi=np.array([a if a is not None else np.nan for a in age_hi], dtype=np.float32),
        age_mid=np.array([a if a is not None else np.nan for a in age_mid], dtype=np.float32),
        meta=json.dumps(
            {
                "model": args.model,
                "text_col": args.text_col,
                "id_col": args.id_col,
                "max_len": args.max_len,
                "stride": args.stride,
            }
        ),
    )

    pd.DataFrame({"item_id": item_ids, "n_chunks": chunk_counts}).to_csv(args.outdir / "chunk_counts.csv", index=False)
    print(f"Wrote: {args.outdir / 'story_vectors.npz'}  shape={vec_arr.shape}")


if __name__ == "__main__":
    main()
