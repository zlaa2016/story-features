#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _normalize_story_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float)):
        try:
            return str(int(value)).zfill(6)
        except Exception:
            return None
    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"\d{1,6}", s):
        return s.zfill(6)
    return None


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
        df = pd.read_csv(path, low_memory=False)
        if "story_id" in df.columns:
            df["story_id"] = df["story_id"].map(_normalize_story_id)
        if "video_id" in df.columns:
            df["video_id"] = df["video_id"].astype(str)
        return df
    raise ValueError(f"Unsupported input: {path}")


def filter_cleaned_rows(df: pd.DataFrame, meta_path: Path, cleaned_col: str) -> pd.DataFrame:
    meta = pd.read_csv(meta_path, low_memory=False)
    keep = pd.to_numeric(meta.get(cleaned_col), errors="coerce").fillna(0).astype(int) == 1
    meta = meta[keep].copy()
    matched = pd.Series(False, index=df.index)
    out = df.copy()
    if "story_id" in out.columns and "story_id" in meta.columns:
        meta["story_id"] = meta["story_id"].map(_normalize_story_id)
        matched = matched | out["story_id"].isin(set(meta["story_id"].dropna().astype(str)))
    if "video_id" in out.columns and "video_id" in meta.columns:
        meta["video_id"] = meta["video_id"].astype(str).str.strip()
        out["video_id"] = out["video_id"].astype(str).str.strip()
        matched = matched | out["video_id"].isin(set(meta["video_id"].dropna().astype(str)))
    return out[matched].copy()


def split_sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in sents if s and s.strip()]


def build_token_chunks(sentences: List[str], tokenizer: "PreTrainedTokenizerBase", max_len: int, stride_tokens: int) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    buf_tok = 0

    def tok_len(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False)["input_ids"])

    for sent in sentences:
        t = tok_len(sent)
        if buf and buf_tok + t > max_len - 2:
            chunks.append(" ".join(buf))
            while buf and buf_tok > stride_tokens:
                buf_tok -= tok_len(buf.pop(0))
        buf.append(sent)
        buf_tok += t
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def mean_pool_hidden_states(hidden_states: tuple["torch.Tensor", ...], attention_mask: "torch.Tensor") -> "torch.Tensor":
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract pooled per-layer story vectors for generic encoder models.")
    ap.add_argument("--master", type=Path, default=Path("data/processed/master_stories.parquet"))
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--text-col", type=str, default="transcript_best")
    ap.add_argument("--id-col", type=str, default="video_id")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cleaned-meta", type=Path, default=None)
    ap.add_argument("--cleaned-col", type=str, default="cleaned")
    args = ap.parse_args()

    global torch
    import torch  # type: ignore
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    df = read_table(args.master)
    if args.cleaned_meta is not None and args.cleaned_meta.exists():
        df = filter_cleaned_rows(df, args.cleaned_meta, args.cleaned_col)
    cols = [c for c in [args.id_col, "story_id", "age_lo", "age_hi", "age_mid", args.text_col] if c in df.columns]
    work = df[cols].copy().rename(columns={args.id_col: "item_id", args.text_col: "text"})
    work["item_id"] = work["item_id"].astype(str)
    work = work[work["text"].notna()].reset_index(drop=True)
    if args.limit and args.limit > 0:
        work = work.head(args.limit).reset_index(drop=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    vectors = []
    item_ids: List[str] = []
    story_ids: List[str] = []
    age_lo: List[float] = []
    age_hi: List[float] = []
    age_mid: List[float] = []
    chunk_rows: List[Dict[str, Any]] = []

    for row in work.itertuples(index=False):
        sents = split_sentences(str(row.text))
        chunks = build_token_chunks(sents, tokenizer, max_len=args.max_len, stride_tokens=args.stride)
        if not chunks:
            continue
        pooled_chunks = []
        for start in range(0, len(chunks), args.batch_size):
            batch_texts = chunks[start : start + args.batch_size]
            enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, return_dict=True)
            pooled = mean_pool_hidden_states(out.hidden_states, enc["attention_mask"])
            pooled_chunks.append(pooled.cpu())
        pooled_all = torch.cat(pooled_chunks, dim=0)
        story_vec = pooled_all.mean(dim=0).to(dtype=torch.float32).cpu().numpy()
        vectors.append(story_vec.astype(np.float32, copy=False))
        item_ids.append(str(row.item_id))
        sid = getattr(row, "story_id", "")
        story_ids.append("" if sid is None else str(sid))
        age_lo.append(float("nan") if pd.isna(getattr(row, "age_lo", np.nan)) else float(getattr(row, "age_lo")))
        age_hi.append(float("nan") if pd.isna(getattr(row, "age_hi", np.nan)) else float(getattr(row, "age_hi")))
        age_mid.append(float("nan") if pd.isna(getattr(row, "age_mid", np.nan)) else float(getattr(row, "age_mid")))
        chunk_rows.append({"item_id": str(row.item_id), "n_chunks": int(pooled_all.shape[0])})

    if not vectors:
        raise SystemExit("No vectors extracted.")
    vec_arr = np.stack(vectors, axis=0)
    np.savez_compressed(
        args.outdir / "story_vectors.npz",
        vectors=vec_arr,
        item_id=np.array(item_ids),
        story_id=np.array(story_ids),
        age_lo=np.array(age_lo, dtype=np.float32),
        age_hi=np.array(age_hi, dtype=np.float32),
        age_mid=np.array(age_mid, dtype=np.float32),
        meta=json.dumps({"model": args.model, "family": "encoder", "max_len": args.max_len, "stride": args.stride}),
    )
    pd.DataFrame(chunk_rows).to_csv(args.outdir / "chunk_counts.csv", index=False)
    print(f"Wrote: {args.outdir / 'story_vectors.npz'} shape={vec_arr.shape}")


if __name__ == "__main__":
    main()
