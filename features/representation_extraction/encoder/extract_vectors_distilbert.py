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
        keep_story = set(meta["story_id"].dropna().astype(str))
        matched = matched | out["story_id"].isin(keep_story)

    if "video_id" in out.columns and "video_id" in meta.columns:
        meta["video_id"] = meta["video_id"].astype(str).str.strip()
        out["video_id"] = out["video_id"].astype(str).str.strip()
        keep_video = set(meta["video_id"].dropna().astype(str))
        matched = matched | out["video_id"].isin(keep_video)

    out = out[matched].copy()
    out["cleaned"] = 1
    return out


def split_sentences(text: str) -> List[str]:
    # Rough sentence split; matches the notebook intent.
    sents = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in sents if s and s.strip()]


def _safe_json_loads(value: Any) -> Optional[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def event_texts_from_review_events(review_events_json: Any) -> Optional[List[Dict[str, Any]]]:
    events = _safe_json_loads(review_events_json)
    if not isinstance(events, list) or not events:
        return None
    out: List[Dict[str, Any]] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        sents = e.get("sentences")
        if not isinstance(sents, list) or not sents:
            continue
        parts: List[str] = []
        for s in sents:
            if isinstance(s, dict) and "text" in s:
                parts.append(str(s["text"]))
            elif isinstance(s, str):
                parts.append(s)
        text = " ".join(p.strip() for p in parts if p and str(p).strip()).strip()
        if not text:
            continue
        out.append({"event": e.get("event"), "type": e.get("type"), "text": text})
    return out or None


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
    ap.add_argument("--outdir", type=Path, default=Path("results/features/representation_extraction/encoder/distilbert_vectors"))
    ap.add_argument("--model", type=str, default="distilbert-base-uncased")
    ap.add_argument("--text-col", type=str, default="transcript_best")
    ap.add_argument("--id-col", type=str, default="video_id")
    ap.add_argument("--unit", type=str, choices=["story", "event"], default="story")
    ap.add_argument(
        "--segment-agg",
        type=str,
        choices=["mean", "concat_mean_std"],
        default="mean",
        help="How to aggregate multiple segments (events) into one story vector.",
    )
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of stories (0 = no limit).")
    ap.add_argument(
        "--cleaned-meta",
        type=Path,
        default=None,
        help="Optional metadata CSV with story_id/video_id and a cleaned column; keeps only cleaned=1 rows.",
    )
    ap.add_argument("--cleaned-col", type=str, default="cleaned")
    args = ap.parse_args()

    global torch
    import torch  # type: ignore
    from transformers import DistilBertModel, DistilBertTokenizerFast  # type: ignore

    df = read_table(args.master)
    if args.cleaned_meta is not None and args.cleaned_meta.exists():
        df = filter_cleaned_rows(df, args.cleaned_meta, args.cleaned_col)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column: {args.text_col}")
    if args.id_col not in df.columns:
        raise ValueError(f"Missing id column: {args.id_col}")
    if args.unit == "event" and "review_events_json" not in df.columns:
        raise ValueError("--unit event requires review_events_json in master (merge review_samples first).")

    cols = [args.id_col, "story_id", "age_lo", "age_hi", "age_mid", args.text_col]
    if args.unit == "event":
        cols.append("review_events_json")
    work = df[cols].copy()
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
    segment_counts: List[int] = []
    total_chunk_counts: List[int] = []
    seg_rows: List[Dict[str, Any]] = []

    for row in work.itertuples(index=False):
        segments: List[Dict[str, Any]] = []
        if args.unit == "event":
            evs = event_texts_from_review_events(getattr(row, "review_events_json"))
            if evs:
                segments = evs
        if not segments:
            segments = [{"event": None, "type": None, "text": str(row.text)}]

        pooled_segments: List[torch.Tensor] = []
        story_total_chunks = 0
        for seg_i, seg in enumerate(segments):
            seg_text = str(seg.get("text", ""))
            seg_sents = split_sentences(seg_text)
            # Chunking happens within each event; we never mix tokens across events.
            chunks = build_token_chunks(seg_sents, tokenizer, max_len=args.max_len, stride_tokens=args.stride)
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
            seg_vec = pooled_all.mean(dim=0)  # (n_layers, hidden)
            pooled_segments.append(seg_vec)
            story_total_chunks += int(pooled_all.shape[0])

            seg_rows.append(
                {
                    "item_id": str(row.item_id),
                    "story_id": row.story_id if isinstance(row.story_id, str) else "",
                    "unit": args.unit,
                    "event": seg.get("event"),
                    "event_type": seg.get("type"),
                    "segment": int(seg_i),
                    "n_chunks": int(pooled_all.shape[0]),
                }
            )

        if not pooled_segments:
            continue

        seg_mat = torch.stack(pooled_segments, dim=0)  # (n_segments, n_layers, hidden)
        story_mean = seg_mat.mean(dim=0)  # (n_layers, hidden)
        if args.segment_agg == "concat_mean_std":
            story_std = seg_mat.std(dim=0, unbiased=False)  # (n_layers, hidden)
            story_vec = torch.cat([story_mean, story_std], dim=-1)  # (n_layers, 2*hidden)
        else:
            story_vec = story_mean  # (n_layers, hidden)

        vectors.append(story_vec.numpy().astype(np.float32))
        item_ids.append(str(row.item_id))
        story_ids.append(row.story_id if isinstance(row.story_id, str) else (None if pd.isna(row.story_id) else str(row.story_id)))
        age_lo.append(None if pd.isna(row.age_lo) else float(row.age_lo))
        age_hi.append(None if pd.isna(row.age_hi) else float(row.age_hi))
        age_mid.append(None if pd.isna(row.age_mid) else float(row.age_mid))
        segment_counts.append(int(len(pooled_segments)))
        total_chunk_counts.append(int(story_total_chunks))

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
                "unit": args.unit,
                "segment_agg": args.segment_agg,
            }
        ),
    )

    pd.DataFrame(
        {"item_id": item_ids, "n_segments": segment_counts, "n_chunks_total": total_chunk_counts}
    ).to_csv(args.outdir / "segment_counts.csv", index=False)
    if seg_rows:
        pd.DataFrame(seg_rows).to_csv(args.outdir / "segments_index.csv", index=False)
    print(f"Wrote: {args.outdir / 'story_vectors.npz'}  shape={vec_arr.shape}")


if __name__ == "__main__":
    main()
