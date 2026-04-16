#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
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


def sent_split(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s and s.strip()]


def chunk_by_chars(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    if max_chars <= 0:
        return [text]
    s = text.strip()
    if len(s) <= max_chars:
        return [s]
    chunks: List[str] = []
    step = max(1, max_chars - max(0, overlap_chars))
    for start in range(0, len(s), step):
        end = min(len(s), start + max_chars)
        chunks.append(s[start:end])
        if end == len(s):
            break
    return chunks


def _safe_json_loads(value: Any) -> Optional[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    s = str(value).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def build_event_texts(review_events_json: Any) -> Optional[List[Dict[str, Any]]]:
    events = _safe_json_loads(review_events_json)
    if not isinstance(events, list):
        return None
    out: List[Dict[str, Any]] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        sentences = e.get("sentences")
        if not isinstance(sentences, list):
            continue
        texts: List[str] = []
        for s in sentences:
            if isinstance(s, dict) and "text" in s:
                texts.append(str(s["text"]))
            elif isinstance(s, str):
                texts.append(s)
        joined = " ".join(t.strip() for t in texts if t and str(t).strip()).strip()
        if not joined:
            continue
        out.append(
            {
                "event": e.get("event"),
                "type": e.get("type"),
                "text": joined,
            }
        )
    return out or None


def mean_pool_hidden_states(hidden_states: Tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Returns (n_layers, hidden) pooled vectors for a single example.
    Pools over non-padding tokens and drops the first token if possible (often a special token).
    """
    # hidden_states: tuple(n_layers, (1, seq, hidden))
    valid = attention_mask[0].bool()  # (seq,)
    seq_len = int(valid.shape[0])
    if seq_len > 1:
        valid[0] = False
    denom = valid.sum().clamp(min=1).to(hidden_states[0].device)

    pooled: List[torch.Tensor] = []
    for hs in hidden_states:
        # (1, seq, hidden)
        vecs = hs[0]
        masked = vecs * valid.unsqueeze(-1)
        pooled.append(masked.sum(dim=0) / denom)
    return torch.stack(pooled, dim=0)  # (n_layers, hidden)


def build_messages(text: str, reasoning: str) -> List[Dict[str, str]]:
    # gpt-oss uses a chat template that applies the “harmony response format”.
    system = f"Reasoning: {reasoning}\nYou are a helpful assistant."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract pooled per-layer vectors from openai/gpt-oss-* activations.")
    ap.add_argument("--master", type=Path, default=Path("data/processed/master_stories.parquet"))
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/gpt_oss_vectors"))
    ap.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    ap.add_argument("--unit", type=str, choices=["story", "event"], default="story")
    ap.add_argument("--text-col", type=str, default="transcript_best")
    ap.add_argument("--id-col", type=str, default="video_id")
    ap.add_argument("--reasoning", type=str, default="low", choices=["low", "medium", "high"])
    ap.add_argument(
        "--device-map",
        type=str,
        choices=["auto", "none"],
        default="auto",
        help="Use transformers device_map. 'auto' requires accelerate; 'none' loads on a single device.",
    )
    ap.add_argument("--max-length", type=int, default=1024, help="Max tokens per forward pass.")
    ap.add_argument("--chunk-chars", type=int, default=8000, help="Story chunk size in characters (story unit).")
    ap.add_argument("--chunk-overlap", type=int, default=1000, help="Chunk overlap in characters (story unit).")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of items (0 = no limit).")
    args = ap.parse_args()

    global torch
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    df = read_table(args.master)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column: {args.text_col}")
    if args.id_col not in df.columns:
        raise ValueError(f"Missing id column: {args.id_col}")

    cols = [args.id_col, "story_id", "age_lo", "age_hi", "age_mid", args.text_col]
    if args.unit == "event":
        cols.append("review_events_json")
    cols = [c for c in cols if c in df.columns]
    work = df[cols].copy()
    work = work.rename(columns={args.id_col: "item_id", args.text_col: "text"})
    work["item_id"] = work["item_id"].astype(str)
    work = work[work["text"].notna()].reset_index(drop=True)
    if args.limit and args.limit > 0:
        work = work.head(args.limit).reset_index(drop=True)

    args.outdir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Use chat template to apply harmony format. Also ensures pad token exists.
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model}  device_map={args.device_map}", flush=True)
    device = None
    input_device = None
    if args.device_map == "auto":
        # device_map=auto requires accelerate; if it's missing, this will throw.
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
            device = None
        except Exception as e:
            print("Failed to load with device_map=auto.", flush=True)
            print(f"Exception: {type(e).__name__}: {e}", flush=True)
            print("Tip: install `accelerate` in the conda env, or rerun with --device-map none (may OOM).", flush=True)
            raise
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
        model.to(device)
    model.eval()
    try:
        input_device = model.get_input_embeddings().weight.device
    except Exception:
        input_device = next(model.parameters()).device
    print(f"Model input device: {input_device}", flush=True)

    item_ids: List[str] = []
    story_ids: List[str] = []
    age_lo: List[float] = []
    age_hi: List[float] = []
    age_mid: List[float] = []
    vectors: List[np.ndarray] = []
    segment_counts: List[int] = []

    seg_rows: List[Dict[str, Any]] = []

    for row in work.itertuples(index=False):
        text = str(row.text).strip()
        if not text:
            continue

        segments: List[Dict[str, Any]] = []
        if args.unit == "event" and hasattr(row, "review_events_json"):
            evs = build_event_texts(getattr(row, "review_events_json"))
            if evs:
                segments = evs

        if not segments:
            # Fall back to story chunks
            for idx, chunk in enumerate(chunk_by_chars(text, args.chunk_chars, args.chunk_overlap)):
                segments.append({"event": None, "type": None, "text": chunk, "segment": idx})

        pooled_segments: List[torch.Tensor] = []
        for seg in segments:
            messages = build_messages(seg["text"], args.reasoning)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            enc = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                padding=False,
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            # With device_map=auto, the embedding layer typically lives on cuda:0; inputs must match.
            input_ids = input_ids.to(input_device)
            attention_mask = attention_mask.to(input_device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            pooled = mean_pool_hidden_states(out.hidden_states, attention_mask)
            pooled_segments.append(pooled.detach().cpu())

            seg_rows.append(
                {
                    "item_id": str(row.item_id),
                    "story_id": "" if not isinstance(getattr(row, "story_id", None), str) else getattr(row, "story_id"),
                    "unit": args.unit,
                    "event": seg.get("event"),
                    "event_type": seg.get("type"),
                    "segment": seg.get("segment"),
                    "n_prompt_tokens": int(input_ids.shape[1]),
                }
            )

        story_vec = torch.stack(pooled_segments, dim=0).mean(dim=0)  # (n_layers, hidden)
        # Ensure numpy conversion works regardless of torch dtype/device (e.g., bfloat16)
        story_vec_np = story_vec.to(dtype=torch.float32).cpu().numpy()
        vectors.append(story_vec_np.astype(np.float32, copy=False))
        item_ids.append(str(row.item_id))
        story_ids.append("" if not isinstance(getattr(row, "story_id", None), str) else getattr(row, "story_id"))

        def _to_f(x: Any) -> float:
            try:
                if pd.isna(x):
                    return float("nan")
            except Exception:
                pass
            try:
                return float(x)
            except Exception:
                return float("nan")

        age_lo.append(_to_f(getattr(row, "age_lo", float("nan"))))
        age_hi.append(_to_f(getattr(row, "age_hi", float("nan"))))
        age_mid.append(_to_f(getattr(row, "age_mid", float("nan"))))
        segment_counts.append(len(pooled_segments))
        if len(item_ids) % 25 == 0:
            print(f"Processed {len(item_ids)}/{len(work)} items...", flush=True)

    if not vectors:
        raise SystemExit("No vectors extracted (no usable texts found).")

    vec_arr = np.stack(vectors, axis=0)  # (n_items, n_layers, hidden)
    np.savez_compressed(
        args.outdir / "story_vectors.npz",
        vectors=vec_arr,
        item_id=np.array(item_ids),
        story_id=np.array(story_ids),
        age_lo=np.array(age_lo, dtype=np.float32),
        age_hi=np.array(age_hi, dtype=np.float32),
        age_mid=np.array(age_mid, dtype=np.float32),
        meta=json.dumps(
            {
                "model": args.model,
                "unit": args.unit,
                "text_col": args.text_col,
                "id_col": args.id_col,
                "reasoning": args.reasoning,
                "chunk_chars": args.chunk_chars,
                "chunk_overlap": args.chunk_overlap,
                "max_length": args.max_length,
            }
        ),
    )

    pd.DataFrame({"item_id": item_ids, "n_segments": segment_counts}).to_csv(args.outdir / "segment_counts.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(args.outdir / "segments_index.csv", index=False)
    print(f"Wrote: {args.outdir / 'story_vectors.npz'}  shape={vec_arr.shape}")


if __name__ == "__main__":
    main()
