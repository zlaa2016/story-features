"""Story-level DistilBERT vectors for the cleaned stories of ``story_level_export.jsonl``.

Same method as ``scripts/extract_vectors_distilbert.py``: a story is cut into chunks of whole sentences of at
most ``--max-len`` tokens (consecutive chunks overlap by about ``--stride`` tokens), every chunk goes through
DistilBERT with all hidden states (the punctuated story is used for sentence boundaries; a "sentence" longer than
the window is cut into word windows so nothing is truncated), tokens are mean-pooled per layer without padding, [CLS] and [SEP], and a
story's vector is the mean over its chunks.  Output shape (stories, 7 hidden states, 768).

One row per *story* (a video can hold several), keyed by ``story_key``, with the video id, story id and the
log view count kept so that popularity-based selections can be made later.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part and part.strip()]


def split_long(sentence: str, token_length, limit: int) -> list[str]:
    """A "sentence" longer than the window (text without punctuation) is cut into word windows of at most ``limit`` tokens."""
    if token_length(sentence) <= limit:
        return [sentence]
    pieces, current, size = [], [], 0
    for word in sentence.split():
        length = token_length(word)
        if current and size + length > limit:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += length
    if current:
        pieces.append(" ".join(current))
    return pieces


def build_chunks(sentences: list[str], token_length, max_len: int, stride: int) -> list[str]:
    chunks, buffer, tokens = [], [], 0
    sentences = [piece for sentence in sentences for piece in split_long(sentence, token_length, max_len - 2)]
    for sentence in sentences:
        size = token_length(sentence)
        if buffer and tokens + size > max_len - 2:
            chunks.append(" ".join(buffer))
            while buffer and tokens > stride:
                tokens -= token_length(buffer.pop(0))
        buffer.append(sentence)
        tokens += size
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks


def pool(hidden_states, attention_mask):
    """(batch, layers, hidden) token means without padding, [CLS] and [SEP]."""
    import torch

    valid = attention_mask.bool().clone()
    valid[:, 0] = False
    lengths = attention_mask.long().sum(dim=1).clamp(min=1)
    valid[torch.arange(len(valid), device=valid.device), (lengths - 1).clamp(min=0)] = False
    denominator = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return torch.stack([(layer * valid.unsqueeze(-1)).sum(dim=1) / denominator for layer in hidden_states], dim=1)


def read_stories(export: Path, flat_csv: Path | None) -> pd.DataFrame:
    keep_videos = set(pd.read_csv(flat_csv, low_memory=False)["video_id"].astype(str)) if flat_csv else None
    rows = []
    with export.open() as handle:
        for line in handle:
            record = json.loads(line)
            if keep_videos is not None and str(record["video_id"]) not in keep_videos:
                continue
            story = record.get("segmented_story") or {}
            text = story.get("punctuated_story") or story.get("cleaned_story_verbatim") or ""
            if not text.strip():
                continue
            views = (record.get("metadata") or {}).get("view_count")
            rows.append({"story_key": record["story_key"], "story_id": record.get("story_id"), "video_id": str(record["video_id"]),
                         "log_views": float(np.log1p(float(views))) if views not in (None, "") else np.nan, "text": text})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, default=Path("data/exports/story_level_export/story_level_export.jsonl"))
    parser.add_argument("--flat-csv", type=Path, default=Path("data/processed/flat_videos_D_2-12.csv"), help="only stories whose video is in this list (pass '' for all)")
    parser.add_argument("--output", type=Path, required=True, help="npz to write")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    stories = read_stories(args.export, args.flat_csv if str(args.flat_csv) else None)
    if args.limit:
        stories = stories.head(args.limit)
    print(f"{len(stories)} stories", flush=True)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    cache: dict[str, int] = {}

    def token_length(sentence: str) -> int:
        if sentence not in cache:
            cache[sentence] = len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
        return cache[sentence]

    chunk_text, owner = [], []
    for index, text in enumerate(stories["text"]):
        for chunk in build_chunks(split_sentences(text), token_length, args.max_len, args.stride):
            chunk_text.append(chunk)
            owner.append(index)
    owner_array = np.asarray(owner)
    order = np.argsort([len(chunk) for chunk in chunk_text])
    print(f"{len(chunk_text)} chunks on {device}", flush=True)

    sums = np.zeros((len(stories), model.config.n_layers + 1, model.config.dim), dtype=np.float64)
    counts = np.zeros(len(stories))
    started = time.time()
    with torch.no_grad():
        for number, start in enumerate(range(0, len(order), args.batch_size)):
            batch = order[start: start + args.batch_size]
            encoded = tokenizer([chunk_text[i] for i in batch], padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
            hidden = model(**encoded, output_hidden_states=True).hidden_states
            pooled = pool(hidden, encoded["attention_mask"]).float().cpu().numpy()
            for row, chunk_index in enumerate(batch):
                sums[owner_array[chunk_index]] += pooled[row]
                counts[owner_array[chunk_index]] += 1
            if number % 100 == 0:
                print(f"batch {number}/{len(order) // args.batch_size}  {time.time() - started:.0f}s", flush=True)
    vectors = (sums / np.maximum(counts, 1)[:, None, None]).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, vectors=vectors, story_key=stories["story_key"].to_numpy(dtype=str), story_id=stories["story_id"].astype(str).to_numpy(),
             video_id=stories["video_id"].to_numpy(dtype=str), log_views=stories["log_views"].to_numpy(dtype=float), n_chunks=counts,
             meta=np.asarray(json.dumps({"model": args.model, "max_len": args.max_len, "stride": args.stride, "unit": "story", "pooling": "token mean per layer, mean over chunks"})))
    print(f"wrote {args.output}: {vectors.shape} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
