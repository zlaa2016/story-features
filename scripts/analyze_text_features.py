#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "you",
    "your",
    "yours",
    "yourself",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
}


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


def word_split(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def basic_features(text: str) -> Dict[str, Any]:
    sents = sent_split(text)
    words = word_split(text)
    num_sents = len(sents)
    num_words = len(words)
    mean_sent_len = (num_words / num_sents) if num_sents else 0.0
    mean_word_len = (sum(len(w) for w in words) / num_words) if num_words else 0.0
    pronouns = sum(1 for w in words if w in PRONOUNS)
    return {
        "num_sentences": num_sents,
        "num_words": num_words,
        "mean_sentence_length_words": float(mean_sent_len),
        "mean_word_length_chars": float(mean_word_len),
        "num_pronouns": int(pronouns),
    }


def try_textstat(text: str) -> Dict[str, Any]:
    try:
        import textstat  # type: ignore
    except Exception:
        return {}
    try:
        return {"flesch_reading_ease": float(textstat.flesch_reading_ease(text))}
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute low-level text features for stories and (optionally) events.")
    ap.add_argument("--master", type=Path, default=Path("data/processed/master_stories.parquet"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/text_features.csv"))
    ap.add_argument("--text-col", type=str, default="transcript_best")
    ap.add_argument("--id-col", type=str, default="video_id")
    ap.add_argument("--with-textstat", action="store_true", help="Add textstat metrics if available.")
    args = ap.parse_args()

    df = read_table(args.master)
    needed = [args.id_col, "story_id", "age_lo", "age_hi", "age_mid", args.text_col, "review_n_events"]
    have = [c for c in needed if c in df.columns]
    work = df[have].copy()

    work = work.rename(columns={args.id_col: "item_id", args.text_col: "text"})
    work["item_id"] = work["item_id"].astype(str)
    work = work[work["text"].notna()].reset_index(drop=True)

    feats: List[Dict[str, Any]] = []
    for row in work.itertuples(index=False):
        text = str(row.text)
        f = basic_features(text)
        if args.with_textstat:
            f.update(try_textstat(text))
        out = {"item_id": row.item_id, "story_id": getattr(row, "story_id", None)}
        for k in ["age_lo", "age_hi", "age_mid", "review_n_events"]:
            if hasattr(row, k):
                out[k] = getattr(row, k)
        out.update(f)
        feats.append(out)

    out_df = pd.DataFrame(feats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}  rows={len(out_df)}")


if __name__ == "__main__":
    main()
