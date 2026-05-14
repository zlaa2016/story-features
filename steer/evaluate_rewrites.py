#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.analyze_text_features import (
    aoa_features,
    basic_features,
    estimate_syllables,
    frequency_features,
    lexical_diversity_features,
    load_aoa_lexicon,
    readability_features,
    sent_split,
    word_split,
)


def text_metrics(text: str, aoa_map: Dict[str, float], corpus_counts: Dict[str, int], corpus_total: int) -> Dict[str, Any]:
    words = word_split(text)
    base = basic_features(text)
    syllables = sum(estimate_syllables(w) for w in words)
    out: Dict[str, Any] = {}
    out.update(base)
    out.update(lexical_diversity_features(words))
    out.update(readability_features(int(base.get("num_sentences", 0)), int(base.get("num_words", 0)), syllables))
    out["syllable_count_est"] = int(syllables)
    out.update(frequency_features(words, corpus_counts, corpus_total))
    if aoa_map:
        out.update(aoa_features(words, aoa_map, lookup_mode="wordform"))
        vals = [aoa_map[w] for w in words if w in aoa_map]
        if vals:
            arr = np.asarray(vals, dtype=float)
            out["aoa_median"] = float(np.median(arr))
            out["aoa_p90"] = float(np.quantile(arr, 0.90))
            out["aoa_max"] = float(arr.max())
        else:
            out["aoa_median"] = float("nan")
            out["aoa_p90"] = float("nan")
            out["aoa_max"] = float("nan")
    return out


def maybe_score_age(texts: List[str], model_dir: Path, batch_size: int = 8) -> np.ndarray:
    import torch  # type: ignore
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    scores: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.append(probs.cpu().numpy())
    return np.concatenate(scores, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate original vs steered rewritten stories.")
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--original-col", type=str, default="original_text")
    ap.add_argument("--rewritten-col", type=str, default="rewritten_text")
    ap.add_argument("--aoa-lexicon", type=Path, default=Path("data/aoa/AoA_ratings_Kuperman_et_al_BRM_with_PoS.xlsx"))
    ap.add_argument("--age-model-dir", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)
    df = df[df[args.original_col].notna() & df[args.rewritten_col].notna()].copy()
    df[args.original_col] = df[args.original_col].astype(str)
    df[args.rewritten_col] = df[args.rewritten_col].astype(str)

    all_texts = pd.concat([df[args.original_col], df[args.rewritten_col]], ignore_index=True).astype(str).tolist()
    corpus_counts: Dict[str, int] = {}
    corpus_total = 0
    for t in all_texts:
        ws = word_split(t)
        corpus_total += len(ws)
        for w in ws:
            corpus_counts[w] = corpus_counts.get(w, 0) + 1
    aoa_map = load_aoa_lexicon(args.aoa_lexicon)

    rows: List[Dict[str, Any]] = []
    originals = df[args.original_col].tolist()
    rewrites = df[args.rewritten_col].tolist()
    vec = TfidfVectorizer(min_df=1, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(originals + rewrites)
    n = len(originals)
    cos = (X[:n].multiply(X[n:]).sum(axis=1)).A1

    age_orig = age_rewrite = None
    if args.age_model_dir is not None and args.age_model_dir.exists():
        age_orig = maybe_score_age(originals, args.age_model_dir)
        age_rewrite = maybe_score_age(rewrites, args.age_model_dir)

    for i, row in df.reset_index(drop=True).iterrows():
        om = text_metrics(str(row[args.original_col]), aoa_map, corpus_counts, corpus_total)
        rm = text_metrics(str(row[args.rewritten_col]), aoa_map, corpus_counts, corpus_total)
        out: Dict[str, Any] = {}
        for k, v in row.items():
            out[k] = v
        out["tfidf_cosine"] = float(cos[i])
        if age_orig is not None and age_rewrite is not None:
            out["age_prob_original"] = float(age_orig[i])
            out["age_prob_rewritten"] = float(age_rewrite[i])
            out["delta_age_prob"] = float(age_rewrite[i] - age_orig[i])
        for k, v in om.items():
            out[f"orig_{k}"] = v
        for k, v in rm.items():
            out[f"rew_{k}"] = v
        for k in ["aoa_mean", "aoa_median", "aoa_p90", "aoa_max", "flesch_kincaid_grade_est", "mean_sentence_length_words", "rare_frac_corpus_le_5"]:
            out[f"delta_{k}"] = float(out.get(f"rew_{k}", np.nan) - out.get(f"orig_{k}", np.nan))
        rows.append(out)

    out_df = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.outdir / "per_story_metrics.csv", index=False)
    summary: Dict[str, Any] = {
        "n_items": int(len(out_df)),
        "mean_tfidf_cosine": float(out_df["tfidf_cosine"].mean()),
    }
    for c in [c for c in out_df.columns if c.startswith("delta_")]:
        summary[f"mean_{c}"] = float(pd.to_numeric(out_df[c], errors="coerce").mean())
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
