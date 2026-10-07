#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.eda.analyze_text_features import (
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


def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    for i, ta in enumerate(a, start=1):
        for j, tb in enumerate(b, start=1):
            if ta == tb:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return int(dp[len(a), len(b)])


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref = word_split(reference)
    pred = word_split(prediction)
    if not ref or not pred:
        return 0.0
    lcs = lcs_len(ref, pred)
    prec = lcs / max(1, len(pred))
    rec = lcs / max(1, len(ref))
    if prec + rec == 0.0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def bleu2(reference: str, prediction: str) -> float:
    ref = word_split(reference)
    pred = word_split(prediction)
    if not ref or not pred:
        return 0.0
    ref_unigrams = {}
    for t in ref:
        ref_unigrams[t] = ref_unigrams.get(t, 0) + 1
    pred_unigrams = {}
    for t in pred:
        pred_unigrams[t] = pred_unigrams.get(t, 0) + 1
    overlap1 = sum(min(c, ref_unigrams.get(t, 0)) for t, c in pred_unigrams.items())
    p1 = overlap1 / max(1, len(pred))

    def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    ref_bi = {}
    for ng in ngrams(ref, 2):
        ref_bi[ng] = ref_bi.get(ng, 0) + 1
    pred_bi = {}
    for ng in ngrams(pred, 2):
        pred_bi[ng] = pred_bi.get(ng, 0) + 1
    overlap2 = sum(min(c, ref_bi.get(ng, 0)) for ng, c in pred_bi.items())
    p2 = overlap2 / max(1, max(0, len(pred) - 1))

    if p1 <= 0.0 or p2 <= 0.0:
        return 0.0
    bp = 1.0 if len(pred) > len(ref) else math.exp(1.0 - len(ref) / max(1, len(pred)))
    return float(bp * math.exp(0.5 * math.log(p1) + 0.5 * math.log(p2)))


def distinct_n(text: str, n: int) -> float:
    toks = word_split(text)
    if len(toks) < n or n <= 0:
        return 0.0
    vals = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return float(len(set(vals)) / max(1, len(vals)))


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


def maybe_score_age(texts: List[str], model_dir: Path, age_cutoff: float = 5.0, batch_size: int = 8) -> Dict[str, np.ndarray]:
    import torch  # type: ignore
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    raw_scores: List[np.ndarray] = []
    older_scores: List[np.ndarray] = []
    task = "regression" if getattr(model.config, "num_labels", 2) == 1 or getattr(model.config, "problem_type", "") == "regression" else "classification"
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            if task == "regression":
                vals = logits.reshape(-1)
                probs = torch.sigmoid(vals - float(age_cutoff))
                raw_scores.append(vals.cpu().numpy())
                older_scores.append(probs.cpu().numpy())
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1]
                raw_scores.append(probs.cpu().numpy())
                older_scores.append(probs.cpu().numpy())
    return {
        "task": np.array([task] * len(texts), dtype=object),
        "raw": np.concatenate(raw_scores, axis=0),
        "older_prob": np.concatenate(older_scores, axis=0),
    }


def maybe_compute_perplexity(texts: List[str], model_name_or_dir: str, batch_size: int = 2, max_length: int = 1024) -> np.ndarray:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_dir, torch_dtype="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    vals: List[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            out = model(**enc, labels=labels)
            logits = out.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            active = shift_labels != -100
            log_probs = torch.log_softmax(logits, dim=-1)
            gather = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            nll = -(gather * active).sum(dim=1) / active.sum(dim=1).clamp(min=1)
            ppls = torch.exp(nll).cpu().numpy().tolist()
            vals.extend(float(x) for x in ppls)
    return np.asarray(vals, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate original vs steered rewritten stories.")
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--original-col", type=str, default="original_text")
    ap.add_argument("--rewritten-col", type=str, default="rewritten_text")
    ap.add_argument("--aoa-lexicon", type=Path, default=Path("data/aoa/AoA_ratings_Kuperman_et_al_BRM_with_PoS.xlsx"))
    ap.add_argument("--age-cutoff", type=float, default=5.0)
    ap.add_argument("--age-model-dir", type=Path, default=None)
    ap.add_argument("--ppl-model", type=str, default=None, help="Optional causal LM model name/path for src/analysis/natural_language/perplexity scoring.")
    ap.add_argument("--ppl-batch-size", type=int, default=2)
    ap.add_argument("--ppl-max-length", type=int, default=1024)
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
        age_orig = maybe_score_age(originals, args.age_model_dir, age_cutoff=float(args.age_cutoff))
        age_rewrite = maybe_score_age(rewrites, args.age_model_dir, age_cutoff=float(args.age_cutoff))

    ppl_orig = ppl_rewrite = None
    if args.ppl_model:
        ppl_orig = maybe_compute_perplexity(originals, args.ppl_model, batch_size=int(args.ppl_batch_size), max_length=int(args.ppl_max_length))
        ppl_rewrite = maybe_compute_perplexity(rewrites, args.ppl_model, batch_size=int(args.ppl_batch_size), max_length=int(args.ppl_max_length))

    for i, row in df.reset_index(drop=True).iterrows():
        orig_text = str(row[args.original_col])
        rew_text = str(row[args.rewritten_col])
        om = text_metrics(str(row[args.original_col]), aoa_map, corpus_counts, corpus_total)
        rm = text_metrics(str(row[args.rewritten_col]), aoa_map, corpus_counts, corpus_total)
        out: Dict[str, Any] = {}
        for k, v in row.items():
            out[k] = v
        out["tfidf_cosine"] = float(cos[i])
        if age_orig is not None and age_rewrite is not None:
            out["age_model_task"] = str(age_orig["task"][i])
            out["age_score_original"] = float(age_orig["raw"][i])
            out["age_score_rewritten"] = float(age_rewrite["raw"][i])
            out["delta_age_score"] = float(age_rewrite["raw"][i] - age_orig["raw"][i])
            out["older_prob_original"] = float(age_orig["older_prob"][i])
            out["older_prob_rewritten"] = float(age_rewrite["older_prob"][i])
            out["delta_older_prob"] = float(age_rewrite["older_prob"][i] - age_orig["older_prob"][i])
        if ppl_orig is not None and ppl_rewrite is not None:
            out["orig_perplexity"] = float(ppl_orig[i])
            out["rew_perplexity"] = float(ppl_rewrite[i])
            out["delta_perplexity"] = float(ppl_rewrite[i] - ppl_orig[i])
        out["rouge_l_f1"] = float(rouge_l_f1(orig_text, rew_text))
        out["bleu2"] = float(bleu2(orig_text, rew_text))
        out["orig_distinct1"] = float(distinct_n(orig_text, 1))
        out["orig_distinct2"] = float(distinct_n(orig_text, 2))
        out["rew_distinct1"] = float(distinct_n(rew_text, 1))
        out["rew_distinct2"] = float(distinct_n(rew_text, 2))
        out["delta_distinct1"] = float(out["rew_distinct1"] - out["orig_distinct1"])
        out["delta_distinct2"] = float(out["rew_distinct2"] - out["orig_distinct2"])
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
        "mean_rouge_l_f1": float(pd.to_numeric(out_df["rouge_l_f1"], errors="coerce").mean()),
        "mean_bleu2": float(pd.to_numeric(out_df["bleu2"], errors="coerce").mean()),
        "mean_rew_distinct1": float(pd.to_numeric(out_df["rew_distinct1"], errors="coerce").mean()),
        "mean_rew_distinct2": float(pd.to_numeric(out_df["rew_distinct2"], errors="coerce").mean()),
    }
    for c in [c for c in out_df.columns if c.startswith("delta_")]:
        summary[f"mean_{c}"] = float(pd.to_numeric(out_df[c], errors="coerce").mean())
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
