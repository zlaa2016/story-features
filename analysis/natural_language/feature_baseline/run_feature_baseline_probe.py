#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BIN_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*$")


def parse_bins(spec: str) -> List[Tuple[float, float]]:
    bins: List[Tuple[float, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = BIN_RE.match(part)
        if not m:
            raise ValueError(f"Bad bin: {part} (expected like 0-4)")
        lo, hi = float(m.group(1)), float(m.group(2))
        if hi < lo:
            lo, hi = hi, lo
        bins.append((lo, hi))
    if not bins:
        raise ValueError("No bins parsed.")
    return bins


def assign_bin(age_mid: float, bins: List[Tuple[float, float]]) -> Optional[int]:
    for i, (lo, hi) in enumerate(bins):
        if lo <= age_mid <= hi:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline: predict age group from non-LLM features only.")
    ap.add_argument("--features", type=Path, required=True, help="CSV from analyze_text_features.py")
    ap.add_argument("--out", type=Path, default=Path("results/analysis/natural_language/feature_baseline/feature_baseline.csv"))
    ap.add_argument("--bins", type=str, default="0-3,4-6,7-12")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument(
        "--cols",
        type=str,
        default="ttr,root_ttr,log_ttr,hapax_frac,vocab_types,mean_log10_corpus_freq,rare_frac_corpus_le_5,avg_syllables_per_word,flesch_kincaid_grade_est,flesch_reading_ease_est,mean_sentence_length_words,mean_word_length_chars,num_words,num_sentences,num_pronouns,sentiment_compound",
        help="Comma-separated feature columns to use if present.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.features, low_memory=False)
    df["age_mid"] = pd.to_numeric(df.get("age_mid"), errors="coerce")
    df = df[df["age_mid"].notna()].copy()

    bins = parse_bins(args.bins)
    df["y"] = df["age_mid"].apply(lambda a: assign_bin(float(a), bins) if not pd.isna(a) else None)
    df = df[df["y"].notna()].copy()
    df["y"] = df["y"].astype(int)

    col_list = [c.strip() for c in args.cols.split(",") if c.strip()]
    cols = [c for c in col_list if c in df.columns]
    if not cols:
        raise SystemExit("No requested feature columns found in the CSV.")

    Xdf = df[cols].apply(pd.to_numeric, errors="coerce")

    # Drop columns that are entirely NaN (e.g., sentiment if dependency not installed)
    non_empty_cols = [c for c in cols if Xdf[c].notna().any()]
    if not non_empty_cols:
        raise SystemExit("All selected feature columns are empty/NaN.")
    Xdf = Xdf[non_empty_cols]

    # Median-impute remaining NaNs
    med = Xdf.median(numeric_only=True)
    Xdf = Xdf.fillna(med)

    # Drop any rows still containing NaNs (e.g., columns with all-NaN medians)
    ok = Xdf.notna().all(axis=1)
    X = Xdf[ok].to_numpy(dtype=np.float32)
    y = df.loc[ok, "y"].to_numpy(dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(X_train_s, y_train)
    pred = clf.predict(X_test_s)

    out = pd.DataFrame(
        [
            {
                "n_rows": int(len(y)),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "n_classes": int(len(np.unique(y))),
                "accuracy": float(accuracy_score(y_test, pred)),
                "f1_macro": float(f1_score(y_test, pred, average="macro")) if len(np.unique(y_test)) > 1 else float("nan"),
                "bins": args.bins,
                "cols_used": ",".join(non_empty_cols),
            }
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
