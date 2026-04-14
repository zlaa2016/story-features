#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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
            raise ValueError(f"Bad bin: {part} (expected like 2-5)")
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
    ap = argparse.ArgumentParser(description="Layer-wise linear probing over extracted story vectors.")
    ap.add_argument("--vectors", type=Path, default=Path("artifacts/distilbert_vectors/story_vectors.npz"))
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/linear_probe"))
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--bins",
        type=str,
        default="0-3,4-6,7-12",
        help="Comma bins for age_mid like '0-3,4-6,7-12'.",
    )
    ap.add_argument("--shuffle-labels", action="store_true", help="Control: shuffle labels before training.")
    args = ap.parse_args()

    data = np.load(args.vectors, allow_pickle=True)
    vectors = data["vectors"]  # (n_items, n_layers, hidden)
    item_id = data["item_id"].astype(str).tolist()
    age_mid = data["age_mid"].astype(np.float32)

    if vectors.ndim != 3:
        raise ValueError(f"Unexpected vectors shape: {vectors.shape}")

    bins = parse_bins(args.bins) if args.bins else None
    labels: List[Optional[int]] = []
    for a in age_mid:
        if np.isnan(a):
            labels.append(None)
            continue
        if bins is None:
            labels.append(int(round(float(a) * 2)))  # coarse default: 0.5-year bins
        else:
            labels.append(assign_bin(float(a), bins))

    keep_idx = [i for i, y in enumerate(labels) if y is not None]
    if not keep_idx:
        raise SystemExit("No labeled examples (age_mid missing). Consider rebuilding master or passing different bins.")

    X = vectors[keep_idx]  # (n, n_layers, hidden)
    y = np.array([labels[i] for i in keep_idx], dtype=np.int64)
    kept_ids = [item_id[i] for i in keep_idx]

    if args.shuffle_labels:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(y)

    n_items, n_layers, hidden = X.shape
    args.outdir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    reports: Dict[str, Any] = {}

    for layer in range(n_layers):
        Xl = X[:, layer, :]
        X_train, X_test, y_train, y_test = train_test_split(
            Xl,
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

        acc = float(accuracy_score(y_test, pred))
        f1m = float(f1_score(y_test, pred, average="macro")) if len(np.unique(y_test)) > 1 else float("nan")
        cm = confusion_matrix(y_test, pred).tolist()
        rep = classification_report(y_test, pred, output_dict=True, zero_division=0)

        results.append(
            {
                "layer": layer,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "n_classes": int(len(np.unique(y))),
                "accuracy": acc,
                "f1_macro": f1m,
            }
        )
        reports[str(layer)] = {"confusion_matrix": cm, "report": rep}

    pd.DataFrame(results).to_csv(args.outdir / "probe_results.csv", index=False)
    (args.outdir / "probe_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    best = max(results, key=lambda r: (r["accuracy"], -r["layer"]))
    print(f"Wrote: {args.outdir / 'probe_results.csv'}")
    print(f"Best layer: {best['layer']}  acc={best['accuracy']:.4f}")


if __name__ == "__main__":
    main()
