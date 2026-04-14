#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer-wise linear regression probes on vector activations.")
    ap.add_argument("--vectors", type=Path, required=True, help="NPZ from extract_vectors_* (vectors, item_id).")
    ap.add_argument("--features", type=Path, required=True, help="CSV with item_id and target column.")
    ap.add_argument("--target", type=str, required=True, help="Target column name (continuous).")
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/layerwise_regression_probe"))
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha.")
    args = ap.parse_args()

    vec = np.load(args.vectors, allow_pickle=True)
    X = vec["vectors"]  # (n_items, n_layers, hidden)
    item_id = vec["item_id"].astype(str)

    df = pd.read_csv(args.features, low_memory=False)
    if "item_id" not in df.columns:
        raise SystemExit("features CSV must include item_id")
    if args.target not in df.columns:
        raise SystemExit(f"Missing target column: {args.target}")

    df = df[["item_id", args.target]].copy()
    df["item_id"] = df["item_id"].astype(str)
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df = df[df[args.target].notna()].copy()

    # Join by item_id
    idx_map = {k: i for i, k in enumerate(item_id.tolist())}
    keep_rows: List[int] = []
    y_vals: List[float] = []
    for r in df.itertuples(index=False):
        i = idx_map.get(r.item_id)
        if i is None:
            continue
        keep_rows.append(i)
        y_vals.append(float(getattr(r, args.target)))

    if not keep_rows:
        raise SystemExit("No overlap between vectors item_id and features item_id.")

    Xk = X[keep_rows]  # (n, n_layers, hidden)
    y = np.array(y_vals, dtype=np.float32)

    n, n_layers, hidden = Xk.shape
    args.outdir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for layer in range(n_layers):
        Xl = Xk[:, layer, :]
        X_train, X_test, y_train, y_test = train_test_split(
            Xl, y, test_size=args.test_size, random_state=args.seed
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=args.alpha, random_state=args.seed)
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)

        results.append(
            {
                "layer": int(layer),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "r2": float(r2_score(y_test, pred)),
                "mae": float(mean_absolute_error(y_test, pred)),
            }
        )

    out_csv = args.outdir / f"probe_{args.target}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()

