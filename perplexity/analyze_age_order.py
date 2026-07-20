#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def rank_corr(x: pd.Series, y: pd.Series) -> float:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 3:
        return float("nan")
    return float(tmp["x"].rank(method="average").corr(tmp["y"].rank(method="average")))


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 3:
        return float("nan")
    return float(tmp["x"].corr(tmp["y"]))


def adjacent_agreement(age: pd.Series, metric: pd.Series, direction: str) -> float:
    tmp = pd.DataFrame({"age": age, "metric": metric}).dropna().sort_values(["age", "metric"])
    if len(tmp) < 3:
        return float("nan")
    by_age = tmp.groupby("age", as_index=False)["metric"].median().sort_values("age")
    if len(by_age) < 3:
        return float("nan")
    deltas = by_age["metric"].diff().dropna()
    if direction == "increasing":
        return float((deltas >= 0).mean())
    if direction == "decreasing":
        return float((deltas <= 0).mean())
    raise ValueError(direction)


def fit_slope(age: pd.Series, metric: pd.Series) -> float:
    tmp = pd.DataFrame({"age": age, "metric": metric}).dropna()
    if len(tmp) < 3:
        return float("nan")
    slope, _ = np.polyfit(tmp["age"].astype(float), tmp["metric"].astype(float), deg=1)
    return float(slope)


def read_runs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("No run CSVs found.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze whether story perplexity tracks continuous Amazon median age.")
    ap.add_argument("--runs", type=Path, nargs="+", default=[Path("artifacts/perplexity/runs")])
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/perplexity/analysis"))
    ap.add_argument("--metric", type=str, default="mean_nll", choices=["mean_nll", "log10_perplexity", "perplexity"])
    args = ap.parse_args()

    paths: list[Path] = []
    for p in args.runs:
        if p.is_dir():
            paths.extend(sorted(p.glob("*.csv")))
        else:
            paths.append(p)
    df = read_runs(paths)
    df["age_median"] = pd.to_numeric(df["age_median"], errors="coerce")
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    scored = df[df["age_median"].notna() & df[args.metric].notna()].copy()

    group_cols = ["model_key", "model_name", "family", "variant", "quantization"]
    rows = []
    for group_key, g in df.groupby(group_cols, dropna=False):
        model_key, model_name, family, variant, quantization = group_key
        g_scored = g[g["age_median"].notna() & g[args.metric].notna()]
        rho = rank_corr(g_scored["age_median"], g_scored[args.metric])
        r = pearson_corr(g_scored["age_median"], g_scored[args.metric])
        slope = fit_slope(g_scored["age_median"], g_scored[args.metric])
        single_window_share = (
            float(pd.to_numeric(g["fits_single_window"], errors="coerce").fillna(0).mean())
            if "fits_single_window" in g.columns
            else 1.0
        )
        rows.append(
            {
                "model_key": model_key,
                "model_name": model_name,
                "family": family,
                "variant": variant,
                "quantization": quantization,
                "metric": args.metric,
                "n_rows": len(g),
                "n_scored": len(g_scored),
                "n_unscored_or_skipped": len(g) - len(g_scored),
                "n_age_values": g_scored["age_median"].nunique(),
                "single_window_share": single_window_share,
                "spearman_age_metric": rho,
                "pearson_age_metric": r,
                "linear_slope_per_age_year": slope,
                "adjacent_increasing_share": adjacent_agreement(g_scored["age_median"], g_scored[args.metric], "increasing"),
                "adjacent_decreasing_share": adjacent_agreement(g_scored["age_median"], g_scored[args.metric], "decreasing"),
                "median_metric": float(g_scored[args.metric].median()) if len(g_scored) else float("nan"),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["family", "model_key"])
    age_bins = (
        scored.groupby(group_cols + ["age_median"], dropna=False)
        .agg(
            n_stories=("item_id", "count"),
            metric_mean=(args.metric, "mean"),
            metric_median=(args.metric, "median"),
            metric_std=(args.metric, "std"),
            token_median=("n_tokens", "median"),
        )
        .reset_index()
        .sort_values(["model_key", "age_median"])
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / f"age_order_summary_{args.metric}.csv"
    bins_path = args.outdir / f"age_bins_{args.metric}.csv"
    summary.to_csv(summary_path, index=False)
    age_bins.to_csv(bins_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {bins_path}")


if __name__ == "__main__":
    main()
