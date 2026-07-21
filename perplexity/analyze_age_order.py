#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from perplexity.models import COMPARISON_SETS
except ModuleNotFoundError:
    from models import COMPARISON_SETS


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


def pairwise_age_similarity(age_bins: pd.DataFrame, group_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    summary_rows = []
    for group_key, g in age_bins.groupby(group_cols, dropna=False):
        values = (
            g[["age_median", "metric_median", "metric_mean", "n_stories"]]
            .dropna(subset=["age_median", "metric_median"])
            .sort_values("age_median")
            .reset_index(drop=True)
        )
        if len(values) < 3:
            continue

        metadata = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,)))
        pairs = []
        for i, j in combinations(range(len(values)), 2):
            left = values.iloc[i]
            right = values.iloc[j]
            age_gap = float(right["age_median"] - left["age_median"])
            age_rank_gap = int(j - i)
            abs_metric_diff = abs(float(right["metric_median"]) - float(left["metric_median"]))
            row = {
                **metadata,
                "age_a": float(left["age_median"]),
                "age_b": float(right["age_median"]),
                "age_gap_years": age_gap,
                "age_rank_gap": age_rank_gap,
                "adjacent_age_bin": age_rank_gap == 1,
                "metric_median_a": float(left["metric_median"]),
                "metric_median_b": float(right["metric_median"]),
                "abs_metric_median_diff": abs_metric_diff,
                "signed_metric_median_diff": float(right["metric_median"]) - float(left["metric_median"]),
                "n_stories_a": int(left["n_stories"]),
                "n_stories_b": int(right["n_stories"]),
            }
            pairs.append(row)
            pair_rows.append(row)

        pair_df = pd.DataFrame(pairs)
        adjacent = pair_df[pair_df["adjacent_age_bin"]]
        nonadjacent = pair_df[~pair_df["adjacent_age_bin"]]
        nearest_adjacent_hits = []
        for age in values["age_median"].astype(float):
            age_pairs = pair_df[(pair_df["age_a"] == age) | (pair_df["age_b"] == age)].copy()
            if age_pairs.empty:
                continue
            nearest = age_pairs.sort_values(["abs_metric_median_diff", "age_rank_gap"]).iloc[0]
            nearest_adjacent_hits.append(bool(nearest["adjacent_age_bin"]))

        adjacent_median = float(adjacent["abs_metric_median_diff"].median()) if len(adjacent) else float("nan")
        nonadjacent_median = float(nonadjacent["abs_metric_median_diff"].median()) if len(nonadjacent) else float("nan")
        adjacent_mean = float(adjacent["abs_metric_median_diff"].mean()) if len(adjacent) else float("nan")
        nonadjacent_mean = float(nonadjacent["abs_metric_median_diff"].mean()) if len(nonadjacent) else float("nan")
        summary_rows.append(
            {
                **metadata,
                "n_age_values": len(values),
                "n_age_pairs": len(pair_df),
                "n_adjacent_pairs": len(adjacent),
                "n_nonadjacent_pairs": len(nonadjacent),
                "spearman_age_gap_abs_metric_diff": rank_corr(pair_df["age_gap_years"], pair_df["abs_metric_median_diff"]),
                "spearman_age_rank_gap_abs_metric_diff": rank_corr(pair_df["age_rank_gap"], pair_df["abs_metric_median_diff"]),
                "adjacent_abs_diff_median": adjacent_median,
                "nonadjacent_abs_diff_median": nonadjacent_median,
                "adjacent_to_nonadjacent_median_ratio": adjacent_median / nonadjacent_median if nonadjacent_median > 0 else float("nan"),
                "adjacent_abs_diff_mean": adjacent_mean,
                "nonadjacent_abs_diff_mean": nonadjacent_mean,
                "adjacent_to_nonadjacent_mean_ratio": adjacent_mean / nonadjacent_mean if nonadjacent_mean > 0 else float("nan"),
                "nearest_age_bin_is_adjacent_share": float(np.mean(nearest_adjacent_hits)) if nearest_adjacent_hits else float("nan"),
            }
        )

    return pd.DataFrame(pair_rows), pd.DataFrame(summary_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze whether story perplexity tracks continuous Amazon median age.")
    ap.add_argument("--runs", type=Path, nargs="+", default=[Path("artifacts/perplexity/runs")])
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/perplexity/analysis"))
    ap.add_argument("--metric", type=str, default="mean_nll", choices=["mean_nll", "log10_perplexity", "perplexity"])
    ap.add_argument("--comparison-set", choices=sorted(COMPARISON_SETS), default=None, help="Analyze only one named model comparison set.")
    args = ap.parse_args()

    paths: list[Path] = []
    for p in args.runs:
        if p.is_dir():
            paths.extend(sorted(p.glob("*.csv")))
        else:
            paths.append(p)
    df = read_runs(paths)
    if args.comparison_set:
        keys = set(COMPARISON_SETS[args.comparison_set])
        df = df[df["model_key"].astype(str).isin(keys)].copy()
        if df.empty:
            raise ValueError(f"No run rows matched comparison set: {args.comparison_set}")
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
    if args.comparison_set:
        summary.insert(0, "comparison_set", args.comparison_set)
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
    if args.comparison_set:
        age_bins.insert(0, "comparison_set", args.comparison_set)

    pairwise, similarity = pairwise_age_similarity(age_bins, group_cols)
    if args.comparison_set and not pairwise.empty:
        pairwise.insert(0, "comparison_set", args.comparison_set)
    if args.comparison_set and not similarity.empty:
        similarity.insert(0, "comparison_set", args.comparison_set)

    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.comparison_set}_{args.metric}" if args.comparison_set else args.metric
    summary_path = args.outdir / f"age_order_summary_{suffix}.csv"
    bins_path = args.outdir / f"age_bins_{suffix}.csv"
    pairwise_path = args.outdir / f"age_pairwise_similarity_{suffix}.csv"
    similarity_path = args.outdir / f"age_similarity_summary_{suffix}.csv"
    summary.to_csv(summary_path, index=False)
    age_bins.to_csv(bins_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    similarity.to_csv(similarity_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {bins_path}")
    print(f"Wrote {pairwise_path}")
    print(f"Wrote {similarity_path}")


if __name__ == "__main__":
    main()
