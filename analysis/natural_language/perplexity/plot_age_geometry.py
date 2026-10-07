#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def safe_name(value: str) -> str:
    out = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def rank_corr(x: Iterable[float], y: Iterable[float]) -> float:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 3:
        return float("nan")
    return float(tmp["x"].rank(method="average").corr(tmp["y"].rank(method="average")))


def pearson_corr(x: Iterable[float], y: Iterable[float]) -> float:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 3:
        return float("nan")
    return float(tmp["x"].corr(tmp["y"]))


def classical_mds(distance_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    n = distance_matrix.shape[0]
    d2 = distance_matrix**2
    center = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * center @ d2 @ center
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = np.maximum(eigvals[:n_components], 0)
    coords = eigvecs[:, :n_components] * np.sqrt(keep)
    if coords.shape[1] < n_components:
        coords = np.pad(coords, ((0, 0), (0, n_components - coords.shape[1])))
    return coords


def zscore_columns(frame: pd.DataFrame) -> pd.DataFrame:
    centered = frame.copy()
    for col in centered.columns:
        values = pd.to_numeric(centered[col], errors="coerce")
        sd = float(values.std(skipna=True))
        if not math.isfinite(sd) or sd == 0:
            centered[col] = np.nan
        else:
            centered[col] = (values - float(values.mean(skipna=True))) / sd
    return centered


def pairwise_age_distances(pivot: pd.DataFrame, min_shared_models: int) -> tuple[np.ndarray, pd.DataFrame]:
    ages = pivot.index.to_numpy(dtype=float)
    arr = pivot.to_numpy(dtype=float)
    n = len(ages)
    dist = np.zeros((n, n), dtype=float)
    rows = []
    for i, j in combinations(range(n), 2):
        valid = np.isfinite(arr[i]) & np.isfinite(arr[j])
        n_shared = int(valid.sum())
        if n_shared < min_shared_models:
            d = float("nan")
        else:
            d = float(np.sqrt(np.mean((arr[i, valid] - arr[j, valid]) ** 2)))
        dist[i, j] = dist[j, i] = d
        rows.append(
            {
                "age_a": ages[i],
                "age_b": ages[j],
                "age_gap_years": float(abs(ages[j] - ages[i])),
                "age_rank_gap": int(j - i),
                "adjacent_age_bin": int(j - i) == 1,
                "profile_distance": d,
                "n_shared_models": n_shared,
            }
        )
    pairwise = pd.DataFrame(rows)
    if np.isnan(dist).any():
        finite = dist[np.isfinite(dist) & (dist > 0)]
        fill = float(finite.max()) if len(finite) else 1.0
        dist = np.where(np.isfinite(dist), dist, fill)
        np.fill_diagonal(dist, 0.0)
    return dist, pairwise


def label_age(age: float) -> str:
    if float(age).is_integer():
        return str(int(age))
    return f"{age:g}"


def plot_geometry(
    coords: np.ndarray,
    ages: np.ndarray,
    pairwise: pd.DataFrame,
    out_base: Path,
    title: str,
    metric_label: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(np.min(ages)), vmax=float(np.max(ages)))

    ax = axes[0]
    ax.plot(coords[:, 0], coords[:, 1], color="0.72", linewidth=2.0, zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], s=52, c=ages, cmap=cmap, norm=norm, marker="D", edgecolor="white", linewidth=0.6, zorder=2)
    for idx, age in enumerate(ages):
        if idx in {0, len(ages) - 1} or age in {2, 4, 6, 8, 10, 12, 14, 16, 18}:
            ax.text(coords[idx, 0], coords[idx, 1], label_age(float(age)), fontsize=7, ha="center", va="center")
    ax.set_title("Age manifold (MDS)")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.spines[["top", "right"]].set_visible(False)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Median Amazon age")

    ax = axes[1]
    valid = pairwise[np.isfinite(pairwise["profile_distance"])].copy()
    adjacent = valid[valid["adjacent_age_bin"]]
    nonadjacent = valid[~valid["adjacent_age_bin"]]
    ax.scatter(nonadjacent["age_gap_years"], nonadjacent["profile_distance"], s=12, color="#4e79a7", alpha=0.22, label="Non-adjacent")
    ax.scatter(adjacent["age_gap_years"], adjacent["profile_distance"], s=18, color="#f28e2b", alpha=0.75, label="Adjacent")
    if len(valid) >= 3:
        x = valid["age_gap_years"].to_numpy(dtype=float)
        y = valid["profile_distance"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xs, slope * xs + intercept, color="0.2", linewidth=1.4)
        r = pearson_corr(x, y)
        rho = rank_corr(x, y)
        ax.text(0.04, 0.96, f"r={r:.2f}\nrho={rho:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.set_title("Scaled isometry check")
    ax.set_xlabel("Age distance (years)")
    ax.set_ylabel(f"Distance in {metric_label} profile")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    data = [adjacent["profile_distance"].dropna(), nonadjacent["profile_distance"].dropna()]
    ax.boxplot(data, labels=["Adjacent", "Non-adjacent"], showfliers=False, patch_artist=True, boxprops={"facecolor": "0.9", "edgecolor": "0.35"}, medianprops={"color": "black"})
    rng = np.random.default_rng(7)
    for xpos, values, color in [(1, data[0], "#f28e2b"), (2, data[1], "#4e79a7")]:
        if len(values):
            jitter = rng.normal(0, 0.035, len(values))
            ax.scatter(np.full(len(values), xpos) + jitter, values, s=10, alpha=0.35, color=color)
    adj_med = float(data[0].median()) if len(data[0]) else float("nan")
    non_med = float(data[1].median()) if len(data[1]) else float("nan")
    ratio = adj_med / non_med if non_med > 0 else float("nan")
    ax.text(0.04, 0.96, f"median ratio={ratio:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.set_title("Neighbor similarity")
    ax.set_ylabel(f"Distance in {metric_label} profile")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.savefig(out_base.with_suffix(".png"), dpi=220)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_model_curves(age_bins: pd.DataFrame, out_base: Path, metric_label: str, max_models: int) -> None:
    import matplotlib.pyplot as plt

    counts = age_bins.groupby("model_key")["age_median"].nunique().sort_values(ascending=False)
    model_keys = list(counts.head(max_models).index)
    n = len(model_keys)
    if n == 0:
        return
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.7 * nrows), sharex=True, constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, model_key in zip(axes_arr, model_keys):
        g = age_bins[age_bins["model_key"] == model_key].sort_values("age_median")
        ax.plot(g["age_median"], g["metric_median"], color="0.25", linewidth=1.2)
        ax.scatter(g["age_median"], g["metric_median"], c=g["age_median"], cmap="viridis", s=22, edgecolor="white", linewidth=0.4)
        ax.set_title(model_key, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel(metric_label)
    for ax in axes_arr[n:]:
        ax.axis("off")
    for ax in axes_arr[-ncols:]:
        ax.set_xlabel("Median Amazon age")
    fig.suptitle("Per-age analysis/natural_language/perplexity curves by model", fontsize=14)
    fig.savefig(out_base.with_suffix(".png"), dpi=220)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot story age geometry in the style of manifold-steering age figures.")
    ap.add_argument("--age-bins", type=Path, default=Path("results/analysis/natural_language/perplexity/analysis/age_bins_mean_nll.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("results/analysis/natural_language/perplexity/figures"))
    ap.add_argument("--metric-column", default="metric_median", choices=["metric_median", "metric_mean"])
    ap.add_argument("--metric-label", default="median mean NLL")
    ap.add_argument("--comparison-set", default=None, help="Optional comparison_set value to filter if present in the age-bin file.")
    ap.add_argument("--min-age-values-per-model", type=int, default=8)
    ap.add_argument("--min-shared-models", type=int, default=2)
    ap.add_argument("--max-curve-models", type=int, default=12)
    args = ap.parse_args()

    age_bins = pd.read_csv(args.age_bins, low_memory=False)
    if args.comparison_set and "comparison_set" in age_bins.columns:
        age_bins = age_bins[age_bins["comparison_set"].astype(str) == args.comparison_set].copy()
    if age_bins.empty:
        raise ValueError(f"No rows found in {args.age_bins}")

    required = {"age_median", "model_key", args.metric_column}
    missing = required - set(age_bins.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.age_bins}: {sorted(missing)}")

    age_bins["age_median"] = pd.to_numeric(age_bins["age_median"], errors="coerce")
    age_bins[args.metric_column] = pd.to_numeric(age_bins[args.metric_column], errors="coerce")
    age_bins = age_bins.dropna(subset=["age_median", "model_key", args.metric_column]).copy()
    age_bins = age_bins.rename(columns={args.metric_column: "plot_metric"})

    pivot = age_bins.pivot_table(index="age_median", columns="model_key", values="plot_metric", aggfunc="median").sort_index()
    enough = pivot.notna().sum(axis=0) >= args.min_age_values_per_model
    pivot = pivot.loc[:, enough]
    if pivot.shape[0] < 3 or pivot.shape[1] < args.min_shared_models:
        raise ValueError(
            f"Need at least 3 ages and {args.min_shared_models} models with enough age bins; "
            f"got {pivot.shape[0]} ages and {pivot.shape[1]} models."
        )
    pivot_z = zscore_columns(pivot).dropna(axis=1, how="all")

    dist, pairwise = pairwise_age_distances(pivot_z, min_shared_models=args.min_shared_models)
    coords = classical_mds(dist, n_components=2)
    ages = pivot_z.index.to_numpy(dtype=float)

    valid_pairs = pairwise[np.isfinite(pairwise["profile_distance"])].copy()
    adjacent = valid_pairs[valid_pairs["adjacent_age_bin"]]
    nonadjacent = valid_pairs[~valid_pairs["adjacent_age_bin"]]
    adj_med = float(adjacent["profile_distance"].median()) if len(adjacent) else float("nan")
    non_med = float(nonadjacent["profile_distance"].median()) if len(nonadjacent) else float("nan")
    summary = pd.DataFrame(
        [
            {
                "source_age_bins": str(args.age_bins),
                "comparison_set": args.comparison_set or "",
                "metric_column": args.metric_column,
                "n_age_values": len(ages),
                "n_models_used": pivot_z.shape[1],
                "models_used": ",".join(map(str, pivot_z.columns)),
                "spearman_age_gap_profile_distance": rank_corr(valid_pairs["age_gap_years"], valid_pairs["profile_distance"]),
                "pearson_age_gap_profile_distance": pearson_corr(valid_pairs["age_gap_years"], valid_pairs["profile_distance"]),
                "adjacent_profile_distance_median": adj_med,
                "nonadjacent_profile_distance_median": non_med,
                "adjacent_to_nonadjacent_median_ratio": adj_med / non_med if non_med > 0 else float("nan"),
                "adjacent_profile_distance_mean": float(adjacent["profile_distance"].mean()) if len(adjacent) else float("nan"),
                "nonadjacent_profile_distance_mean": float(nonadjacent["profile_distance"].mean()) if len(nonadjacent) else float("nan"),
            }
        ]
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = safe_name(args.comparison_set or args.metric_column)
    pairwise_path = args.outdir / f"age_geometry_pairwise_{suffix}.csv"
    summary_path = args.outdir / f"age_geometry_summary_{suffix}.csv"
    mds_path = args.outdir / f"age_geometry_mds_{suffix}"
    curves_path = args.outdir / f"age_geometry_model_curves_{suffix}"

    pairwise.to_csv(pairwise_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_geometry(coords, ages, valid_pairs, mds_path, "Story Age Geometry", args.metric_label)
    plot_model_curves(age_bins.rename(columns={"plot_metric": "metric_median"}), curves_path, args.metric_label, args.max_curve_models)

    print(f"Wrote {summary_path}")
    print(f"Wrote {pairwise_path}")
    print(f"Wrote {mds_path.with_suffix('.png')}")
    print(f"Wrote {mds_path.with_suffix('.pdf')}")
    print(f"Wrote {curves_path.with_suffix('.png')}")
    print(f"Wrote {curves_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
