"""How much do the principal components of one age's stories overlap with another age's, at several variance levels?

For each age a (2..12) take the stories whose stated age range covers a, run PCA on their story
embeddings, and keep the top k components that capture v of the variance (v = 50%, 80%, 95%).  The
overlap between two ages is  ||U_a[:k] U_b[:k]^T||_F^2 / k  with k = min(k_a, k_b): 1 means the
smaller subspace lies inside the larger, a random pair of k-dimensional subspaces gives about k/d.

Shown against age: the dimensionality k_a(v), the overlap of each age with the next age, and the
overlap of each age with the youngest age.

Every cell is cut to the same number of stories and every quantity is computed on random half-splits
(without replacement), so all of them use the same sample size:

* observed: overlap between a random half of age a and a random half of the other age; the band is the
  spread over repeated random splits;
* ceiling: overlap between the two halves of the *same* age (how similar two samples of one population
  look; the largest overlap sampling noise permits at this sample size);
* permuted-age band: the same analysis after the age ranges are shuffled among stories (content no
  longer tied to age, but wide-range stories still sit in many ages, so sharing between neighbouring
  ages is preserved).  Observed values outside this band are age-specific structure;
* chance: k/d, the overlap of random subspaces.

``--mode multi`` uses every story covering the age (as the flat subset is defined); ``--mode disjoint``
gives each story to one age (balanced matching) so neighbouring ages share no stories.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

base = importlib.import_module("analysis.representation.geometry.within_story_compression.analyze")

AGES = np.arange(2, 13)
LEVELS = (0.5, 0.8, 0.95)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_distilbert(path: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    ids = payload["item_id"].astype(str)
    vectors = payload["vectors"][:, layer, :].astype(np.float32)
    keep = ~pd.Series(ids).duplicated().to_numpy()
    return ids[keep], vectors[keep]


def load_gpt_oss_global(directory: Path, key: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    ids: list[str] = []
    rows: list[np.ndarray] = []
    for feature_path in sorted(directory.glob("global_features_*.npz")):
        suffix = feature_path.stem.removeprefix("global_features_")
        metadata = pd.read_csv(directory / f"global_metadata_{suffix}.csv", low_memory=False)
        with np.load(feature_path, allow_pickle=False) as payload:
            array = payload[key]
            layers = payload["layers"].astype(int).tolist()
            position = len(layers) - 1 if layer == -1 else layers.index(layer)
            chosen = np.flatnonzero(metadata["record_type"].astype(str).eq("original").to_numpy())
            rows.append(array[chosen, position, :].astype(np.float32))
        ids.extend(metadata.iloc[chosen]["video_id"].astype(str).tolist())
    vectors = np.concatenate(rows)
    keep = ~pd.Series(ids).duplicated().to_numpy()
    return np.asarray(ids)[keep], vectors[keep]


def story_table(ids: np.ndarray, vectors: np.ndarray, flat_csv: Path, standardize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = pd.read_csv(flat_csv, low_memory=False).drop_duplicates("video_id")
    flat["video_id"] = flat["video_id"].astype(str)
    frame = pd.DataFrame({"video_id": ids, "row": np.arange(len(ids))}).merge(flat[["video_id", "range_lower", "range_upper"]], on="video_id", how="inner")
    frame = frame.dropna(subset=["range_lower", "range_upper"])
    x = base.normalize_rows(vectors[frame["row"].to_numpy()]).astype(np.float64)
    if standardize:
        x = x / (x.std(axis=0) + 1e-6)
    return x, frame["range_lower"].to_numpy(dtype=int), frame["range_upper"].to_numpy(dtype=int)


# --------------------------------------------------------------------------------------
# Cells, PCs, overlaps
# --------------------------------------------------------------------------------------
def multi_cells(lo: np.ndarray, hi: np.ndarray, per_age: int, rng: np.random.Generator) -> dict[int, np.ndarray]:
    cells = {}
    for age in AGES:
        members = np.flatnonzero((lo <= age) & (hi >= age))
        cells[int(age)] = np.sort(rng.choice(members, per_age, replace=False)) if len(members) > per_age else members
    return cells


def disjoint_cells(lo: np.ndarray, hi: np.ndarray, per_age: int, rng: np.random.Generator) -> dict[int, np.ndarray]:
    assigned = base.assign_ages(np.clip(lo, AGES[0], AGES[-1]), np.clip(hi, AGES[0], AGES[-1]), AGES, per_age, rng)
    return {int(age): np.flatnonzero(assigned == age) for age in AGES}


def equal_cells(build: Callable[..., dict[int, np.ndarray]], lo: np.ndarray, hi: np.ndarray, per_age: int, rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Cells of exactly the same size for every age (the size of the smallest age, at most ``per_age``)."""
    first = build(lo, hi, per_age, rng)
    size = min(len(members) for members in first.values())
    return build(lo, hi, size, rng)


def principal_axes(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = rows - rows.mean(axis=0)
    try:
        _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as scipy_svd

        _, singular, axes = scipy_svd(centered, full_matrices=False, lapack_driver="gesvd")
    variance = singular**2
    return axes, np.cumsum(variance) / variance.sum()


def components_at(cumulative: np.ndarray, level: float) -> int:
    return int(np.searchsorted(cumulative, level - 1e-12) + 1)


def subspace_overlap(first: np.ndarray, second: np.ndarray, k: int) -> float:
    return float(np.linalg.norm(first[:k] @ second[:k].T) ** 2 / k)


def half_curves(x: np.ndarray, cells: dict[int, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One random split of every age into two halves; k(v), same-age ceiling, overlap with next age and with age 2."""
    halves = {}
    for age, members in cells.items():
        if len(members) < 20:
            continue
        order = rng.permutation(len(members))
        middle = len(order) // 2
        halves[age] = (principal_axes(x[members[order[:middle]]]), principal_axes(x[members[order[middle: 2 * middle]]]))
    shape = (len(AGES), len(LEVELS))
    out = {name: np.full(shape, np.nan) for name in ("k", "ceiling", "adjacent", "youngest")}
    first = int(AGES[0])
    for i, age in enumerate(AGES):
        if int(age) not in halves:
            continue
        (axes_a, cumulative_a), (axes_b, cumulative_b) = halves[int(age)]
        for j, level in enumerate(LEVELS):
            k_a, k_b = components_at(cumulative_a, level), components_at(cumulative_b, level)
            out["k"][i, j] = k_a
            out["ceiling"][i, j] = subspace_overlap(axes_a, axes_b, min(k_a, k_b))
            for target, name in ((int(age) + 1, "adjacent"), (first, "youngest")):
                if target in halves and target != int(age):
                    other_axes, other_cumulative = halves[target][1]
                    out[name][i, j] = subspace_overlap(axes_a, other_axes, min(k_a, components_at(other_cumulative, level)))
    return out


def summarise_draws(draws: list[dict[str, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {name: (np.nanmean(stack := np.stack([d[name] for d in draws]), axis=0), np.nanquantile(stack, 0.025, axis=0), np.nanquantile(stack, 0.975, axis=0))
            for name in draws[0]}


def run_mode(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, mode: str, per_age: int, splits: int, permutations: int, seed: int,
             log: Callable[[str], None] = print) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    build = multi_cells if mode == "multi" else disjoint_cells
    cells = equal_cells(build, lo, hi, per_age, rng)
    log(f"[{mode}] stories per age: {[len(cells[int(a)]) for a in AGES]}")
    observed = summarise_draws([half_curves(x, cells, rng) for _ in range(splits)])
    null_draws = []
    for _ in range(permutations):
        order = rng.permutation(len(lo))
        null_draws.append(half_curves(x, equal_cells(build, lo[order], hi[order], per_age, rng), rng))
    null = summarise_draws(null_draws) if null_draws else None
    return {"cells": {int(a): len(cells[int(a)]) for a in AGES}, "observed": observed, "null": null, "dimension": x.shape[1]}


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
def summary_table(result: dict[str, object], mode: str) -> pd.DataFrame:
    rows = []
    for i, age in enumerate(AGES):
        for j, level in enumerate(LEVELS):
            row = {"mode": mode, "age": int(age), "variance_level": level, "n_stories": result["cells"][int(age)]}
            for name in ("k", "ceiling", "adjacent", "youngest"):
                mean, low, high = result["observed"][name]
                row[f"{name}_mean"], row[f"{name}_lo"], row[f"{name}_hi"] = mean[i, j], low[i, j], high[i, j]
                if result["null"] is not None:
                    mean, low, high = result["null"][name]
                    row[f"null_{name}_mean"], row[f"null_{name}_lo"], row[f"null_{name}_hi"] = mean[i, j], low[i, j], high[i, j]
            row["chance_k_over_d"] = row["k_mean"] / result["dimension"]
            rows.append(row)
    return pd.DataFrame(rows)


def trend_report(result: dict[str, object]) -> dict[str, dict[str, float]]:
    """Spearman of each curve with age, observed and for the mean of the permuted-age null."""
    report: dict[str, dict[str, float]] = {}
    for name in ("k", "adjacent", "youngest"):
        for j, level in enumerate(LEVELS):
            entry = {}
            for label, bundle in (("observed", result["observed"]), ("permuted_age", result["null"])):
                if bundle is None:
                    continue
                values = bundle[name][0][:, j]
                ok = np.isfinite(values)
                entry[f"{label}_spearman_with_age"] = float(spearmanr(AGES[ok], values[ok]).statistic) if ok.sum() > 3 else float("nan")
            report[f"{name}@{int(level * 100)}%"] = entry
    return report


def plot(result: dict[str, object], path: Path, title: str, subtitle: str, note: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [("k", "components needed (k)"), ("adjacent", "overlap with the next age"), ("youngest", "overlap with age 2")]
    figure, axes = plt.subplots(3, 3, figsize=(15, 11.2), facecolor="#fcfcfb", sharex=True)
    for r, (name, ylabel) in enumerate(rows):
        for j, level in enumerate(LEVELS):
            axis = axes[r][j]
            mean, low, high = result["observed"][name]
            if result["null"] is not None:
                null_mean, null_low, null_high = result["null"][name]
                axis.fill_between(AGES, null_low[:, j], null_high[:, j], color="#c9c8c3", alpha=0.75, lw=0, label="permuted ages (no age signal)")
            axis.fill_between(AGES, low[:, j], high[:, j], color="#2a5db0", alpha=0.2, lw=0)
            axis.plot(AGES, mean[:, j], color="#2a5db0", lw=2.2, marker="o", ms=4, label="observed (random halves)")
            if name != "k":
                axis.plot(AGES, result["observed"]["ceiling"][0][:, j], color="#2b2b29", lw=1.4, ls=":", label="ceiling: two halves of the same age")
                axis.plot(AGES, result["observed"]["k"][0][:, j] / result["dimension"], color="#8a8a85", lw=1.0, ls="--", label="chance (k/d)")
                axis.set_ylim(0, 1.02)
            axis.set_xticks(AGES)
            if r == 0:
                axis.set_title(f"PCs capturing {int(level * 100)}% of the variance", fontsize=11, loc="left")
            if j == 0:
                axis.set_ylabel(ylabel, fontsize=9.5)
            if r == 2:
                axis.set_xlabel("age")
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
    axes[0][0].legend(fontsize=7.5, loc="best")
    axes[1][0].legend(fontsize=7.5, loc="lower left")
    figure.suptitle(f"{title}\n{subtitle}", fontsize=12, x=0.01, ha="left")
    figure.text(0.01, 0.008, note, fontsize=8.2, color="#6b6b66", wrap=True)
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(path, dpi=140, facecolor="#fcfcfb")
    plt.close(figure)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["distilbert", "gpt_oss_global"], default="distilbert")
    parser.add_argument("--npz", type=Path, default=Path("artifacts/distilbert_vectors_events/story_vectors.npz"), help="distilbert source: story_vectors.npz")
    parser.add_argument("--global-dir", type=Path, default=Path(f"{base.DEFAULT_ROOT}/gpt_oss_20b_global_current"), help="gpt_oss_global source: folder with global_features_*.npz")
    parser.add_argument("--vector-key", default="mean_vectors", help="gpt_oss_global: mean_vectors or eos_vectors")
    parser.add_argument("--layer", type=int, default=-1, help="distilbert: index into the 7 stored hidden states; gpt_oss: model layer, -1 = last stored")
    parser.add_argument("--flat-csv", type=Path, default=Path("data/processed/flat_videos_D_2-12.csv"))
    parser.add_argument("--standardize", action="store_true", help="divide each dimension by its standard deviation (tames rogue dimensions in LLM states)")
    parser.add_argument("--modes", default="multi", help="multi, disjoint or both; disjoint needs many stories (about 40 per age with 571 stories is too few to read)")
    parser.add_argument("--per-age", type=int, default=500, help="stories per age at most; all ages are cut to the smallest age")
    parser.add_argument("--splits", type=int, default=40, help="random half-splits for the observed curves")
    parser.add_argument("--permutations", type=int, default=30, help="permuted-age draws for the null band")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/representation/geometry/gist_structure/pc_overlap"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "distilbert":
        ids, vectors = load_distilbert(args.npz, args.layer)
        label = f"DistilBERT story vectors, hidden state {args.layer}"
    else:
        ids, vectors = load_gpt_oss_global(args.global_dir, args.vector_key, args.layer)
        label = f"GPT-OSS {args.vector_key}, layer {args.layer}"
    x, lo, hi = story_table(ids, vectors, args.flat_csv, args.standardize)
    print(f"{len(x)} stories with embeddings and an age range ({label}); dimension {x.shape[1]}", flush=True)
    tables, trends = [], {}
    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        result = run_mode(x, lo, hi, mode, args.per_age, args.splits, args.permutations, args.seed)
        tables.append(summary_table(result, mode))
        trends[mode] = trend_report(result)
        size = next(iter(result["cells"].values()))
        what = ("every story whose stated range covers the age, so stories sit in several ages" if mode == "multi"
                else "each story assigned to one age, so neighbouring ages share no stories")
        plot(result, args.output_dir / f"fig_pc_overlap_{mode}.png",
             f"PC overlap between ages at several variance levels ({label})",
             f"{size} stories per age ({size // 2} per half); {what}",
             "Overlap = ||U_a U_b^T||^2 / k, k = the smaller number of components; 1 = one subspace lies inside the other. All curves use random half-splits of equal size. "
             "Blue band: spread over repeated splits. Grey band: same analysis with age ranges shuffled among stories. Dotted: two halves of the same age.")
    pd.concat(tables, ignore_index=True).to_csv(args.output_dir / "pc_overlap_by_age.csv", index=False)
    (args.output_dir / "pc_overlap_trends.json").write_text(json.dumps(trends, indent=2))
    print(json.dumps(trends, indent=2))


if __name__ == "__main__":
    main()
