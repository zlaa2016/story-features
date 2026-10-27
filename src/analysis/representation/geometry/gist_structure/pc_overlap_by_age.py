"""How do the principal components and the geometry of one age's stories relate to every other age's, as the variance captured changes?

For each age a (2..12) take the stories whose stated age range covers a and run PCA on their story
embeddings.  Three heatmaps put age on the x axis and the share of variance captured on the y axis:

1. number of PCs needed (k) to capture that share of the variance;
2. PC overlap: the mean over all other ages b of  ||U_a[:k] U_b[:k]^T||_F^2 / k  (k = the smaller of the
   two ages' component counts at that variance level).  1 means one subspace lies inside the other;
   random k-dimensional subspaces give about k/d;
3. geometry preservation (linear CKA): every story of the collection is projected onto age a's top k
   PCs and onto age b's top k PCs; CKA between those two projections of the *same* stories says whether
   the story space has the same geometry seen through the two ages' axes.  Unlike subspace overlap it
   weighs directions by the variance they carry.  (Plain CKA cannot compare two ages directly because
   they contain different stories; projecting one common set of stories removes that problem.)  CKA is
   close to 1 for any pair here, so geometry change is shown as 1 - CKA on a log scale.

Each heatmap has a companion that shows observed minus a permuted-age null: the same analysis after the
age ranges are shuffled among stories (content no longer tied to age, but wide-range stories still sit
in many ages, so sharing between ages is preserved).  Cells are cut to the same size and every number
comes from random half-splits without replacement.  The diagonal of the supplementary matrices (two
disjoint halves of one age) is not a ceiling in ``multi`` mode: ages with overlapping stated ranges share
stories, so neighbouring ages can exceed it.

``--modes multi`` uses every story covering the age (as the flat subset is defined); ``disjoint`` gives
each story to one age.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

base = importlib.import_module("analysis.representation.geometry.within_story_compression.analyze")

AGES = np.arange(2, 13)
V_K = np.round(np.arange(0.10, 0.991, 0.01), 2)
V_PAIR = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
SHOWN = (0.5, 0.8, 0.95)


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


def corner_sums(block: np.ndarray) -> np.ndarray:
    """S[i, j] = sum of squares of block[:i + 1, :j + 1]; the numerators for every k at once."""
    return np.cumsum(np.cumsum(block**2, axis=0), axis=1)


def levels_to_k(cumulative: np.ndarray, levels: np.ndarray | tuple[float, ...]) -> np.ndarray:
    return np.searchsorted(cumulative, np.asarray(levels) - 1e-12) + 1


def one_split(x: np.ndarray, centred_all: np.ndarray, cells: dict[int, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One random half-split of every age: k(v), overlap and CKA between every pair of ages at each v.

    Overlap and CKA compare a random half of age a with the *other* random half of age b, so the
    diagonal (a = b) is the same-age ceiling at this sample size.
    """
    pieces = {}
    for age, members in cells.items():
        if len(members) < 20:
            continue
        order = rng.permutation(len(members))
        middle = len(order) // 2
        first, cumulative_first = principal_axes(x[members[order[:middle]]])
        second, cumulative_second = principal_axes(x[members[order[middle: 2 * middle]]])
        projected = (centred_all @ first.T, centred_all @ second.T)
        pieces[int(age)] = {
            "axes": (first, second), "projected": projected, "k_fine": levels_to_k(cumulative_first, V_K),
            "k_pair": (levels_to_k(cumulative_first, V_PAIR), levels_to_k(cumulative_second, V_PAIR)),
            "self_sums": (corner_sums(projected[0].T @ projected[0]), corner_sums(projected[1].T @ projected[1])),
        }
    count, levels = len(AGES), len(V_PAIR)
    out = {"k": np.full((count, len(V_K)), np.nan), "overlap": np.full((count, count, levels), np.nan), "cka": np.full((count, count, levels), np.nan)}
    for i, age_a in enumerate(AGES):
        if int(age_a) not in pieces:
            continue
        a = pieces[int(age_a)]
        out["k"][i] = a["k_fine"]
        for j, age_b in enumerate(AGES):
            if int(age_b) not in pieces:
                continue
            b = pieces[int(age_b)]
            overlap_sums = corner_sums(a["axes"][0] @ b["axes"][1].T)
            cross_sums = corner_sums(a["projected"][0].T @ b["projected"][1])
            for v in range(levels):
                width = int(min(a["k_pair"][0][v], b["k_pair"][1][v], overlap_sums.shape[0], overlap_sums.shape[1]))
                out["overlap"][i, j, v] = overlap_sums[width - 1, width - 1] / width
                denominator = np.sqrt(a["self_sums"][0][width - 1, width - 1] * b["self_sums"][1][width - 1, width - 1])
                out["cka"][i, j, v] = cross_sums[width - 1, width - 1] / denominator if denominator > 0 else np.nan
    return out


def mean_over(draws: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.nanmean(np.stack([draw[name] for draw in draws]), axis=0) for name in draws[0]}


def run_mode(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, mode: str, per_age: int, splits: int, permutations: int, seed: int,
             log: Callable[[str], None] = print) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    build = multi_cells if mode == "multi" else disjoint_cells
    centred_all = x - x.mean(axis=0)
    cells = equal_cells(build, lo, hi, per_age, rng)
    log(f"[{mode}] stories per age: {[len(cells[int(a)]) for a in AGES]}")
    observed = mean_over([one_split(x, centred_all, cells, rng) for _ in range(splits)])
    null = None
    if permutations:
        draws = []
        for _ in range(permutations):
            order = rng.permutation(len(lo))
            draws.append(one_split(x, centred_all, equal_cells(build, lo[order], hi[order], per_age, rng), rng))
        null = mean_over(draws)
        null["k_sd"] = np.nanstd(np.stack([d["k"] for d in draws]), axis=0)
    return {"cells": {int(a): len(cells[int(a)]) for a in AGES}, "observed": observed, "null": null, "dimension": x.shape[1]}


def others_mean(matrix: np.ndarray) -> np.ndarray:
    """Mean over the other ages b != a of a (ages x ages x levels) matrix -> (ages x levels)."""
    masked = matrix.copy()
    for i in range(masked.shape[0]):
        masked[i, i] = np.nan
    return np.nanmean(masked, axis=1)


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
def tables(result: dict[str, object], mode: str) -> dict[str, pd.DataFrame]:
    observed, null = result["observed"], result["null"]
    k_rows, pair_rows = [], []
    for i, age in enumerate(AGES):
        for index, level in enumerate(V_K):
            row = {"mode": mode, "age": int(age), "variance": float(level), "k": observed["k"][i, index]}
            if null is not None:
                row["k_null_mean"] = null["k"][i, index]
                row["k_null_sd"] = null["k_sd"][i, index]
            k_rows.append(row)
        for v, level in enumerate(V_PAIR):
            for j, other in enumerate(AGES):
                row = {"mode": mode, "age": int(age), "other_age": int(other), "variance": float(level),
                       "overlap": observed["overlap"][i, j, v], "cka": observed["cka"][i, j, v]}
                if null is not None:
                    row["overlap_null_mean"], row["cka_null_mean"] = null["overlap"][i, j, v], null["cka"][i, j, v]
                pair_rows.append(row)
    return {"k_by_age_variance": pd.DataFrame(k_rows), "pairs_by_age_variance": pd.DataFrame(pair_rows)}


def heatmap(axis, matrix: np.ndarray, rows: list[str], title: str, cmap: str, diverging: bool = False, log: bool = False, vmin=None, vmax=None, label: str = ""):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if diverging:
        limit = max(float(np.nanmax(np.abs(matrix))), 1e-9)
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=-limit, vmax=limit)
    elif log:
        low = vmin if vmin is not None else max(float(np.nanmin(matrix)), 1.0)
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, norm=LogNorm(vmin=low, vmax=vmax if vmax is not None else float(np.nanmax(matrix))))
    else:
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(range(len(AGES)))
    axis.set_xticklabels([str(int(a)) for a in AGES], fontsize=8.5)
    step = max(1, len(rows) // 10)
    axis.set_yticks(range(0, len(rows), step))
    axis.set_yticklabels(rows[::step], fontsize=8.5)
    axis.set_xlabel("age")
    axis.set_ylabel("share of variance captured")
    axis.set_title(title, fontsize=10, loc="left")
    colorbar = plt.colorbar(image, ax=axis, shrink=0.9)
    colorbar.set_label(label, fontsize=8.5)
    return image


def plot_main(result: dict[str, object], path: Path, title: str, subtitle: str, note: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observed, null = result["observed"], result["null"]
    fine_rows = [f"{int(round(v * 100))}%" for v in V_K]
    pair_rows = [f"{int(round(v * 100))}%" for v in V_PAIR]
    figure, axes = plt.subplots(3, 2, figsize=(13.5, 14), facecolor="#fcfcfb")
    heatmap(axes[0][0], observed["k"].T, fine_rows, "1. number of PCs needed", "Blues", log=True, label="PCs (log scale)")
    heatmap(axes[1][0], others_mean(observed["overlap"]).T, pair_rows, "2. PC overlap with all other ages (mean)", "Blues", vmin=0, vmax=1, label="overlap")
    change = np.maximum(1.0 - others_mean(observed["cka"]), 1e-4)
    heatmap(axes[2][0], change.T, pair_rows, "3. geometry change vs all other ages (1 - mean linear CKA; 0 = preserved)", "Blues", log=True, vmin=float(np.nanmin(change)), vmax=float(np.nanmax(change)), label="1 - CKA (log scale)")
    if null is not None:
        heatmap(axes[0][1], (observed["k"] - null["k"]).T, fine_rows, "observed minus permuted-age null", "RdBu_r", diverging=True, label="PCs")
        heatmap(axes[1][1], (others_mean(observed["overlap"]) - others_mean(null["overlap"])).T, pair_rows, "observed minus permuted-age null", "RdBu_r", diverging=True, label="overlap")
        heatmap(axes[2][1], ((1.0 - others_mean(observed["cka"])) - (1.0 - others_mean(null["cka"]))).T, pair_rows, "observed minus permuted-age null (red = more geometry change)", "RdBu_r", diverging=True, label="1 - CKA")
    else:
        for row in axes:
            row[1].axis("off")
    figure.suptitle(f"{title}\n{subtitle}", fontsize=12, x=0.01, ha="left")
    figure.text(0.01, 0.006, note, fontsize=8.2, color="#6b6b66", wrap=True)
    figure.tight_layout(rect=(0, 0.03, 1, 0.945))
    figure.savefig(path, dpi=140, facecolor="#fcfcfb")
    plt.close(figure)


def plot_matrices(result: dict[str, object], path: Path, title: str) -> None:
    """Supplement: the full age x age matrices at three variance levels (diagonal = same-age ceiling)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observed = result["observed"]
    figure, axes = plt.subplots(2, len(SHOWN), figsize=(5 * len(SHOWN), 9.4), facecolor="#fcfcfb")
    for column, level in enumerate(SHOWN):
        v = V_PAIR.index(level)
        for row, (name, label) in enumerate((("overlap", "PC overlap"), ("cka", "geometry change, 1 - CKA"))):
            axis = axes[row][column]
            if name == "cka":
                from matplotlib.colors import LogNorm

                values = np.maximum(1.0 - observed[name][:, :, v], 1e-4)
                image = axis.imshow(values, origin="lower", cmap="Blues", norm=LogNorm(vmin=1e-3, vmax=0.1))
            else:
                image = axis.imshow(observed[name][:, :, v], origin="lower", cmap="Blues", vmin=0, vmax=1)
            axis.set_xticks(range(len(AGES)))
            axis.set_xticklabels([str(int(a)) for a in AGES], fontsize=8)
            axis.set_yticks(range(len(AGES)))
            axis.set_yticklabels([str(int(a)) for a in AGES], fontsize=8)
            axis.set_xlabel("other age")
            axis.set_ylabel("age")
            axis.set_title(f"{label} at {int(level * 100)}% of the variance", fontsize=10, loc="left")
            plt.colorbar(image, ax=axis, shrink=0.85)
    figure.suptitle(title, fontsize=12, x=0.01, ha="left")
    figure.text(0.01, 0.008, "Diagonal: two disjoint random halves of the same age. Off-diagonal: a half of one age vs a half of another; ages whose stated ranges overlap share stories, so neighbours can exceed the diagonal.", fontsize=8.2, color="#6b6b66")
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
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
    parser.add_argument("--modes", default="multi", help="multi, disjoint or both; disjoint needs many stories")
    parser.add_argument("--per-age", type=int, default=500, help="stories per age at most; all ages are cut to the smallest age")
    parser.add_argument("--splits", type=int, default=30, help="random half-splits averaged for the observed heatmaps")
    parser.add_argument("--permutations", type=int, default=20, help="permuted-age draws for the null")
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
    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        result = run_mode(x, lo, hi, mode, args.per_age, args.splits, args.permutations, args.seed)
        for name, table in tables(result, mode).items():
            table.to_csv(args.output_dir / f"{name}_{mode}.csv", index=False)
        size = next(iter(result["cells"].values()))
        what = ("every story whose stated range covers the age, so stories sit in several ages" if mode == "multi"
                else "each story assigned to one age, so ages share no stories")
        plot_main(result, args.output_dir / f"fig_age_by_variance_{mode}.png",
                  f"Age by variance captured ({label})", f"{size} stories per age ({size // 2} per half); {what}",
                  "PC overlap = ||U_a U_b^T||^2 / k (k = smaller component count; 1 = one subspace inside the other); geometry change = 1 - linear CKA between all stories projected through each age's top k PCs. "
                  "Left: mean over all other ages. Right: observed minus the same analysis with age ranges shuffled among stories. Rows at or below about 30% variance use only 1-3 components, so differences there are coarse (integer k).")
        plot_matrices(result, args.output_dir / f"fig_age_by_age_matrices_{mode}.png", f"Age x age overlap and CKA at 50 / 80 / 95% of the variance ({label}; {size} stories per age)")
        observed = result["observed"]
        print(json.dumps({"mode": mode, "mean_k_at_80pct": float(np.nanmean(observed["k"][:, list(V_K).index(0.8)])),
                          "mean_other_age_overlap_at_80pct": float(np.nanmean(others_mean(observed["overlap"])[:, V_PAIR.index(0.8)])),
                          "mean_other_age_cka_at_80pct": float(np.nanmean(others_mean(observed["cka"])[:, V_PAIR.index(0.8)]))}))


if __name__ == "__main__":
    main()
