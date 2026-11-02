"""Spectral change from details to gist by age, one pipeline for any model and any saved batch.

Three views of how fast variance collapses onto a few principal components, each drawn as the same
age x share-of-variance heatmap (number of PCs needed, log colour, white contours):

A. pooled      one vector per story (whole-story state); the stories whose stated range covers age a
               are pooled and their PCs counted.  Between-story structure.
B. within      each story's own detail vectors (units x dimensions) are analysed alone; the number of
               PCs that story needs for each share of variance is averaged over the stories at age a.
               "Collapses onto few gists" = few PCs reach a high share of that story's variance.
C. within-pooled  every story's details are centred on that story's own mean (the story's theme is
               removed) and pooled across the stories at age a; PCs of the pooled deviations are the
               age's shared gist axes.  PC overlap between ages (half-split, permuted-age null) says
               whether those axes are the same across ages.

Other reductions reported next to PCA: participation ratio, entropy effective rank, spectral decay
exponent and the TwoNN intrinsic dimension (nonlinear manifold dimension).

Input: a batch folder from ``subsets.py select`` (``subset_videos.csv`` + ``membership.csv``).  With
``--reselect-available`` the batch is rebuilt on the videos that really have vectors and the batch
actually run on is written to ``<output-dir>/selection/``.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.representation.geometry.spectral_age import subsets

overlap = importlib.import_module("analysis.representation.geometry.gist_structure.pc_overlap_by_age")
base = overlap.base

# Folders read per model (all that exist are combined, the first folder wins when a video is in several).  The
# "spectral" folders hold the batch videos extracted by submit_extraction.sh.
MODEL_DIRS = {
    "gpt_oss_20b": {"global": ["gpt_oss_20b_global_current", "gpt_oss_20b_spectral_global"],
                    "isolated": ["gpt_oss_20b_detail_controls", "gpt_oss_20b_spectral_detail_controls"]},
    "qwen3_8b": {"global": ["qwen3_8b_spectral_global"], "isolated": ["qwen3_8b_spectral_detail_controls"]},
}
STATS = ["k50", "k80", "k90", "kn50", "kn80", "kn90", "k90_over_k50", "pr", "effrank", "alpha", "head3", "twonn"]


# --------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------
def usable(folders: list[Path]) -> list[Path]:
    return [folder for folder in folders if folder.exists() and any(folder.glob("*_metadata_*.csv"))]


def whole_story(folders: list[Path], key: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    ids: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    for folder in usable(folders):
        found_ids, found_vectors = overlap.load_gpt_oss_global(folder, key, layer)
        ids.append(found_ids)
        vectors.append(found_vectors)
    if not ids:
        raise FileNotFoundError(f"No whole-story vectors under {[str(f) for f in folders]}")
    all_ids, all_vectors = np.concatenate(ids), np.concatenate(vectors)
    keep = ~pd.Series(all_ids).duplicated().to_numpy()
    return all_ids[keep], all_vectors[keep]


def detail_table(folders: list[Path], view: str) -> pd.DataFrame:
    parts = []
    for index, folder in enumerate(usable(folders)):
        try:
            table = base.load_metadata(folder, view)
        except FileNotFoundError:
            continue
        table["folder"] = str(folder)
        parts.append(table)
    if not parts:
        raise FileNotFoundError(f"No {view} detail vectors under {[str(f) for f in folders]}")
    table = pd.concat(parts, ignore_index=True)
    table["video_id"] = table["video_id"].astype(str)
    first_folder = table.groupby("video_id")["folder"].transform("first")
    return table[table["folder"] == first_folder].reset_index(drop=True)


def detail_vectors(details: pd.DataFrame, view: str, layer: int, key: str | None) -> np.ndarray:
    out: np.ndarray | None = None
    for folder, group in details.groupby("folder", sort=False):
        found = base.load_vectors(Path(folder), view, group, layer, key)
        if out is None:
            out = np.zeros((len(details), found.shape[1]), dtype=np.float32)
        out[details.index.get_indexer(group.index)] = found
    return out


def twonn(rows: np.ndarray, limit: int = 1500, seed: int = 0) -> float:
    """TwoNN intrinsic dimension (Facco et al.): maximum likelihood on the ratio of the two nearest-neighbour distances."""
    if len(rows) > limit:
        rows = rows[np.random.default_rng(seed).choice(len(rows), limit, replace=False)]
    if len(rows) < 8:
        return float("nan")
    squared = np.sum(rows**2, axis=1)
    distance = np.maximum(squared[:, None] + squared[None, :] - 2.0 * rows @ rows.T, 0.0)
    np.fill_diagonal(distance, np.inf)
    nearest = np.sort(distance, axis=1)[:, :2]
    nearest = nearest[nearest[:, 0] > 1e-12]
    mu = np.sqrt(nearest[:, 1] / nearest[:, 0])
    mu = mu[mu > 1.0 + 1e-9]
    return float(len(mu) / np.log(mu).sum()) if len(mu) > 5 else float("nan")


def spectrum_row(rows: np.ndarray) -> dict[str, float]:
    """All spectrum statistics of one matrix of (already normalised and scaled) unit vectors."""
    n = len(rows)
    gram = base.center_gram(rows @ rows.T)
    values = base.eigenvalues(gram / (n - 1))[: n - 1]
    total = float(values.sum())
    out: dict[str, float] = {"n_units": float(n)}
    if total <= 0:
        return {**out, **{f"k_{int(round(v * 100))}": np.nan for v in overlap.V_K}}
    cumulative = np.cumsum(values) / total
    ks = overlap.levels_to_k(cumulative, overlap.V_K)
    for v, k in zip(overlap.V_K, ks):
        out[f"k_{int(round(v * 100))}"] = float(k)
    stats = base.spectrum_stats(values, n)
    for name in ("pr", "effrank", "alpha", "head3"):
        out[name] = stats[name]
    for pct in (50, 80, 90):
        out[f"k{pct}"] = out[f"k_{pct}"]
        out[f"kn{pct}"] = out[f"k_{pct}"] / (n - 1)
    out["k90_over_k50"] = out["k_90"] / max(out["k_50"], 1.0)
    out["twonn"] = twonn(rows)
    return out


def axes_cumulative(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return overlap.principal_axes(rows)


def heat(axis, matrix: np.ndarray, ages: np.ndarray, title: str) -> None:
    overlap.AGES = ages
    rows = [f"{int(round(v * 100))}%" for v in overlap.V_K]
    overlap.heatmap(axis, matrix.T, rows, title, "viridis", log=True, label="PCs (log scale)", contours=(2, 5, 10, 25, 50, 100))


# --------------------------------------------------------------------------------------
# Selection and loading
# --------------------------------------------------------------------------------------
def available_videos(args: argparse.Namespace, dirs: dict[str, Path]) -> set[str]:
    ids, _ = whole_story(dirs["global"], args.vector_key, args.layer)
    found = set(ids.tolist())
    if args.run_within:
        found &= set(detail_table(dirs[args.detail_view], args.detail_view)["video_id"])
    return found


def prepare_selection(args: argparse.Namespace, dirs: dict[str, Path], ages: list[int]) -> Path:
    out = args.output_dir / "selection"
    if not args.reselect_available:
        out.mkdir(parents=True, exist_ok=True)
        for name in ("subset_videos.csv", "membership.csv", "summary.json"):
            (out / name).write_bytes((args.batch_dir / name).read_bytes())
        return out
    baseline = pd.read_csv(args.baseline_csv, low_memory=False, dtype={"video_id": str})
    available = available_videos(args, dirs)
    membership, subset, summary = subsets.select_flat(baseline, ages, args.per_age, available, equal=True)
    summary["videos_with_vectors"] = len(available)
    subsets.write_batch(out, membership, subset, summary)
    return out


def popularity_for(ids: np.ndarray, subset: pd.DataFrame) -> np.ndarray:
    lookup = subset.drop_duplicates("video_id").set_index("video_id")["log_views"]
    return pd.Series(ids).map(lookup).to_numpy(dtype=float)


# --------------------------------------------------------------------------------------
# A. pooled whole-story spectrum
# --------------------------------------------------------------------------------------
def run_pooled(args, dirs, selection: Path, ages: np.ndarray, label: str) -> dict:
    subset = pd.read_csv(selection / "subset_videos.csv", dtype={"video_id": str})
    ids, vectors = whole_story(dirs["global"], args.vector_key, args.layer)
    x, lo, hi, pop = overlap.story_table(ids, vectors, selection / "subset_videos.csv", args.standardize, popularity_for(ids, subset))
    overlap.AGES = ages
    out = args.output_dir / "pooled"
    out.mkdir(parents=True, exist_ok=True)
    result = overlap.run_mode(x, lo, hi, "multi", args.per_age, args.splits, args.permutations, args.seed, pop=pop)
    for name, table in overlap.tables(result, "multi").items():
        table.to_csv(out / f"{name}.csv", index=False)
    size = next(iter(result["cells"].values()))
    overlap.plot_main(result, out / "fig_age_by_variance.png", f"Age by variance captured, pooled stories ({label})",
                      f"{size} most-viewed stories per age ({size // 2} per half), {result['unique_stories']} distinct stories",
                      "A. One whole-story vector per story; each age pools the stories whose stated range covers it.")
    rows = []
    cells = overlap.equal_cells(overlap.multi_cells, lo, hi, args.per_age, np.random.default_rng(args.seed), pop)
    for age, members in cells.items():
        _, cumulative = overlap.principal_axes(x[members])
        values = np.diff(np.concatenate([[0.0], cumulative]))
        stats = base.spectrum_stats(values * 1.0, len(members))
        rows.append({"age": age, "n_stories": len(members), "pr": stats["pr"], "effrank": stats["effrank"], "alpha": stats["alpha"],
                     "k80": int(overlap.components_at(cumulative, 0.8)), "twonn": twonn(x[members])})
    pd.DataFrame(rows).to_csv(out / "other_reductions_by_age.csv", index=False)
    return {"k": result["observed"]["k"], "cells": result["cells"]}


# --------------------------------------------------------------------------------------
# B and C. within-story spectrum and pooled within-story deviations
# --------------------------------------------------------------------------------------
def thinned_details(details: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if maximum <= 0:
        return details
    parts = []
    for _, group in details.sort_values(["story_key", "detail_index"]).groupby("story_key", sort=False):
        parts.append(group if len(group) <= maximum else group.iloc[np.unique(np.round(np.linspace(0, len(group) - 1, maximum)).astype(int))])
    return pd.concat(parts, ignore_index=True)


def story_cells(stories: pd.DataFrame, membership: pd.DataFrame, ages: np.ndarray) -> dict[int, np.ndarray]:
    index = {video: np.flatnonzero(stories["video_id"].astype(str).to_numpy() == video) for video in stories["video_id"].astype(str).unique()}
    cells = {}
    for age in ages:
        videos = membership.loc[membership["age"] == age].sort_values("rank")["video_id"].astype(str)
        cells[int(age)] = np.concatenate([index[v] for v in videos if v in index] or [np.array([], dtype=int)]).astype(int)
    return cells


def run_within(args, dirs, selection: Path, ages: np.ndarray, label: str) -> dict:
    overlap.AGES = ages
    membership = pd.read_csv(selection / "membership.csv", dtype={"video_id": str})
    details = detail_table(dirs[args.detail_view], args.detail_view)
    stories = base.build_story_table(details, selection / "subset_videos.csv", int(ages.min()), int(ages.max()), args.min_units)
    stories = stories[stories["video_id"].isin(membership["video_id"])].reset_index(drop=True)
    details = thinned_details(details[details["story_key"].isin(stories["story_key"])], args.max_units)
    vectors = base.normalize_rows(detail_vectors(details, args.detail_view, args.layer, args.detail_key))
    scale = vectors[np.random.default_rng(args.seed).choice(len(vectors), min(len(vectors), 60000), replace=False)].std(axis=0) + 1e-6 if args.standardize else np.ones(vectors.shape[1])
    out = args.output_dir / f"within_{args.detail_view}"
    out.mkdir(parents=True, exist_ok=True)
    cells = story_cells(stories, membership, ages)
    print(f"[within:{args.detail_view}] {len(stories)} stories, {len(details)} details; stories per age {[len(cells[int(a)]) for a in ages]}", flush=True)
    rng = np.random.default_rng(args.seed)
    profiles, pooled_rows, row_story = [], [], []
    story_position = {key: i for i, key in enumerate(stories["story_key"])}
    grouped = details.groupby("story_key", sort=False).indices
    for key, rows_index in grouped.items():
        z = vectors[rows_index] / scale
        profile = {"story_key": key, "video_id": stories.loc[story_position[key], "video_id"], **spectrum_row(z)}
        profiles.append(profile)
        centred = z - z.mean(axis=0)
        keep = np.unique(np.round(np.linspace(0, len(centred) - 1, min(len(centred), args.pool_rows))).astype(int))
        pooled_rows.append(centred[keep].astype(np.float32))
        row_story.append(np.full(len(keep), story_position[key]))
    profile_frame = pd.DataFrame(profiles).merge(stories[["story_key", "age_lo", "age_hi"]], on="story_key")
    profile_frame["age_mid"] = (profile_frame["age_lo"] + profile_frame["age_hi"]) / 2
    profile_frame.to_csv(out / "story_profiles.csv", index=False)

    # B: mean k(v) per age over the stories in each age's cell
    k_columns = [f"k_{int(round(v * 100))}" for v in overlap.V_K]
    by_key = profile_frame.set_index("story_key")
    k_mean, k_rel, rows = np.full((len(ages), len(k_columns)), np.nan), np.full((len(ages), len(k_columns)), np.nan), []
    for i, age in enumerate(ages):
        members = by_key.loc[stories["story_key"].to_numpy()[cells[int(age)]]]
        values = members[k_columns].to_numpy(dtype=float)
        k_mean[i] = np.nanmean(values, axis=0)
        k_rel[i] = np.nanmean(values / (members["n_units"].to_numpy()[:, None] - 1), axis=0)
        for j, v in enumerate(overlap.V_K):
            rows.append({"age": int(age), "variance": float(v), "mean_k": k_mean[i, j], "sem_k": np.nanstd(values[:, j]) / np.sqrt(len(values)),
                         "mean_k_over_units": k_rel[i, j], "n_stories": len(values)})
        rows[-1]["mean_units"] = float(members["n_units"].mean())
    pd.DataFrame(rows).to_csv(out / "k_by_age_variance_within.csv", index=False)
    frame = profile_frame.rename(columns={"age_mid": "age"})
    trends = [base.age_association(frame, stat, rng, args.trend_permutations, bootstrap=200) for stat in STATS if stat in frame.columns]
    pd.DataFrame(trends).to_csv(out / "age_trend_within.csv", index=False)

    # C: pooled within-story deviations, reduced to a shared basis, half-split overlap between ages
    stacked = np.concatenate(pooled_rows).astype(np.float64)
    story_of_row = np.concatenate(row_story)
    covariance = stacked.T @ stacked
    eigenvalues, axes = np.linalg.eigh(covariance)
    reduce = min(args.reduce_dim, stacked.shape[1])
    basis = axes[:, ::-1][:, :reduce]
    reduced = stacked @ basis
    rows_of_story = pd.Series(np.arange(len(story_of_row))).groupby(story_of_row).apply(lambda s: s.to_numpy()).to_dict()
    result = within_pooled(reduced, rows_of_story, cells, args, ages)
    result["share_kept"] = float(eigenvalues[::-1][:reduce].sum() / eigenvalues.sum())
    plot_within(profile_frame, k_mean, k_rel, result, out, ages, label, args)
    c_rows = [{"age": int(a), "variance": float(v), "k": result["observed"]["k"][i, j], "k_null": result["null"]["k"][i, j]}
              for i, a in enumerate(ages) for j, v in enumerate(overlap.V_K)] if result["null"] else []
    pd.DataFrame(c_rows).to_csv(out / "k_by_age_variance_within_pooled.csv", index=False)
    np.savez(out / "overlap_within_pooled.npz", overlap=result["observed"]["overlap"], ages=ages, levels=np.array(overlap.V_PAIR),
             **({"null_overlap": result["null"]["overlap"]} if result["null"] else {}))
    return {"k_within": k_mean, "k_pooled_within": result["observed"]["k"]}


def split_once(reduced: np.ndarray, rows_of_story: dict[int, np.ndarray], cells: dict[int, np.ndarray], rng: np.random.Generator,
               ages: np.ndarray) -> dict[str, np.ndarray]:
    pieces = {}
    for age, members in cells.items():
        order = rng.permutation(len(members))
        middle = len(order) // 2
        halves = [np.concatenate([rows_of_story[s] for s in members[order[a * middle:(a + 1) * middle]]]) for a in (0, 1)]
        (first, cum_first), (second, cum_second) = [overlap.principal_axes(reduced[h]) for h in halves]
        pieces[age] = (first, second, overlap.levels_to_k(cum_first, overlap.V_K), overlap.levels_to_k(cum_first, overlap.V_PAIR), overlap.levels_to_k(cum_second, overlap.V_PAIR))
    count, levels = len(ages), len(overlap.V_PAIR)
    out = {"k": np.full((count, len(overlap.V_K)), np.nan), "overlap": np.full((count, count, levels), np.nan)}
    for i, a in enumerate(ages):
        out["k"][i] = pieces[int(a)][2]
        for j, b in enumerate(ages):
            sums = overlap.corner_sums(pieces[int(a)][0] @ pieces[int(b)][1].T)
            for v in range(levels):
                width = int(min(pieces[int(a)][3][v], pieces[int(b)][4][v], *sums.shape))
                out["overlap"][i, j, v] = sums[width - 1, width - 1] / width
    return out


def within_pooled(reduced, rows_of_story, cells, args, ages) -> dict:
    rng = np.random.default_rng(args.seed + 1)
    size = min(len(m) for m in cells.values())
    cells = {a: m[:size] for a, m in cells.items()}
    observed = overlap.mean_over([split_once(reduced, rows_of_story, cells, rng, ages) for _ in range(args.splits)])
    null = None
    if args.permutations:
        story_count = len(rows_of_story)
        draws = []
        for _ in range(args.permutations):
            order = rng.permutation(story_count)
            draws.append(split_once(reduced, rows_of_story, {a: order[m] for a, m in cells.items()}, rng, ages))
        null = overlap.mean_over(draws)
        null["k_sd"] = np.nanstd(np.stack([d["k"] for d in draws]), axis=0)
        stacked = np.stack([overlap.others_mean(d["overlap"]) for d in draws])
        null["others_mean"], null["others_sd"] = np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)
    return {"observed": observed, "null": null, "cell_size": size}


def plot_within(profile_frame, k_mean, k_rel, result, out: Path, ages, label, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlap.AGES = ages
    pair_rows = [f"{int(round(v * 100))}%" for v in overlap.V_PAIR]
    figure = plt.figure(figsize=(13.5, 15.5), facecolor="#fcfcfb")
    grid = figure.add_gridspec(4, 2)
    heat(figure.add_subplot(grid[0, :]), k_mean, ages, "B. PCs a story needs for each share of its own variance, averaged over the stories at each age")
    overlap.heatmap(figure.add_subplot(grid[1, :]), k_rel.T, [f"{int(round(v * 100))}%" for v in overlap.V_K], "B. the same as a fraction of the story's units (size-free)",
                    "viridis", label="PCs / (units - 1)")
    heat(figure.add_subplot(grid[2, :]), result["observed"]["k"], ages, "C. PCs of the pooled within-story deviations at each age (shared gist axes)")
    observed = overlap.others_mean(result["observed"]["overlap"])
    overlap.heatmap(figure.add_subplot(grid[3, 0]), observed.T, pair_rows, "C. gist-axis overlap with all other ages", "viridis", vmin=0, vmax=1, label="overlap")
    if result["null"] is not None:
        z = overlap.standardised(observed, result["null"]["others_mean"], result["null"]["others_sd"], 0.005)
        overlap.heatmap(figure.add_subplot(grid[3, 1]), z.T, pair_rows, "overlap vs permuted-age null (SDs; red = more than shuffled)", "RdBu_r", diverging=True, label="SD")
    figure.suptitle(f"Within-story spectrum by age ({label}; {args.detail_view} details)", fontsize=12, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(out / "fig_within_and_pooled_within.png", dpi=140, facecolor="#fcfcfb")
    plt.close(figure)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-dir", type=Path, required=True, help="batch folder from subsets.py select")
    parser.add_argument("--baseline-csv", type=Path, default=Path("data/processed/spectral_age/baseline_span_lt7.csv"))
    parser.add_argument("--model-key", choices=sorted(MODEL_DIRS), default="gpt_oss_20b")
    parser.add_argument("--extraction-root", type=Path, default=Path(base.DEFAULT_ROOT))
    parser.add_argument("--global-dir", type=Path, default=None, help="override <root>/<model global dir>")
    parser.add_argument("--detail-dir", type=Path, default=None, help="override the folder of detail vectors for --detail-view")
    parser.add_argument("--detail-view", choices=["isolated", "contextual"], default="isolated")
    parser.add_argument("--vector-key", default="mean_vectors", help="whole-story array: mean_vectors or eos_vectors")
    parser.add_argument("--detail-key", default=None)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--steps", default="pooled,within", help="comma list of pooled, within")
    parser.add_argument("--reselect-available", action="store_true", help="rebuild the batch on the videos that have vectors")
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--per-age", type=int, default=None, help="default: the batch's own target")
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--min-units", type=int, default=20)
    parser.add_argument("--max-units", type=int, default=300)
    parser.add_argument("--pool-rows", type=int, default=40, help="rows per story kept for the pooled within-story deviations")
    parser.add_argument("--reduce-dim", type=int, default=256)
    parser.add_argument("--trend-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.batch_dir / "summary.json").read_text())
    ages = np.array(summary["ages"], dtype=int)
    args.per_age = args.per_age or int(summary["target_per_age"])
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    args.run_within = "within" in steps
    root = args.extraction_root
    model_dirs = MODEL_DIRS[args.model_key]
    dirs = {"global": [args.global_dir] if args.global_dir else [root / name for name in model_dirs["global"]],
            "isolated": [args.detail_dir] if args.detail_dir and args.detail_view == "isolated" else [root / name for name in model_dirs["isolated"]]}
    dirs["contextual"] = [args.detail_dir] if args.detail_dir and args.detail_view == "contextual" else dirs["global"]
    if "check" in steps:
        check_coverage(args, dirs, ages)
        return
    selection = prepare_selection(args, dirs, [int(a) for a in ages])
    chosen = json.loads((selection / "summary.json").read_text())
    ages = np.array(chosen["ages"], dtype=int)
    print(f"selection: {chosen['cell_size']} videos per age, {chosen['distinct_videos']} distinct; by age {[v['selected'] for v in chosen['by_age'].values()]}", flush=True)
    label = f"{args.model_key}, layer {args.layer}"
    results = {}
    if "pooled" in steps:
        results["pooled"] = run_pooled(args, dirs, selection, ages, label)
    if "within" in steps:
        results.update(run_within(args, dirs, selection, ages, label))
    plot_summary(results, ages, args.output_dir / "fig_spectrum_by_age.png", label, args)
    (args.output_dir / "run_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n")


def check_coverage(args, dirs, ages: np.ndarray) -> None:
    """Which batch / baseline videos have whole-story and detail vectors (run this first on the cluster)."""
    ids, _ = whole_story(dirs["global"], args.vector_key, args.layer)
    whole = set(ids.tolist())
    report: dict[str, object] = {"model": args.model_key, "folders_used": {name: [str(f) for f in usable(folders)] for name, folders in dirs.items()},
                                 "videos_with_whole_story_vectors": len(whole)}
    detail_sets = {}
    for view in ("isolated", "contextual"):
        try:
            detail_sets[view] = set(detail_table(dirs[view], view)["video_id"])
        except (FileNotFoundError, KeyError) as error:
            report[f"{view}_detail_vectors"] = f"unavailable: {error}"
    for view, found in detail_sets.items():
        report[f"videos_with_{view}_detail_vectors"] = len(found)
    batch = pd.read_csv(args.batch_dir / "subset_videos.csv", dtype={"video_id": str})
    membership = pd.read_csv(args.batch_dir / "membership.csv", dtype={"video_id": str})
    baseline = pd.read_csv(args.baseline_csv, low_memory=False, dtype={"video_id": str})
    for name, videos in (("batch", set(batch["video_id"])), ("baseline", set(baseline["video_id"]))):
        report[f"{name}_videos"] = len(videos)
        report[f"{name}_with_whole_story"] = len(videos & whole)
        for view, found in detail_sets.items():
            report[f"{name}_with_{view}_details"] = len(videos & found)
    per_age = {}
    for age in ages:
        cell = set(membership.loc[membership["age"] == age, "video_id"])
        per_age[int(age)] = {"cell": len(cell), "whole": len(cell & whole), **{view: len(cell & found) for view, found in detail_sets.items()}}
    report["batch_cells_covered_by_age"] = per_age
    text = json.dumps(report, indent=2)
    print(text)
    (args.output_dir / "vector_coverage.json").write_text(text + "\n")


def plot_summary(results: dict, ages: np.ndarray, path: Path, label: str, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("A. pooled whole-story vectors: PCs needed per share of variance (each age on its own)", results["pooled"]["k"] if "pooled" in results else None),
              (f"B. within story ({args.detail_view} details): mean PCs a story needs", results.get("k_within")),
              ("C. pooled within-story deviations: PCs needed", results.get("k_pooled_within"))]
    panels = [p for p in panels if p[1] is not None]
    figure, axes = plt.subplots(len(panels), 1, figsize=(13, 4.2 * len(panels)), facecolor="#fcfcfb")
    for axis, (title, matrix) in zip(np.atleast_1d(axes), panels):
        heat(axis, matrix, ages, title)
    figure.suptitle(f"Number of PCs to capture each share of variance, by age ({label})", fontsize=12, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=140, facecolor="#fcfcfb")
    plt.close(figure)


if __name__ == "__main__":
    main()
