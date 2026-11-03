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

D. halves      (step ``halves``) each story's details are split at random into two halves, each half gets its own PCA and the
               two sets of PCs are compared (subspace overlap at each share of variance); the mean over the stories at an
               age is the panel.  Details are never pooled across stories.  The null compares a half of the story with a
               half of another story of that age.  ``fig_four_panels.png`` puts A (PCs; PC overlap between two halves of the stories at an age), B and
               D side by side.

Other reductions reported next to PCA: participation ratio, entropy effective rank, spectral decay
exponent and the TwoNN intrinsic dimension (nonlinear manifold dimension).

Input: a batch folder from ``subsets.py select`` (``subset_stories.csv`` + ``membership.csv``; the unit is the story).  With
``--reselect-available`` the batch is rebuilt on the stories that really have vectors and the batch
actually run on is written to ``<output-dir>/selection/``.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.representation.geometry.spectral_age import age_tests, subsets

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


def load_global_by_story(directory: Path, key: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Whole-story vectors of the original records in one folder, keyed by ``story_key``."""
    ids: list[str] = []
    rows: list[np.ndarray] = []
    for feature_path in sorted(directory.glob("global_features_*.npz")):
        suffix = feature_path.stem.removeprefix("global_features_")
        metadata_path = directory / f"global_metadata_{suffix}.csv"
        if not metadata_path.exists():
            print(f"[warn] skipping {feature_path.name}: no metadata yet (shard still being written?)", flush=True)
            continue
        metadata = pd.read_csv(metadata_path, low_memory=False)
        try:
            with np.load(feature_path, allow_pickle=False) as payload:
                array = payload[key]
                layers = payload["layers"].astype(int).tolist()
                position = len(layers) - 1 if layer == -1 else layers.index(layer)
                chosen = np.flatnonzero(metadata["record_type"].astype(str).eq("original").to_numpy())
                rows.append(array[chosen, position, :].astype(np.float32))
        except (OSError, ValueError, EOFError) as error:
            print(f"[warn] skipping unreadable {feature_path.name}: {error}", flush=True)
            continue
        ids.extend(metadata.iloc[chosen]["story_key"].astype(str).tolist())
    if not rows:
        return np.array([], dtype=str), np.zeros((0, 0), dtype=np.float32)
    return np.asarray(ids), np.concatenate(rows)


def whole_story(folders: list[Path], key: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """One vector per story (``story_key``) from every usable folder; the first folder wins when a story is in several."""
    ids: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    for folder in usable(folders):
        found_ids, found_vectors = load_global_by_story(folder, key, layer)
        if len(found_ids):
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
    table["story_key"] = table["story_key"].astype(str)
    first_folder = table.groupby("story_key")["folder"].transform("first")
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
        return {**out, "top5_share": np.nan, **{f"k_{int(round(v * 100))}": np.nan for v in overlap.V_K}}
    cumulative = np.cumsum(values) / total
    out["top5_share"] = float(cumulative[min(age_tests.TOP_PCS, len(cumulative)) - 1])
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
def available_stories(args: argparse.Namespace, dirs: dict[str, list[Path]]) -> set[str]:
    ids, _ = whole_story(dirs["global"], args.vector_key, args.layer)
    found = set(ids.tolist())
    if args.run_within:
        found &= set(detail_table(dirs[args.detail_view], args.detail_view)["story_key"])
    return found


SPECTRAL_DATA = Path("data/processed/spectral_age")


def baseline_path(args: argparse.Namespace) -> Path:
    """The story baseline the batch was built from (named in its summary.json) unless ``--baseline-csv`` is given."""
    if args.baseline_csv:
        return args.baseline_csv
    name = json.loads((args.batch_dir / "summary.json").read_text()).get("baseline") or "baseline_stories_span_lt8.csv"
    return SPECTRAL_DATA / name


def prepare_selection(args: argparse.Namespace, dirs: dict[str, Path], ages: list[int]) -> Path:
    out = args.output_dir / "selection"
    if not args.reselect_available:
        out.mkdir(parents=True, exist_ok=True)
        for name in ("subset_stories.csv", "membership.csv", "summary.json"):
            (out / name).write_bytes((args.batch_dir / name).read_bytes())
        return out
    baseline = pd.read_csv(baseline_path(args), low_memory=False, dtype={"story_key": str, "video_id": str})
    available = available_stories(args, dirs)
    batch = json.loads((args.batch_dir / "summary.json").read_text())
    if batch.get("design") == "flat":
        membership, subset, summary = subsets.solve_flat(baseline, ages, available, None, None, baseline_path(args).name)
    elif batch.get("design") == "random":
        membership, subset, summary = subsets.select_random(baseline, ages, args.per_age, available, int(batch.get("seed", args.seed)), baseline_path(args).name)
    else:
        membership, subset, summary = subsets.select_flat(baseline, ages, args.per_age, available, equal=args.equal_cells, baseline_name=baseline_path(args).name)
    summary["stories_with_vectors"] = len(available)
    subsets.write_batch(out, membership, subset, summary)
    return out


def pooled_table(ids: np.ndarray, vectors: np.ndarray, subset: pd.DataFrame, standardize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unit-normalised (optionally standardised) story vectors with each story's range and popularity."""
    frame = pd.DataFrame({"story_key": ids, "row": np.arange(len(ids))}).merge(
        subset.drop_duplicates("story_key")[["story_key", "range_lower", "range_upper", "log_views"]], on="story_key", how="inner")
    x = base.normalize_rows(vectors[frame["row"].to_numpy()]).astype(np.float64)
    if standardize:
        x = x / (x.std(axis=0) + 1e-6)
    return x, frame["range_lower"].to_numpy(dtype=int), frame["range_upper"].to_numpy(dtype=int), frame["log_views"].to_numpy(dtype=float)


# --------------------------------------------------------------------------------------
# A. pooled whole-story spectrum
# --------------------------------------------------------------------------------------
def run_pooled(args, dirs, selection: Path, ages: np.ndarray, label: str) -> dict:
    """Panel A.  By default every age keeps its own cell size (the batch's target where enough stories exist); ``--equal-cells`` cuts all ages to the smallest."""
    original = overlap.equal_cells
    if not args.equal_cells:
        overlap.equal_cells = lambda build, lo, hi, per_age, rng, pop=None: build(lo, hi, per_age, rng, pop)
    try:
        return _run_pooled(args, dirs, selection, ages, label)
    finally:
        overlap.equal_cells = original


def _run_pooled(args, dirs, selection: Path, ages: np.ndarray, label: str) -> dict:
    subset = pd.read_csv(selection / "subset_stories.csv", dtype={"story_key": str, "video_id": str})
    ids, vectors = whole_story(dirs["global"], args.vector_key, args.layer)
    x, lo, hi, pop = pooled_table(ids, vectors, subset, args.standardize)
    random_design = json.loads((selection / "summary.json").read_text()).get("design") == "random"
    if random_design:
        pop = None  # random draws, not the most viewed
    overlap.AGES = ages
    out = args.output_dir / "pooled"
    out.mkdir(parents=True, exist_ok=True)
    result = overlap.run_mode(x, lo, hi, "multi", args.per_age, args.splits, args.permutations, args.seed, pop=pop)
    for name, table in overlap.tables(result, "multi").items():
        table.to_csv(out / f"{name}.csv", index=False)
    sizes = sorted(set(result["cells"].values()))
    size = f"{sizes[0]}-{sizes[-1]}" if len(sizes) > 1 else str(sizes[0])
    overlap.plot_main(result, out / "fig_age_by_variance.png", f"Age by variance captured, pooled stories ({label})",
                      f"{size} {'randomly drawn' if random_design else 'most-viewed'} stories per age (cells split in halves), {result['unique_stories']} distinct stories",
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
    test = age_tests.pooled_curve_test(x, lo, hi, pop, ages, args.per_age, overlap.multi_cells, args.curve_vmin, args.curve_vmax, args.pooled_curve_permutations, args.seed)
    age_tests.save(test, out)
    top = age_tests.pooled_share_test(x, lo, hi, pop, ages, args.per_age, overlap.multi_cells, args.pooled_curve_permutations, args.seed)
    age_tests.save_top(top, out, "age_top5_test.json")
    print(f"[pooled] top-{age_tests.TOP_PCS} age test: rho = {top['T']:.3f}, p = {top['p_two_sided']:.4g}", flush=True)
    print(f"[pooled] curve-level age test: T = {test['T']:.3f}, p = {test['p_two_sided']:.4g} ({test['stories_per_age']} stories per age)", flush=True)
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
    position = {key: i for i, key in enumerate(stories["story_key"].astype(str))}
    cells = {}
    for age in ages:
        keys = membership.loc[membership["age"] == age].sort_values("rank")["story_key"].astype(str)
        cells[int(age)] = np.array([position[k] for k in keys if k in position], dtype=int)
    return cells


def run_within(args, dirs, selection: Path, ages: np.ndarray, label: str) -> dict:
    overlap.AGES = ages
    membership = pd.read_csv(selection / "membership.csv", dtype={"story_key": str})
    subset = pd.read_csv(selection / "subset_stories.csv", dtype={"story_key": str, "video_id": str})
    details = detail_table(dirs[args.detail_view], args.detail_view)
    stories = (details.groupby("story_key", sort=True).agg(n_details=("detail_index", "size"), video_id=("video_id", "first")).reset_index()
               .merge(subset[["story_key", "range_lower", "range_upper"]], on="story_key"))
    stories = stories[(stories["n_details"] >= args.min_units) & stories["story_key"].isin(membership["story_key"])].copy()
    stories["age_lo"] = np.maximum(stories["range_lower"].astype(int), int(ages.min()))
    stories["age_hi"] = np.minimum(stories["range_upper"].astype(int), int(ages.max()))
    stories = stories.reset_index(drop=True)
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
    k_mean, rows = np.full((len(ages), len(k_columns)), np.nan), []
    for i, age in enumerate(ages):
        members = by_key.loc[stories["story_key"].to_numpy()[cells[int(age)]]]
        values = members[k_columns].to_numpy(dtype=float)
        k_mean[i] = np.nanmean(values, axis=0)
        for j, v in enumerate(overlap.V_K):
            rows.append({"age": int(age), "variance": float(v), "mean_k": k_mean[i, j], "sem_k": np.nanstd(values[:, j]) / np.sqrt(len(values)),
                         "n_stories": len(values)})
        rows[-1]["mean_units"] = float(members["n_units"].mean())
    pd.DataFrame(rows).to_csv(out / "k_by_age_variance_within.csv", index=False)
    test = age_tests.within_curve_test(profile_frame, args.curve_vmin, args.curve_vmax, args.curve_permutations, args.seed)
    age_tests.save(test, out)
    top = age_tests.within_scalar_test(profile_frame, "top5_share", args.curve_permutations, args.seed)
    age_tests.save_top(top, out, "age_top5_test.json")
    print(f"[within:{args.detail_view}] top-{age_tests.TOP_PCS} age test: rho = {top['T']:.3f}, p = {top['p_two_sided']:.4g} ({top['n_stories']} stories)", flush=True)
    print(f"[within:{args.detail_view}] curve-level age test: T = {test['T']:.3f}, p = {test['p_two_sided']:.4g} ({test['n_stories']} stories)", flush=True)
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
    plot_within(profile_frame, k_mean, result, out, ages, label, args)
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
    if args.equal_cells:
        size = min(len(m) for m in cells.values())
        cells = {a: m[:size] for a, m in cells.items()}
    size = min(len(m) for m in cells.values())
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


def plot_within(profile_frame, k_mean, result, out: Path, ages, label, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlap.AGES = ages
    pair_rows = [f"{int(round(v * 100))}%" for v in overlap.V_PAIR]
    figure = plt.figure(figsize=(13.5, 12), facecolor="#fcfcfb")
    grid = figure.add_gridspec(3, 2)
    heat(figure.add_subplot(grid[0, :]), k_mean, ages, "B. PCs a story needs for each share of its own variance, averaged over the stories at each age")
    heat(figure.add_subplot(grid[1, :]), result["observed"]["k"], ages, "C. PCs of the pooled within-story deviations at each age (shared gist axes)")
    observed = overlap.others_mean(result["observed"]["overlap"])
    overlap.heatmap(figure.add_subplot(grid[2, 0]), observed.T, pair_rows, "C. gist-axis overlap with all other ages", "viridis", vmin=0, vmax=1, label="overlap")
    if result["null"] is not None:
        z = overlap.standardised(observed, result["null"]["others_mean"], result["null"]["others_sd"], 0.005)
        overlap.heatmap(figure.add_subplot(grid[2, 1]), z.T, pair_rows, "overlap vs permuted-age null (SDs; red = more than shuffled)", "RdBu_r", diverging=True, label="SD")
    figure.suptitle(f"Within-story spectrum by age ({label}; {args.detail_view} details)", fontsize=12, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(out / "fig_within_and_pooled_within.png", dpi=140, facecolor="#fcfcfb")
    plt.close(figure)


# --------------------------------------------------------------------------------------
# D. PC overlap between two halves of a story's details
# --------------------------------------------------------------------------------------
def axes_overlap(first: np.ndarray, cumulative_first: np.ndarray, second: np.ndarray, cumulative_second: np.ndarray, keep: int) -> np.ndarray:
    """Mean squared cosine between the leading axes of two bases at every share of variance."""
    levels = overlap.V_PAIR
    sums = overlap.corner_sums(first[:keep] @ second[:keep].T)
    k_first, k_second = overlap.levels_to_k(cumulative_first, levels), overlap.levels_to_k(cumulative_second, levels)
    width = np.array([int(min(k_first[j], k_second[j], *sums.shape)) for j in range(len(levels))])
    return np.array([sums[w - 1, w - 1] / w for w in width])


def run_halves(args, dirs, selection: Path, ages: np.ndarray, label: str) -> dict:
    """Each story's details are split at random into two halves, each half gets its own PCA, and the two sets of PCs are compared;
    the mean over the stories at an age is the panel.  The null compares a half of the story with a half of another story of that age."""
    overlap.AGES = ages
    membership = pd.read_csv(selection / "membership.csv", dtype={"story_key": str})
    details = detail_table(dirs[args.detail_view], args.detail_view)
    counts = details.groupby("story_key").size()
    keys = sorted(set(membership["story_key"]) & set(counts[counts >= args.min_units].index))
    details = thinned_details(details[details["story_key"].isin(keys)], args.max_units)
    vectors = base.normalize_rows(detail_vectors(details, args.detail_view, args.layer, args.detail_key)).astype(np.float64)
    if args.standardize:
        vectors = vectors / (vectors.std(axis=0) + 1e-6)
    rows_of = details.groupby("story_key", sort=False).indices
    levels = overlap.V_PAIR
    out = args.output_dir / f"halves_{args.detail_view}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[halves:{args.detail_view}] {len(keys)} stories, {len(details)} detail rows; {args.halves_splits} random splits", flush=True)
    in_cell = {int(a): set(membership.loc[membership["age"] == a, "story_key"]) for a in ages}
    cells = {a: np.array([i for i, k in enumerate(keys) if k in members], dtype=int) for a, members in in_cell.items()}
    rng = np.random.default_rng(args.seed + 2)
    shape = (args.halves_splits, len(ages), len(levels))
    observed, null = np.full(shape, np.nan), np.full(shape, np.nan)
    top = age_tests.TOP_PCS
    story_top, age_top, age_top_null = np.full((args.halves_splits, len(keys)), np.nan), np.full((args.halves_splits, len(ages)), np.nan), np.full((args.halves_splits, len(ages)), np.nan)
    for t in range(args.halves_splits):
        halves = []
        for key in keys:
            rows = rng.permutation(rows_of[key])
            cut = len(rows) // 2
            pair = []
            for part in (rows[:cut], rows[cut:]):
                axes, cumulative = overlap.principal_axes(vectors[part])
                pair.append((axes[: args.halves_axes].astype(np.float32), cumulative))
            halves.append(pair)
        own = np.array([axes_overlap(*h[0], *h[1], args.halves_axes) for h in halves])
        story_top[t] = [overlap.subspace_overlap(h[0][0], h[1][0], top) for h in halves]
        for i, age in enumerate(ages):
            members = cells[int(age)]
            if len(members) < 5:
                continue
            observed[t, i] = np.nanmean(own[members], axis=0)
            partners = [rng.choice(members[members != m]) for m in members]
            null[t, i] = np.nanmean([axes_overlap(*halves[m][0], *halves[p][1], args.halves_axes) for m, p in zip(members, partners)], axis=0)
            age_top[t, i] = story_top[t, members].mean()
            age_top_null[t, i] = np.mean([overlap.subspace_overlap(halves[m][0][0], halves[p][1][0], top) for m, p in zip(members, partners)])
        print(f"[halves] split {t + 1}/{args.halves_splits}", flush=True)
    mean_obs, mean_null, sd_null = np.nanmean(observed, axis=0), np.nanmean(null, axis=0), np.nanstd(null, axis=0)
    records = [{"age": int(age), "n_stories": len(cells[int(age)]), "variance": v, "overlap": mean_obs[i, j], "null_mean": mean_null[i, j], "null_sd": sd_null[i, j]}
               for i, age in enumerate(ages) for j, v in enumerate(levels)]
    pd.DataFrame(records).to_csv(out / "overlap_halves.csv", index=False)
    pd.DataFrame({"age": ages, "n_stories": [len(cells[int(a)]) for a in ages], f"overlap_top{top}": np.nanmean(age_top, axis=0),
                  "null_mean": np.nanmean(age_top_null, axis=0), "null_sd": np.nanstd(age_top_null, axis=0)}).to_csv(out / f"overlap_halves_top{top}.csv", index=False)
    subset = pd.read_csv(selection / "subset_stories.csv", dtype={"story_key": str}).drop_duplicates("story_key").set_index("story_key")
    lower = np.maximum(subset.loc[keys, "range_lower"].to_numpy(dtype=int), int(ages.min()))
    upper = np.minimum(subset.loc[keys, "range_upper"].to_numpy(dtype=int), int(ages.max()))
    frame = pd.DataFrame({"story_key": keys, "age_mid": (lower + upper) / 2, "n_units": [len(rows_of[k]) for k in keys], f"top{top}_overlap": np.nanmean(story_top, axis=0)})
    test = age_tests.within_scalar_test(frame, f"top{top}_overlap", args.curve_permutations, args.seed, panel="4 details within a story (halves)")
    age_tests.save_top(test, out, f"age_top{top}_test.json")
    print(f"[halves] top-{top} age test: rho = {test['T']:.3f}, p = {test['p_two_sided']:.4g} ({test['n_stories']} stories)", flush=True)
    np.savez(out / "overlap_halves.npz", ages=ages, levels=np.array(levels), observed=mean_obs, null_mean=mean_null, null_sd=sd_null)
    plot_halves(mean_obs, mean_null, sd_null, out / "fig_halves_overlap.png", ages, label, args)
    return {}


def plot_halves(observed, null_mean, null_sd, path: Path, ages, label: str, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlap.AGES = ages
    rows = [f"{int(round(v * 100))}%" for v in overlap.V_PAIR]
    figure = plt.figure(figsize=(13.5, 5), facecolor="#fcfcfb")
    grid = figure.add_gridspec(1, 2)
    overlap.heatmap(figure.add_subplot(grid[0, 0]), observed.T, rows, "PC overlap between two halves of a story's details, mean over the stories at each age",
                    "viridis", vmin=0, vmax=1, label="mean squared cosine")
    z = overlap.standardised(observed, null_mean, null_sd, 0.005)
    overlap.heatmap(figure.add_subplot(grid[0, 1]), z.T, rows, "vs a half of another story of that age (SDs; red = more than other stories)", "RdBu_r", diverging=True, label="SD")
    figure.suptitle(f"Within-story detail PC overlap by age ({label}; {args.detail_view} details)", fontsize=12, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=140, facecolor="#fcfcfb")
    plt.close(figure)


def plot_overview(args, ages: np.ndarray, label: str) -> None:
    """The four panels in one figure, read from the step outputs: 1 whole-story PCs, 2 per-story detail PCs, 3 whole-story PC overlap between two halves of the stories at an age,
    4 PC overlap between halves of a story's details."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled, within, halves = args.output_dir / "pooled", args.output_dir / f"within_{args.detail_view}", args.output_dir / f"halves_{args.detail_view}"
    needed = [pooled / "k_by_age_variance.csv", pooled / "pairs_by_age_variance.csv", within / "k_by_age_variance_within.csv", halves / "overlap_halves.csv"]
    if not all(path.exists() for path in needed):
        return
    overlap.AGES = ages
    k_whole = pd.read_csv(needed[0]).query("mode == 'multi'").pivot(index="age", columns="variance", values="k").reindex(ages)
    pairs = pd.read_csv(needed[1]).query("mode == 'multi' and age == other_age")  # the two random halves of the stories at an age
    between = pairs.groupby(["age", "variance"])["overlap"].mean().unstack().reindex(ages)
    k_story = pd.read_csv(needed[2]).pivot(index="age", columns="variance", values="mean_k").reindex(ages)
    half = pd.read_csv(needed[3]).pivot(index="age", columns="variance", values="overlap").reindex(ages)
    pair_rows = [f"{int(round(v * 100))}%" for v in between.columns]
    figure = plt.figure(figsize=(14, 10), facecolor="#fcfcfb")
    grid = figure.add_gridspec(2, 2)
    heat(figure.add_subplot(grid[0, 0]), k_whole.to_numpy(), ages, "1. each story one vector, PCA over the stories of an age: PCs needed")
    heat(figure.add_subplot(grid[0, 1]), k_story.to_numpy(), ages, "2. each detail one vector, PCA within each story: PCs needed (mean over stories)")
    overlap.heatmap(figure.add_subplot(grid[1, 0]), between.to_numpy().T, pair_rows, "3. whole-story PCs within an age: overlap between two random halves of the stories", "viridis", vmin=0, vmax=1, label="overlap")
    overlap.heatmap(figure.add_subplot(grid[1, 1]), half.to_numpy().T, [f"{int(round(v * 100))}%" for v in half.columns], "4. details within a story: overlap between two halves' PCs (mean over stories)",
                    "viridis", vmin=0, vmax=1, label="overlap")
    figure.suptitle(f"Whole stories vs details by age ({label}; {args.detail_view} details)", fontsize=12, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(args.output_dir / "fig_four_panels.png", dpi=140, facecolor="#fcfcfb")
    plt.close(figure)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-dir", type=Path, required=True, help="batch folder from subsets.py select")
    parser.add_argument("--baseline-csv", type=Path, default=None, help="default: the baseline named in the batch's summary.json")
    parser.add_argument("--model-key", choices=sorted(MODEL_DIRS), default="gpt_oss_20b")
    parser.add_argument("--extraction-root", type=Path, default=Path(base.DEFAULT_ROOT))
    parser.add_argument("--global-dir", type=Path, default=None, help="override <root>/<model global dir>")
    parser.add_argument("--detail-dir", type=Path, default=None, help="override the folder of detail vectors for --detail-view")
    parser.add_argument("--detail-view", choices=["isolated", "contextual"], default="isolated")
    parser.add_argument("--vector-key", default="mean_vectors", help="whole-story array: mean_vectors or eos_vectors")
    parser.add_argument("--detail-key", default=None)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--steps", default="pooled,within", help="comma list of pooled, within, halves; overview alone redraws fig_four_panels.png from saved outputs")
    parser.add_argument("--equal-cells", action="store_true", help="cut every age to the smallest age's size (default: each age keeps its own size)")
    parser.add_argument("--reselect-available", action="store_true", help="rebuild the batch on the stories that have vectors")
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--per-age", type=int, default=None, help="default: the batch's own target")
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--min-units", type=int, default=20)
    parser.add_argument("--max-units", type=int, default=300)
    parser.add_argument("--pool-rows", type=int, default=40, help="rows per story kept for the pooled within-story deviations")
    parser.add_argument("--reduce-dim", type=int, default=256)
    parser.add_argument("--trend-permutations", type=int, default=500)
    parser.add_argument("--curve-vmin", type=float, default=0.2, help="lowest variance level in the curve-level age test")
    parser.add_argument("--curve-vmax", type=float, default=0.95)
    parser.add_argument("--curve-permutations", type=int, default=2000, help="permutations for the within-story curve test")
    parser.add_argument("--pooled-curve-permutations", type=int, default=200, help="permuted-age draws for the pooled curve test")
    parser.add_argument("--halves-splits", type=int, default=8, help="random half splits of each story's details for the within-story PC overlap")
    parser.add_argument("--halves-axes", type=int, default=150, help="leading axes of each half kept for that overlap")
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
    args.run_within = bool(steps & {"within", "halves"})  # both need detail vectors
    root = args.extraction_root
    model_dirs = MODEL_DIRS[args.model_key]
    dirs = {"global": [args.global_dir] if args.global_dir else [root / name for name in model_dirs["global"]],
            "isolated": [args.detail_dir] if args.detail_dir and args.detail_view == "isolated" else [root / name for name in model_dirs["isolated"]]}
    dirs["contextual"] = [args.detail_dir] if args.detail_dir and args.detail_view == "contextual" else dirs["global"]
    if "check" in steps:
        check_coverage(args, dirs, ages)
        return
    if steps == {"overview"}:  # redraw the four panels from the saved step outputs
        chosen = json.loads((args.output_dir / "selection" / "summary.json").read_text())
        plot_overview(args, np.array(chosen["ages"], dtype=int), f"{args.model_key}, layer {args.layer}")
        return
    selection = prepare_selection(args, dirs, [int(a) for a in ages])
    chosen = json.loads((selection / "summary.json").read_text())
    ages = np.array(chosen["ages"], dtype=int)
    print(f"selection: {chosen['cell_size']} stories per age, {chosen['distinct_stories']} distinct stories; by age {[v['selected'] for v in chosen['by_age'].values()]}", flush=True)
    label = f"{args.model_key}, layer {args.layer}"
    results = {}
    if "pooled" in steps:
        results["pooled"] = run_pooled(args, dirs, selection, ages, label)
    if "within" in steps:
        results.update(run_within(args, dirs, selection, ages, label))
    if "halves" in steps:
        run_halves(args, dirs, selection, ages, label)
    plot_overview(args, ages, label)
    if results:
        plot_summary(results, ages, args.output_dir / "fig_spectrum_by_age.png", label, args)
    (args.output_dir / "run_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n")


def check_coverage(args, dirs, ages: np.ndarray) -> None:
    """Which batch / baseline stories have whole-story and detail vectors (run this first on the cluster)."""
    ids, _ = whole_story(dirs["global"], args.vector_key, args.layer)
    whole = set(ids.tolist())
    report: dict[str, object] = {"model": args.model_key, "folders_used": {name: [str(f) for f in usable(folders)] for name, folders in dirs.items()},
                                 "stories_with_whole_story_vectors": len(whole), "whole_story_key_examples": sorted(whole)[:3]}
    detail_sets = {}
    for view in ("isolated", "contextual"):
        try:
            detail_sets[view] = set(detail_table(dirs[view], view)["story_key"])
        except (FileNotFoundError, KeyError) as error:
            report[f"{view}_detail_vectors"] = f"unavailable: {error}"
    for view, found in detail_sets.items():
        report[f"stories_with_{view}_detail_vectors"] = len(found)
    batch = pd.read_csv(args.batch_dir / "subset_stories.csv", dtype={"story_key": str})
    membership = pd.read_csv(args.batch_dir / "membership.csv", dtype={"story_key": str})
    baseline = pd.read_csv(baseline_path(args), low_memory=False, dtype={"story_key": str})
    report["batch_story_key_examples"] = batch["story_key"].head(3).tolist()
    for name, keys in (("batch", set(batch["story_key"])), ("baseline", set(baseline["story_key"]))):
        report[f"{name}_stories"] = len(keys)
        report[f"{name}_with_whole_story"] = len(keys & whole)
        for view, found in detail_sets.items():
            report[f"{name}_with_{view}_details"] = len(keys & found)
    per_age = {}
    for age in ages:
        cell = set(membership.loc[membership["age"] == age, "story_key"])
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
