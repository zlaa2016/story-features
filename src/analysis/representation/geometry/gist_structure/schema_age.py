"""Age-level schema dimensionality: how many shared dimensions does a band's stories use, and are they nested?

The unit of analysis is the *deviation* of a detail from its own story's mean, so a story's theme is
removed and what remains is how details elaborate around it.  Details are pooled across the stories
of an age band, giving that band's elaboration ("schema") space.

Per band (matched in number of stories and details per story):

* dimensionality: components needed for 50/80/90% of the plateau of the **held-out** curve (stories
  held out in folds), minus a column-permutation null; also the number of components above a
  column-permutation parallel-analysis threshold;
* transfer: how well band a's subspace reconstructs band b's stories (b held out), as a share of
  b's own held-out fidelity, compared at m no smaller than the larger band's dimension.  If younger
  schemas are contained in older ones, older subspaces explain younger stories better than the
  reverse (negative asymmetry, younger source minus older source);
* subspace overlap between bands (principal-angle overlap) and its decay with age distance;
* per story: how much of its detail structure its band's schema explains, relative to the story's
  own leave-detail-out gist subspace (``schema_share``).

Band labels are permuted across stories within topic x size strata to test every trend statistic.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from shared.io import write_csv, write_json
from shared.research import write_run_provenance

base = importlib.import_module("analysis.representation.geometry.within_story_compression.analyze")
gm = importlib.import_module("analysis.representation.geometry.gist_structure.gist_math")
ws = importlib.import_module("analysis.representation.geometry.gist_structure.within_story")

DEFAULT_ROOT = "results/analysis/representation/geometry/gist_structure"
M_GRID = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
SHARE_M = (4, 8, 16)


# --------------------------------------------------------------------------------------
# Core linear algebra on X with shape (stories, details_per_story, reduced_dim)
# --------------------------------------------------------------------------------------
def top_axes(rows: np.ndarray) -> np.ndarray:
    """Right singular vectors (principal axes) of pooled rows, uncentred (rows are deviations)."""
    return np.linalg.svd(rows, full_matrices=False)[2]


def story_fidelity(axes: np.ndarray, stories: np.ndarray, m_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-story numerator (sum over details of the part reconstructed by m axes) and denominator."""
    count, per_story, dimension = stories.shape
    rows = stories.reshape(-1, dimension)
    cumulative = np.cumsum((rows @ axes.T) ** 2, axis=1)[:, m_index]
    numerator = cumulative.reshape(count, per_story, len(m_index)).sum(axis=1)
    denominator = (rows**2).sum(axis=1).reshape(count, per_story).sum(axis=1)
    return numerator, denominator


def band_cv(
    stories: np.ndarray, groups: np.ndarray, rng: np.random.Generator, folds: int, m_index: np.ndarray, null: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Story-grouped cross-validation.  Returns per-occurrence numerator, denominator and the band curve.

    ``groups`` gives each occurrence's story id; repeated ids (bootstrap) always share a fold.
    """
    data = stories
    if null:
        data = rng.permuted(stories.reshape(-1, stories.shape[2]), axis=0).reshape(stories.shape)
    unique = np.unique(groups)
    folds = max(2, min(folds, len(unique)))
    order = rng.permutation(len(unique))
    fold_of = dict(zip(unique[order], np.arange(len(unique)) % folds))
    fold_index = np.array([fold_of[group] for group in groups])
    numerator = np.zeros((len(groups), len(m_index)))
    denominator = np.zeros(len(groups))
    for fold in range(folds):
        test = fold_index == fold
        train = ~test
        if train.sum() < 3 or not test.any():
            continue
        axes = top_axes(data[train].reshape(-1, data.shape[2]))
        top, bottom = story_fidelity(axes, data[test], m_index)
        numerator[test] = top
        denominator[test] = bottom
    curve = numerator.sum(axis=0) / max(denominator.sum(), 1e-12)
    return numerator, denominator, curve


def dimension_at(excess: np.ndarray, grid: np.ndarray, fraction: float) -> float:
    peak = float(np.nanmax(excess))
    if not np.isfinite(peak) or peak <= 0:
        return float("nan")
    return float(grid[int(np.flatnonzero(excess >= fraction * peak)[0])])


def parallel_analysis_dimension(rows: np.ndarray, rng: np.random.Generator, draws: int = 8) -> int:
    values = np.linalg.svd(rows, compute_uv=False) ** 2
    nulls = np.max([np.linalg.svd(rng.permuted(rows, axis=0), compute_uv=False) ** 2 for _ in range(draws)], axis=0)
    above = values > nulls
    return int(np.argmin(above)) if not above.all() else int(len(above))


def self_fidelity(stories: np.ndarray, rng: np.random.Generator, folds: int, m_index: np.ndarray) -> np.ndarray:
    """Per story: leave-detail-out reconstruction by the story's own axes (uncentred, same scale as the band)."""
    result = np.zeros((len(stories), len(m_index)))
    for number, rows in enumerate(stories):
        n = len(rows)
        order = rng.permutation(n)
        assignment = np.empty(n, dtype=int)
        assignment[order] = np.arange(n) % max(2, min(folds, n))
        numerator = np.zeros(len(m_index))
        denominator = 0.0
        for fold in range(assignment.max() + 1):
            test = assignment == fold
            train = ~test
            axes = top_axes(rows[train])
            rank = axes.shape[0]
            cumulative = np.cumsum((rows[test] @ axes.T) ** 2, axis=1).sum(axis=0)
            denominator += float((rows[test] ** 2).sum())
            numerator += cumulative[np.minimum(m_index, rank - 1)]
        result[number] = numerator / max(denominator, 1e-12)
    return result


def slope(values: np.ndarray, positions: np.ndarray) -> float:
    ok = np.isfinite(values)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(positions[ok], values[ok], 1)[0])


# --------------------------------------------------------------------------------------
# All band statistics for one labelling
# --------------------------------------------------------------------------------------
def schema_stats(
    stories: np.ndarray, labels: np.ndarray, band_names: list[str], band_mid: dict[str, float], rng: np.random.Generator,
    cfg: argparse.Namespace, lite: bool = False,
) -> dict[str, Any]:
    reduced = stories.shape[2]
    grid = np.array([m for m in M_GRID if m <= reduced - 1])
    m_index = grid - 1
    transfer_m = np.array([m for m in cfg.transfer_m if m <= reduced - 1])
    transfer_index = transfer_m - 1
    share_m = np.array([m for m in SHARE_M if m <= reduced - 1])
    share_index = share_m - 1

    sizes = {band: int((labels == band).sum()) for band in band_names}
    target = min(sizes.values())
    chosen = {band: rng.choice(np.flatnonzero(labels == band), target, replace=False) for band in band_names}

    result: dict[str, Any] = {"target_per_band": target, "bands": {}}
    axes_full: dict[str, np.ndarray] = {}
    own_cv: dict[str, np.ndarray] = {}
    per_story: dict[str, dict[str, np.ndarray]] = {}
    for band in band_names:
        idx = chosen[band]
        block = stories[idx]
        _, _, real = band_cv(block, idx, rng, cfg.folds, np.concatenate([m_index, transfer_index, share_index]))
        null = np.mean([band_cv(block, idx, rng, cfg.folds, m_index, null=True)[2] for _ in range(cfg.null_draws)], axis=0)
        curve = real[: len(m_index)]
        excess = curve - null
        own_cv[band] = real[len(m_index): len(m_index) + len(transfer_index)]
        axes_full[band] = top_axes(block.reshape(-1, reduced))
        entry = {
            "R": curve, "R_null": null, "E": excess,
            "D50": dimension_at(excess, grid, 0.5), "D80": dimension_at(excess, grid, 0.8), "D90": dimension_at(excess, grid, 0.9),
            "E_peak": float(np.nanmax(excess)),
            "pa_dim": float(parallel_analysis_dimension(block.reshape(-1, reduced), rng)),
        }
        result["bands"][band] = entry
        if not lite:
            numerator, denominator, _ = band_cv(block, idx, rng, cfg.folds, share_index)
            per_story[band] = {"idx": idx, "band_R": numerator / np.maximum(denominator[:, None], 1e-12)}

    mids = np.array([band_mid[band] for band in band_names])
    for name in ("D50", "D80", "D90", "E_peak", "pa_dim"):
        result[f"slope_{name}"] = slope(np.array([result["bands"][band][name] for band in band_names]), mids)

    ratio = np.full((len(band_names), len(band_names), len(transfer_m)), np.nan)
    raw = np.full_like(ratio, np.nan)
    for a, band_a in enumerate(band_names):
        for b, band_b in enumerate(band_names):
            if a == b:
                raw[a, b] = own_cv[band_b][: len(transfer_m)]
            else:
                numerator, denominator = story_fidelity(axes_full[band_a], stories[chosen[band_b]], transfer_index)
                raw[a, b] = numerator.sum(axis=0) / max(denominator.sum(), 1e-12)
            ratio[a, b] = raw[a, b] / np.maximum(own_cv[band_b][: len(transfer_m)], 1e-12)
    result["transfer_ratio"], result["transfer_raw"], result["transfer_m"] = ratio, raw, transfer_m

    asymmetry, overlap_pairs, distance_pairs = [], [], []
    for a in range(len(band_names)):
        for b in range(a + 1, len(band_names)):
            # Nestedness is compared at m >= the larger band's dimension: below that, which band explains
            # the other better depends on how flat each spectrum is, not on containment.
            larger = np.nanmax([result["bands"][band_names[a]]["D90"], result["bands"][band_names[b]]["D90"]])
            position = int(np.flatnonzero(grid >= larger)[0]) if np.isfinite(larger) and (grid >= larger).any() else len(grid) - 1
            ratios = []
            for source, target in ((a, b), (b, a)):
                numerator, denominator = story_fidelity(axes_full[band_names[source]], stories[chosen[band_names[target]]], np.array([grid[position] - 1]))
                fidelity = numerator.sum() / max(denominator.sum(), 1e-12)
                ratios.append(fidelity / max(result["bands"][band_names[target]]["R"][position], 1e-12))
            asymmetry.append(ratios[0] - ratios[1])
            overlaps = []
            for m in transfer_m:
                overlaps.append(np.linalg.norm(axes_full[band_names[a]][:m] @ axes_full[band_names[b]][:m].T) ** 2 / m)
            overlap_pairs.append(float(np.mean(overlaps)))
            distance_pairs.append(abs(mids[a] - mids[b]))
    result["asymmetry_mean"] = float(np.mean(asymmetry)) if asymmetry else float("nan")
    rho = spearmanr(overlap_pairs, distance_pairs).statistic if len(overlap_pairs) >= 3 else float("nan")
    result["overlap_distance_rho"] = float(rho) if np.isfinite(rho) else float("nan")
    result["overlap_pairs"] = overlap_pairs

    if not lite:
        rows = []
        for band in band_names:
            info = per_story[band]
            self_values = self_fidelity(stories[info["idx"]], rng, cfg.folds, share_index)
            for position, story_index in enumerate(info["idx"]):
                row = {"story_index": int(story_index), "band": band}
                for column, m in enumerate(share_m):
                    row[f"band_R_m{m}"] = float(info["band_R"][position, column])
                    row[f"self_R_m{m}"] = float(self_values[position, column])
                    row[f"schema_share_m{m}"] = float(info["band_R"][position, column] / max(self_values[position, column], 1e-9))
                rows.append(row)
        result["story_rows"] = pd.DataFrame(rows)
    return result


TREND_STATS = ("slope_D50", "slope_D80", "slope_D90", "slope_E_peak", "slope_pa_dim", "asymmetry_mean", "overlap_distance_rho")


def permutation_test(
    stories: np.ndarray, frame: pd.DataFrame, band_names: list[str], band_mid: dict[str, float], observed: dict[str, Any],
    rng: np.random.Generator, cfg: argparse.Namespace,
) -> dict[str, Any]:
    """Shuffle band labels across stories within topic x size strata; recompute every trend statistic."""
    labels = frame["band"].to_numpy()
    groups = ws._strata(frame.assign(n_units=frame["n_details"]))
    draws = {name: [] for name in TREND_STATS}
    for _ in range(cfg.permutations):
        shuffled = labels.copy()
        for members in groups:
            shuffled[members] = labels[rng.permutation(members)]
        stats = schema_stats(stories, shuffled, band_names, band_mid, rng, cfg, lite=True)
        for name in TREND_STATS:
            draws[name].append(stats[name])
    out = {}
    for name in TREND_STATS:
        null = np.asarray(draws[name], dtype=float)
        null = null[np.isfinite(null)]
        value = observed[name]
        out[name] = {
            "observed": float(value) if np.isfinite(value) else None,
            "null_2.5": float(np.quantile(null, 0.025)) if len(null) else None,
            "null_97.5": float(np.quantile(null, 0.975)) if len(null) else None,
            "p": float((1 + (np.abs(null) >= abs(value) - 1e-12).sum()) / (1 + len(null))) if len(null) and np.isfinite(value) else None,
        }
    return out


def bootstrap_dimensions(stories: np.ndarray, labels: np.ndarray, band_names: list[str], rng: np.random.Generator, cfg: argparse.Namespace) -> pd.DataFrame:
    """Story-level bootstrap of each band's held-out dimensions (resampled stories always share a fold)."""
    reduced = stories.shape[2]
    grid = np.array([m for m in M_GRID if m <= reduced - 1])
    rows = []
    for band in band_names:
        members = np.flatnonzero(labels == band)
        draws = []
        for _ in range(cfg.bootstrap):
            idx = rng.choice(members, len(members), replace=True)
            block = stories[idx]
            curve = band_cv(block, idx, rng, cfg.folds, grid - 1)[2]
            null = band_cv(block, idx, rng, cfg.folds, grid - 1, null=True)[2]
            excess = curve - null
            draws.append([dimension_at(excess, grid, f) for f in (0.5, 0.8, 0.9)] + [float(np.nanmax(excess))])
        draws = np.asarray(draws, dtype=float)
        for column, name in enumerate(("D50", "D80", "D90", "E_peak")):
            rows.append({"band": band, "statistic": name, "mean": float(np.nanmean(draws[:, column])),
                         "ci_low": float(np.nanquantile(draws[:, column], 0.025)), "ci_high": float(np.nanquantile(draws[:, column], 0.975))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Data preparation and driver
# --------------------------------------------------------------------------------------
def prepare(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame]:
    """Stories x details x reduced-dimension array of deviations, and the matching story table."""
    details = base.load_metadata(args.isolated_dir, "isolated")
    selected = pd.read_csv(args.selected_csv, low_memory=False)
    details = details[details["story_key"].isin(selected["story_key"])].sort_values(["story_key", "detail_index"]).reset_index(drop=True)
    vectors = base.normalize_rows(base.load_vectors(args.isolated_dir, "isolated", details, args.layer, args.vector_key))
    keys = details["story_key"].to_numpy()
    edges = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    slices = {keys[a]: slice(a, b) for a, b in zip(edges[:-1], edges[1:])}
    rng = np.random.default_rng(args.seed)
    pool_index = rng.choice(len(vectors), size=min(args.pool_size, len(vectors)), replace=False)
    scale = vectors[pool_index].std(axis=0) + 1e-6
    threshold = gm.duplicate_threshold(vectors[pool_index], keys[pool_index], rng)

    kept, deviations, original_counts, duplicate_shares = [], [], [], []
    for story_key, window in slices.items():
        normalized = vectors[window]
        before = len(normalized)
        if args.dedupe:
            normalized = normalized[gm.dedupe_keep(normalized, threshold)]
        if len(normalized) < args.details_per_story:
            continue
        z = normalized / scale
        pick = np.sort(base.stable_rng(args.seed, f"schema|{story_key}").choice(len(z), args.details_per_story, replace=False))
        deviations.append((z - z.mean(axis=0))[pick])
        kept.append(story_key)
        original_counts.append(window.stop - window.start)
        duplicate_shares.append(1.0 - len(normalized) / before)
    pooled = np.concatenate(deviations)
    from sklearn.decomposition import PCA

    fit_rows = pooled[rng.choice(len(pooled), size=min(len(pooled), 30000), replace=False)]
    reducer = PCA(n_components=min(args.reduce_dim, fit_rows.shape[1], len(fit_rows) - 1), svd_solver="randomized", random_state=args.seed)
    reducer.fit(fit_rows)
    stories = np.stack([block @ reducer.components_.T for block in deviations])
    table = selected[selected["story_key"].isin(kept)].set_index("story_key").loc[kept].reset_index()
    table["n_details"] = original_counts
    table["dup_share"] = duplicate_shares
    table["n_units"] = table["n_details"].astype(float)
    covariates = base.story_covariates(details)
    table = table.drop(columns=[c for c in covariates.columns if c != "story_key" and c in table.columns])
    table = table.merge(covariates, on="story_key", how="left")
    return stories, table


def make_plots(summary: dict[str, Any], band_names: list[str], band_mid: dict[str, float], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(band_names), 2)))
    grid = np.array(M_GRID[: len(summary["bands"][band_names[0]]["E"])])
    for color, band in zip(colors, band_names):
        axes[0].plot(grid, summary["bands"][band]["E"], color=color, label=band)
    axes[0].set_xscale("log")
    axes[0].set_title("held-out schema curve, excess over null")
    axes[0].set_xlabel("components m")
    axes[0].legend(fontsize=7)
    middle = len(summary["transfer_m"]) // 2
    matrix = summary["transfer_ratio"][:, :, middle]
    image = axes[1].imshow(matrix, cmap="viridis", vmin=min(0.0, np.nanmin(matrix)), vmax=max(1.0, np.nanmax(matrix)))
    axes[1].set_xticks(range(len(band_names)))
    axes[1].set_xticklabels(band_names, fontsize=7, rotation=45)
    axes[1].set_yticks(range(len(band_names)))
    axes[1].set_yticklabels(band_names, fontsize=7)
    axes[1].set_xlabel("stories reconstructed (target band)")
    axes[1].set_ylabel("subspace from (source band)")
    axes[1].set_title(f"transfer ratio at m={int(summary['transfer_m'][middle])}", fontsize=9)
    figure.colorbar(image, ax=axes[1])
    mids = [band_mid[band] for band in band_names]
    for name, color in (("D50", "tab:blue"), ("D80", "tab:orange"), ("pa_dim", "tab:green")):
        axes[2].plot(mids, [summary["bands"][band][name] for band in band_names], "o-", color=color, label=name)
    axes[2].set_xlabel("band midpoint age")
    axes[2].set_title("held-out schema dimensionality")
    axes[2].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "fig_schema_dimensionality.png", dpi=150)
    plt.close(figure)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isolated-dir", type=Path, default=Path(f"{base.DEFAULT_ROOT}/gpt_oss_20b_detail_controls"))
    parser.add_argument("--selected-csv", type=Path, default=Path(f"{DEFAULT_ROOT}/selection/selected_stories.csv"))
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--vector-key", default=None)
    parser.add_argument("--details-per-story", type=int, default=40)
    parser.add_argument("--reduce-dim", type=int, default=96)
    parser.add_argument("--dedupe", action="store_true", help="robustness: drop near-duplicate details first (stories left with fewer than --details-per-story are excluded, which can bias against repetitive young stories)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--null-draws", type=int, default=2)
    parser.add_argument("--transfer-m", type=lambda text: [int(item) for item in text.split(",")], default=[8, 16, 32])
    parser.add_argument("--bootstrap", type=int, default=60)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--effect-draws", type=int, default=300)
    parser.add_argument("--effect-permutations", type=int, default=300)
    parser.add_argument("--pool-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(f"{DEFAULT_ROOT}/schema/gpt_oss_20b"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    stories, table = prepare(args)
    labels = table["band"].to_numpy()
    band_names = list(table.sort_values("band_mid")["band"].drop_duplicates())
    band_mid = {band: float(table[table["band"] == band]["band_mid"].iloc[0]) for band in band_names}
    print(f"{len(table)} stories x {stories.shape[1]} details x {stories.shape[2]} dims; bands {table.groupby('band').size().to_dict()}", flush=True)
    inclusion = ws.inclusion_report(pd.read_csv(args.selected_csv, low_memory=False), table)
    write_csv(args.output_dir / "inclusion_by_band.csv", inclusion)
    print(f"stories kept / selected per band (a band losing many more stories than the others is a confound):\n{inclusion.to_string(index=False)}", flush=True)

    summary = schema_stats(stories, labels, band_names, band_mid, rng, args)
    grid = np.array([m for m in M_GRID if m <= stories.shape[2] - 1])
    curves = pd.DataFrame([
        {"band": band, "m": int(m), "R": float(r), "R_null": float(n), "E": float(e)}
        for band in band_names
        for m, r, n, e in zip(grid, summary["bands"][band]["R"], summary["bands"][band]["R_null"], summary["bands"][band]["E"])
    ])
    write_csv(args.output_dir / "schema_band_curves.csv", curves)
    dims = pd.DataFrame([{"band": band, **{k: v for k, v in summary["bands"][band].items() if k in ("D50", "D80", "D90", "E_peak", "pa_dim")}} for band in band_names])
    write_csv(args.output_dir / "schema_dimensions.csv", dims)
    write_csv(args.output_dir / "schema_dimension_bootstrap.csv", bootstrap_dimensions(stories, labels, band_names, rng, args))
    transfer_rows = []
    for a, source in enumerate(band_names):
        for b, target in enumerate(band_names):
            for column, m in enumerate(summary["transfer_m"]):
                transfer_rows.append({"source_band": source, "target_band": target, "m": int(m),
                                      "fidelity": float(summary["transfer_raw"][a, b, column]), "ratio_to_own": float(summary["transfer_ratio"][a, b, column])})
    write_csv(args.output_dir / "schema_transfer.csv", pd.DataFrame(transfer_rows))

    tests = permutation_test(stories, table, band_names, band_mid, summary, rng, args)
    write_json(args.output_dir / "schema_trend_tests.json", tests)

    story_rows = summary["story_rows"].copy()
    story_rows["story_key"] = table["story_key"].to_numpy()[story_rows["story_index"].to_numpy()]
    story_frame = table.merge(story_rows.drop(columns=["story_index", "band"]), on="story_key", how="inner")
    write_csv(args.output_dir / "story_schema_share.csv", story_frame)
    outcomes = [f"{prefix}_m{m}" for m in SHARE_M for prefix in ("schema_share", "band_R", "self_R") if f"{prefix}_m{m}" in story_frame.columns]
    effects = ws.age_effects(story_frame, outcomes, rng, argparse.Namespace(bootstrap=args.effect_draws, permutations=args.effect_permutations))
    write_csv(args.output_dir / "schema_story_age_effects.csv", effects)
    make_plots(summary, band_names, band_mid, args.output_dir)
    write_run_provenance(args.output_dir, args, extra={"stories": int(len(table)), "bands": band_names})


if __name__ == "__main__":
    main()
