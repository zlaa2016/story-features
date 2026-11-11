"""One curve-level age test per panel: does the whole "PCs needed vs share of variance" curve change with age?

Instead of one test per variance level, a single statistic summarises the curve:

    T = mean over the variance levels v in [v_min, v_max] of the age association of k(v)

* **B, within story** (stories are the units): the association at each level is the size-adjusted Spearman correlation of a story's
  k(v) / (units - 1) with the midpoint of its age range (story length ranked out).  The null shuffles ages among stories inside strata
  of story length, which keeps every level's dependence on length, and recomputes T.
* **A, pooled** (ages are the units): the association at each level is the Spearman correlation across ages of k(v), the number of PCs
  of that age's pooled stories; every age keeps the same number of stories (the smallest cell).  The null reshuffles the age ranges
  among stories (as in the permuted-age null of the heatmaps) and recomputes T.

Top-5 test: the same two designs applied to one scalar, the share of variance captured by the first 5 PCs (and, for the within-story
overlap, the overlap of the first 5 PCs of two halves of a story), so each panel gets a single test with no choice of variance range.

T > 0 means older ages need more PCs (a less compressed curve), T < 0 fewer.  The p-value is two-sided, (1 + #{|T*| >= |T|}) / (1 + draws).
Levels where k is constant (for example k = 1 at low v) have no association and are skipped.  Stories (B) and ages (A) are treated as
independent; no correction is applied because this is a single test per panel.

    python -m analysis.representation.geometry.spectral_age.age_tests --profiles <run>/within_isolated/story_profiles.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def level_columns(frame: pd.DataFrame, v_min: float, v_max: float) -> tuple[list[str], np.ndarray]:
    levels = [(v / 100, f"k_{v}") for v in range(int(round(v_min * 100)), int(round(v_max * 100)) + 1) if f"k_{v}" in frame.columns]
    return [name for _, name in levels], np.array([v for v, _ in levels])


def residualiser(design: np.ndarray) -> np.ndarray:
    return np.eye(len(design)) - design @ np.linalg.pinv(design)


def column_correlations(left: np.ndarray, right: np.ndarray, right_norm: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(left)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (right.T @ left) / (right_norm * norm)


def within_curve_test(profiles: pd.DataFrame, v_min: float = 0.2, v_max: float = 0.95, permutations: int = 2000, seed: int = 42) -> dict:
    """Curve-level age test over stories (panel B)."""
    columns, levels = level_columns(profiles, v_min, v_max)
    frame = profiles.dropna(subset=["age_mid", "n_units"] + columns[:1]).reset_index(drop=True)
    units = frame["n_units"].to_numpy(dtype=float)
    stats = frame[columns].to_numpy(dtype=float) / (units[:, None] - 1.0)
    age_rank = rankdata(frame["age_mid"].to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(frame)), rankdata(np.log(units))])
    project = residualiser(design)
    stat_resid = project @ np.apply_along_axis(rankdata, 0, stats)
    stat_norm = np.linalg.norm(stat_resid, axis=0)
    usable = stat_norm > 1e-9
    observed_curve = column_correlations(project @ age_rank, stat_resid, stat_norm)
    observed = float(np.nanmean(observed_curve[usable]))
    size_rank = rankdata(units, method="ordinal")
    strata = pd.qcut(size_rank, q=min(5, max(2, len(frame) // 20)), labels=False)
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for draw in range(permutations):
        permuted = age_rank.copy()
        for members in groups:
            permuted[members] = age_rank[rng.permutation(members)]
        null[draw] = np.nanmean(column_correlations(project @ permuted, stat_resid, stat_norm)[usable])
    return {"panel": "B within story", "statistic": "mean over variance levels of the size-adjusted Spearman rho(age midpoint, k(v)/(units-1))",
            "T": observed, "p_two_sided": float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (1 + permutations)),
            "null_mean": float(null.mean()), "null_sd": float(null.std()), "n_stories": int(len(frame)), "levels": [float(levels[0]), float(levels[-1])],
            "levels_used": int(usable.sum()), "permutations": permutations,
            "curve": {f"{v:.2f}": (None if not ok else float(r)) for v, r, ok in zip(levels, observed_curve, usable)}}


def k_curve(x: np.ndarray, members: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Number of PCs of the pooled rows (rows of ``x`` in ``members``) needed for each share of variance (eigenvalues from the Gram matrix)."""
    rows = x[members]
    rows = rows - rows.mean(axis=0)
    values = np.clip(np.linalg.eigvalsh(rows @ rows.T)[::-1], 0.0, None)
    cumulative = np.cumsum(values) / values.sum()
    return np.searchsorted(cumulative, levels - 1e-12) + 1


def spearman_over_ages(ages: np.ndarray, curves: np.ndarray) -> np.ndarray:
    ranks = np.apply_along_axis(rankdata, 0, curves)
    age_rank = rankdata(ages)
    centred_age, centred = age_rank - age_rank.mean(), ranks - ranks.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (centred.T @ centred_age) / (np.linalg.norm(centred, axis=0) * np.linalg.norm(centred_age))


def pooled_curve_test(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, pop: np.ndarray, ages: np.ndarray, per_age: int, multi_cells,
                      v_min: float = 0.2, v_max: float = 0.95, permutations: int = 200, seed: int = 42) -> dict:
    """Curve-level age test across ages (panel A); ``multi_cells`` is ``pc_overlap_by_age.multi_cells``."""
    levels = np.round(np.arange(v_min, v_max + 1e-9, 0.01), 2)
    rng = np.random.default_rng(seed)

    def statistic(lo_values: np.ndarray, hi_values: np.ndarray) -> tuple[float, np.ndarray, int]:
        first = multi_cells(lo_values, hi_values, per_age, rng, pop)
        size = min(len(members) for members in first.values())
        cells = multi_cells(lo_values, hi_values, size, rng, pop)
        curves = np.stack([k_curve(x, cells[int(age)], levels) for age in ages]).astype(float)
        rho = spearman_over_ages(ages.astype(float), curves)
        return float(np.nanmean(rho)), rho, size

    observed, curve, size = statistic(lo, hi)
    null = np.empty(permutations)
    for draw in range(permutations):
        order = rng.permutation(len(lo))
        null[draw] = statistic(lo[order], hi[order])[0]
    usable = np.isfinite(curve)
    return {"panel": "A pooled", "statistic": "mean over variance levels of the Spearman rho across ages between age and k(v)", "T": observed,
            "p_two_sided": float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (1 + permutations)), "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "n_ages": int(len(ages)), "stories_per_age": int(size), "levels": [float(levels[0]), float(levels[-1])], "levels_used": int(usable.sum()), "permutations": permutations,
            "curve": {f"{v:.2f}": (float(r) if np.isfinite(r) else None) for v, r in zip(levels, curve)}}


TOP_PCS = 5


def top_share(x: np.ndarray, members: np.ndarray, n_pcs: int = TOP_PCS) -> float:
    """Share of the variance of the pooled rows (rows of ``x`` in ``members``) captured by their first ``n_pcs`` PCs."""
    rows = x[members]
    rows = rows - rows.mean(axis=0)
    values = np.clip(np.linalg.eigvalsh(rows @ rows.T)[::-1], 0.0, None)
    return float(values[:n_pcs].sum() / values.sum())


def within_scalar_test(frame: pd.DataFrame, column: str, permutations: int = 2000, seed: int = 42, panel: str = "within story", n_pcs: int = TOP_PCS, statistic: str | None = None) -> dict:
    """Age test of one per-story number (stories are the units): Spearman rho with the age midpoint, story length ranked out and
    ages shuffled inside strata of story length (as in ``within_curve_test``)."""
    frame = frame.dropna(subset=["age_mid", "n_units", column]).reset_index(drop=True)
    units = frame["n_units"].to_numpy(dtype=float)
    age_rank = rankdata(frame["age_mid"].to_numpy(dtype=float))
    project = residualiser(np.column_stack([np.ones(len(frame)), rankdata(np.log(units))]))
    value = (project @ rankdata(frame[column].to_numpy(dtype=float)))[:, None]
    norm = np.linalg.norm(value, axis=0)
    observed = float(column_correlations(project @ age_rank, value, norm)[0])
    size_rank = rankdata(units, method="ordinal")
    strata = pd.qcut(size_rank, q=min(5, max(2, len(frame) // 20)), labels=False)
    groups = [np.flatnonzero(strata == level) for level in np.unique(strata)]
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for draw in range(permutations):
        permuted = age_rank.copy()
        for members in groups:
            permuted[members] = age_rank[rng.permutation(members)]
        null[draw] = column_correlations(project @ permuted, value, norm)[0]
    return {"panel": panel, "statistic": statistic or f"size-adjusted Spearman rho(age midpoint, {column}); first {n_pcs} PCs", "T": observed,
            "p_two_sided": float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (1 + permutations)), "null_mean": float(null.mean()),
            "null_sd": float(null.std()), "n_stories": int(len(frame)), "permutations": permutations}


def pooled_share_test(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, pop: np.ndarray, ages: np.ndarray, per_age: int, multi_cells,
                      permutations: int = 200, seed: int = 42, n_pcs: int = TOP_PCS) -> dict:
    """Age test across ages of the share of variance in the first PCs of the pooled stories (panel A); ages are the units, as in ``pooled_curve_test``."""
    rng = np.random.default_rng(seed)

    def statistic(lo_values: np.ndarray, hi_values: np.ndarray) -> tuple[float, np.ndarray, int]:
        first = multi_cells(lo_values, hi_values, per_age, rng, pop)
        size = min(len(members) for members in first.values())
        cells = multi_cells(lo_values, hi_values, size, rng, pop)
        shares = np.array([top_share(x, cells[int(age)], n_pcs) for age in ages])
        return float(spearman_over_ages(ages.astype(float), shares[:, None])[0]), shares, size

    observed, shares, size = statistic(lo, hi)
    null = np.empty(permutations)
    for draw in range(permutations):
        order = rng.permutation(len(lo))
        null[draw] = statistic(lo[order], hi[order])[0]
    return {"panel": "A pooled", "statistic": f"Spearman rho across ages between age and the share of variance in the first {n_pcs} PCs", "T": observed,
            "p_two_sided": float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (1 + permutations)), "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "n_ages": int(len(ages)), "stories_per_age": int(size), "permutations": permutations, "share_by_age": {str(int(a)): float(v) for a, v in zip(ages, shares)}}


def plot_curve(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = [(float(v), r) for v, r in result["curve"].items() if r is not None]
    figure, axis = plt.subplots(figsize=(7, 3.6), facecolor="#fcfcfb")
    axis.plot([p[0] for p in points], [p[1] for p in points], color="#2a6f97")
    axis.axhline(0, color="#999", linewidth=0.8)
    axis.axhline(result["T"], color="#c1440e", linestyle="--", linewidth=1, label=f"T = {result['T']:.3f}, p = {result['p_two_sided']:.3g}")
    axis.set_xlabel("share of variance captured")
    axis.set_ylabel("age association of the number of PCs")
    axis.set_title(f"{result['panel']}: association with age at each variance level", fontsize=10, loc="left")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=140, facecolor="#fcfcfb")
    plt.close(figure)


def save_top(result: dict, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def save(result: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "age_curve_test.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_curve(result, folder / "fig_age_curve_test.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", type=Path, required=True, help="within_<view>/story_profiles.csv of a finished run")
    parser.add_argument("--v-min", type=float, default=0.2)
    parser.add_argument("--v-max", type=float, default=0.95)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    result = within_curve_test(pd.read_csv(args.profiles), args.v_min, args.v_max, args.permutations, args.seed)
    save(result, args.profiles.parent)
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=2))


if __name__ == "__main__":
    main()
