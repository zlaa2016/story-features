"""Within-story compression profile of narrative-detail vectors, aggregated by age.

For each story the matrix of its own detail vectors (units x dimensions) is analysed on its own,
so no PCA basis is shared across stories.  Per story it measures

* the variance spectrum: components needed to reach 20%..90% of variance (``k(v)``), head share
  of the first components, power-law decay exponent, participation ratio;
* incremental dimensionality PR(t) of the first t units, against a shuffled-order null;
* position drift: variance explained by a smooth trend along the story, the spectrum after that
  trend is removed, and how strongly the leading PCs track position.

Every spectrum statistic also gets an excess over a null built from random units of other
stories with the same number of units.  Stories are then assigned to one age each (a balanced
one-story-one-age matching over the age range each story covers), aggregated by age, and tested
with a story-level permutation that is stratified by the number of units.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.stats import rankdata, spearmanr

from shared.io import write_csv, write_json
from shared.research import benjamini_hochberg, write_run_provenance


V_GRID = np.round(np.arange(0.20, 0.901, 0.05), 2)
FRACTIONS = np.round(np.arange(0.2, 1.001, 0.1), 1)
BIG_COST = 1e6
DEFAULT_ROOT = "results/features/representation_extraction/causal_lm/detail_extraction"

VIEWS = {
    "isolated": {
        "feature_glob": "isolated_detail_features_*.npz",
        "metadata_prefix": "isolated_detail_metadata_",
        "feature_prefix": "isolated_detail_features_",
        "key": "mean_vectors",
        "needs_global_join": False,
    },
    "contextual": {
        "feature_glob": "contextual_detail_features_*.npz",
        "metadata_prefix": "contextual_detail_metadata_",
        "feature_prefix": "contextual_detail_features_",
        "key": "mean_control_vectors",
        "needs_global_join": True,
    },
}

TEST_STATS = [
    "k90", "kn90", "kn50", "kn20", "head1", "head3", "pr", "effrank", "alpha",
    "pr_excess", "head3_excess", "kn90_excess", "alpha_excess", "mean_pairwise_cos",
    "prgap_f40", "prgap_f60", "pr_slope_late", "tau_rel",
    "r2_pos", "r2_pos_excess", "head3_detr", "kn90_detr", "alpha_detr", "head3_drop", "pc_pos_absrho",
]


# --------------------------------------------------------------------------------------
# Spectrum primitives
# --------------------------------------------------------------------------------------
def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def center_gram(gram: np.ndarray) -> np.ndarray:
    return gram - gram.mean(axis=0, keepdims=True) - gram.mean(axis=1, keepdims=True) + gram.mean()


def eigenvalues(gram: np.ndarray) -> np.ndarray:
    values = np.linalg.eigvalsh((gram + gram.T) / 2.0)[::-1]
    return np.clip(values, 0.0, None)


def participation_ratio(values: np.ndarray) -> float:
    total = float(values.sum())
    squares = float((values**2).sum())
    return total**2 / squares if squares > 0 else float("nan")


def spectrum_stats(values: np.ndarray, number_of_units: int) -> dict[str, float]:
    """Statistics of a descending covariance spectrum from ``number_of_units`` samples."""
    rank_limit = max(number_of_units - 1, 1)
    values = np.asarray(values, dtype=np.float64)[:rank_limit]
    total = float(values.sum())
    output: dict[str, float] = {}
    if total <= 0:
        for v in V_GRID:
            output[f"k{int(round(v * 100))}"] = float("nan")
            output[f"kn{int(round(v * 100))}"] = float("nan")
        output.update(head1=np.nan, head3=np.nan, pr=np.nan, effrank=np.nan, alpha=np.nan)
        return output
    cumulative = np.cumsum(values) / total
    for v in V_GRID:
        k = int(np.searchsorted(cumulative, v - 1e-12) + 1)
        output[f"k{int(round(v * 100))}"] = float(k)
        output[f"kn{int(round(v * 100))}"] = float(k) / rank_limit
    output["head1"] = float(values[0] / total)
    output["head3"] = float(values[:3].sum() / total)
    output["pr"] = participation_ratio(values)
    shares = values / total
    shares = shares[shares > 0]
    output["effrank"] = float(np.exp(-(shares * np.log(shares)).sum()))
    fit_ranks = max(5, min(20, rank_limit // 2))
    fit_ranks = min(fit_ranks, len(values))
    positive = values[:fit_ranks] > 1e-12
    if positive.sum() >= 3:
        ranks = np.arange(1, fit_ranks + 1)[positive]
        slope = np.polyfit(np.log(ranks), np.log(values[:fit_ranks][positive]), 1)[0]
        output["alpha"] = float(-slope)
    else:
        output["alpha"] = float("nan")
    return output


def prefix_participation(gram_uncentered: np.ndarray, sizes: list[int]) -> np.ndarray:
    result = np.full(len(sizes), np.nan)
    for index, size in enumerate(sizes):
        centered = center_gram(gram_uncentered[:size, :size])
        result[index] = participation_ratio(eigenvalues(centered))
    return result


def saturating_fit(sizes: np.ndarray, participation: np.ndarray, number_of_units: int) -> tuple[float, float]:
    """Fit PR(t) = A (1 - exp(-t / tau)); returns (A, tau) or NaNs."""
    finite = np.isfinite(participation)
    if finite.sum() < 4:
        return float("nan"), float("nan")
    sizes = sizes[finite].astype(float)
    participation = participation[finite]
    top = max(float(participation.max()), 1e-6)
    try:
        parameters, _ = curve_fit(
            lambda t, amplitude, tau: amplitude * (1.0 - np.exp(-t / tau)),
            sizes,
            participation,
            p0=[top, max(number_of_units / 3.0, 1.0)],
            bounds=([1e-6, 0.1], [10.0 * top + 1.0, 1e5]),
            maxfev=2000,
        )
    except (RuntimeError, ValueError):
        return float("nan"), float("nan")
    return float(parameters[0]), float(parameters[1])


def position_basis(number_of_units: int, components: int) -> np.ndarray:
    """Orthonormal low-frequency cosine basis along relative position, orthogonal to a constant."""
    components = max(1, min(components, number_of_units - 2))
    position = (np.arange(number_of_units) + 0.5) / number_of_units
    basis = np.stack([np.cos(np.pi * j * position) for j in range(1, components + 1)], axis=1)
    basis = basis - basis.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(basis)
    return q


def position_fraction(gram_centered: np.ndarray, basis: np.ndarray) -> float:
    total = float(np.trace(gram_centered))
    if total <= 0:
        return float("nan")
    return float(np.trace(basis.T @ gram_centered @ basis) / total)


def detrend_gram(gram_centered: np.ndarray, basis: np.ndarray) -> np.ndarray:
    projector = basis @ basis.T
    identity = np.eye(len(gram_centered))
    complement = identity - projector
    return complement @ gram_centered @ complement


def leading_position_correlation(gram_centered: np.ndarray, leading: int = 3) -> tuple[float, int]:
    values, vectors = np.linalg.eigh((gram_centered + gram_centered.T) / 2.0)
    order = np.argsort(values)[::-1][:leading]
    position = np.arange(len(gram_centered))
    best, best_rank = 0.0, 0
    for rank, column in enumerate(order, start=1):
        correlation = spearmanr(vectors[:, column], position).statistic
        if np.isfinite(correlation) and abs(correlation) > best:
            best, best_rank = abs(float(correlation)), rank
    return best, best_rank


# --------------------------------------------------------------------------------------
# Story profile
# --------------------------------------------------------------------------------------
def stable_rng(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def thin_units(values: np.ndarray, maximum: int) -> tuple[np.ndarray, bool]:
    if maximum <= 0 or len(values) <= maximum:
        return values, False
    index = np.unique(np.round(np.linspace(0, len(values) - 1, maximum)).astype(int))
    return values[index], True


def story_profile(
    normalized: np.ndarray,
    dimension_scale: np.ndarray,
    pool: np.ndarray,
    pool_story: np.ndarray,
    story_key: str,
    rng: np.random.Generator,
    *,
    null_draws: int,
    position_components: int,
) -> dict[str, float]:
    """All within-story statistics for one ordered matrix of unit vectors (rows normalized)."""
    n = len(normalized)
    z = normalized / dimension_scale
    gram0 = z @ z.T
    gram = center_gram(gram0)
    covariance_eigenvalues = eigenvalues(gram / (n - 1))
    observed = spectrum_stats(covariance_eigenvalues, n)

    similarity = normalized @ normalized.T
    row: dict[str, float] = {"n_units": float(n)}
    row["mean_pairwise_cos"] = float((similarity.sum() - np.trace(similarity)) / (n * (n - 1)))
    row.update(observed)

    eligible = np.flatnonzero(pool_story != story_key)
    null_rows: list[dict[str, float]] = []
    for _ in range(null_draws):
        chosen = rng.choice(eligible, size=n, replace=len(eligible) < n)
        sample = pool[chosen]
        null_gram = center_gram(sample @ sample.T)
        null_rows.append(spectrum_stats(eigenvalues(null_gram / (n - 1)), n))
    for name in ("pr", "head3", "kn90", "alpha"):
        null_mean = float(np.nanmean([item[name] for item in null_rows])) if null_rows else float("nan")
        row[f"null_{name}"] = null_mean
        row[f"{name}_excess"] = observed[name] - null_mean

    grid = np.unique(np.round(np.linspace(5, n, 15)).astype(int))
    fraction_sizes = np.array([max(5, int(round(f * n))) for f in FRACTIONS])
    sizes = [int(size) for size in np.unique(np.concatenate([grid, fraction_sizes])) if size <= n]
    actual = prefix_participation(gram0, sizes)
    shuffled = np.zeros(len(sizes))
    for _ in range(null_draws):
        order = rng.permutation(n)
        shuffled += prefix_participation(gram0[np.ix_(order, order)], sizes)
    shuffled = shuffled / max(null_draws, 1)
    lookup = {size: index for index, size in enumerate(sizes)}
    for fraction, size in zip(FRACTIONS, fraction_sizes):
        index = lookup[min(int(size), n)]
        label = int(round(fraction * 100))
        row[f"pr_f{label}"] = float(actual[index])
        row[f"prshuf_f{label}"] = float(shuffled[index])
        row[f"prgap_f{label}"] = float(actual[index] - shuffled[index])
    amplitude, tau = saturating_fit(np.asarray(sizes), actual, n)
    row["pr_inf"] = amplitude
    row["tau"] = tau
    row["tau_rel"] = tau / n if np.isfinite(tau) else float("nan")
    row["pr_slope_late"] = (row["pr_f100"] - row["pr_f60"]) / (0.4 * n)

    basis = position_basis(n, position_components)
    row["r2_pos"] = position_fraction(gram, basis)
    shuffled_r2 = []
    for _ in range(null_draws):
        order = rng.permutation(n)
        shuffled_r2.append(position_fraction(gram[np.ix_(order, order)], basis))
    row["r2_pos_shuffled"] = float(np.nanmean(shuffled_r2)) if shuffled_r2 else float("nan")
    row["r2_pos_excess"] = row["r2_pos"] - row["r2_pos_shuffled"]
    detrended = spectrum_stats(eigenvalues(detrend_gram(gram, basis) / (n - 1)), n)
    row["head3_detr"] = detrended["head3"]
    row["kn90_detr"] = detrended["kn90"]
    row["alpha_detr"] = detrended["alpha"]
    row["pr_detr"] = detrended["pr"]
    row["head3_drop"] = observed["head3"] - detrended["head3"]
    row["pc_pos_absrho"], row["pc_pos_rank"] = leading_position_correlation(gram)
    return row


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def shard_suffixes(directory: Path, view: dict[str, Any]) -> list[str]:
    suffixes = []
    for path_text in sorted(glob.glob(str(directory / view["feature_glob"]))):
        suffix = Path(path_text).stem.removeprefix(view["feature_prefix"])
        if (directory / f"{view['metadata_prefix']}{suffix}.csv").exists():
            suffixes.append(suffix)
    return suffixes


def load_metadata(directory: Path, view_name: str) -> pd.DataFrame:
    view = VIEWS[view_name]
    parts = []
    for suffix in shard_suffixes(directory, view):
        metadata = pd.read_csv(directory / f"{view['metadata_prefix']}{suffix}.csv", low_memory=False)
        if metadata.empty:
            continue
        metadata = metadata.copy()
        metadata["shard"] = suffix
        metadata["row"] = np.arange(len(metadata))
        if view["needs_global_join"] and "video_id" not in metadata.columns:
            global_path = directory / f"global_metadata_{suffix}.csv"
            if not global_path.exists():
                raise FileNotFoundError(f"Contextual shard {suffix} needs {global_path} for video_id")
            keys = pd.read_csv(global_path, low_memory=False)[["record_id", "video_id"]].drop_duplicates("record_id")
            metadata = metadata.merge(keys, on="record_id", how="left")
        parts.append(metadata)
    if not parts:
        raise FileNotFoundError(f"No {view_name} shards under {directory}")
    table = pd.concat(parts, ignore_index=True)
    if "record_type" in table.columns:
        table = table[table["record_type"].astype(str).eq("original")]
    table = table.dropna(subset=["story_key", "source_age"])
    table["detail_index"] = pd.to_numeric(table["detail_index"], errors="coerce")
    table = table.dropna(subset=["detail_index"]).drop_duplicates(["story_key", "detail_index"])
    return table.reset_index(drop=True)


def build_story_table(details: pd.DataFrame, subset_csv: Path | None, age_min: int, age_max: int, min_units: int) -> pd.DataFrame:
    stories = (
        details.groupby("story_key", sort=True)
        .agg(n_details=("detail_index", "size"), video_id=("video_id", "first"), source_age=("source_age", "first"))
        .reset_index()
    )
    stories = stories[stories["n_details"] >= min_units].copy()
    if subset_csv is not None:
        subset = pd.read_csv(subset_csv, low_memory=False)
        required = {"video_id", "range_lower", "range_upper"}
        if not required <= set(subset.columns):
            raise ValueError(f"{subset_csv} needs columns {sorted(required)}")
        subset = subset.drop_duplicates("video_id")[["video_id", "range_lower", "range_upper"]]
        stories["video_id"] = stories["video_id"].astype(str)
        subset["video_id"] = subset["video_id"].astype(str)
        stories = stories.merge(subset, on="video_id", how="inner")
        stories["age_lo"] = pd.to_numeric(stories["range_lower"], errors="coerce")
        stories["age_hi"] = pd.to_numeric(stories["range_upper"], errors="coerce")
    else:
        rounded = np.floor(pd.to_numeric(stories["source_age"], errors="coerce") + 0.5)
        stories["age_lo"] = rounded
        stories["age_hi"] = rounded
    stories = stories.dropna(subset=["age_lo", "age_hi"])
    stories["age_lo"] = np.maximum(stories["age_lo"].astype(int), age_min)
    stories["age_hi"] = np.minimum(stories["age_hi"].astype(int), age_max)
    stories = stories[stories["age_lo"] <= stories["age_hi"]].copy()
    stories["range_width"] = stories["age_hi"] - stories["age_lo"] + 1
    return stories.reset_index(drop=True)


def load_vectors(directory: Path, view_name: str, details: pd.DataFrame, layer: int, key: str | None) -> np.ndarray:
    """Vectors for the rows of ``details`` (columns ``shard`` and ``row``) at one stored layer."""
    view = VIEWS[view_name]
    key = key or view["key"]
    shard_of = details["shard"].to_numpy()
    row_of = details["row"].to_numpy()
    output: np.ndarray | None = None
    for shard in pd.unique(shard_of):
        places = np.flatnonzero(shard_of == shard)
        path = directory / f"{view['feature_prefix']}{shard}.npz"
        with np.load(path, allow_pickle=False) as payload:
            array = payload[key]
            layers = payload["layers"].astype(int).tolist()
            if array.ndim == 3:
                if layer == -1:
                    position = len(layers) - 1
                elif layer in layers:
                    position = layers.index(layer)
                else:
                    raise ValueError(f"Layer {layer} not in {layers} for {path}")
                selected = array[row_of[places], position, :]
            else:
                selected = array[row_of[places]]
        if output is None:
            output = np.zeros((len(details), selected.shape[1]), dtype=np.float32)
        output[places] = selected.astype(np.float32)
    if output is None:
        raise ValueError("No vectors loaded")
    return output


# --------------------------------------------------------------------------------------
# Age assignment and statistics
# --------------------------------------------------------------------------------------
def assign_ages(age_lo: np.ndarray, age_hi: np.ndarray, ages: np.ndarray, per_age: int, rng: np.random.Generator) -> np.ndarray:
    """One age per story, at most ``per_age`` stories per age, only ages inside each story's range.

    Maximises the number of matched stories and, among those, prefers narrow-range stories.
    """
    ages = np.asarray(ages)
    allowed = (ages[None, :] >= age_lo[:, None]) & (ages[None, :] <= age_hi[:, None])
    width = (age_hi - age_lo + 1).astype(float)
    cost_by_age = np.where(allowed, width[:, None] + rng.random(allowed.shape) * 1e-3, BIG_COST)
    cost = np.repeat(cost_by_age, per_age, axis=1)
    rows, columns = linear_sum_assignment(cost)
    assigned = np.full(len(age_lo), np.nan)
    matched = cost[rows, columns] < BIG_COST / 2
    assigned[rows[matched]] = ages[columns[matched] // per_age]
    return assigned


def equalize(assigned: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ages, counts = np.unique(assigned[np.isfinite(assigned)], return_counts=True)
    if not len(ages):
        return assigned
    target = int(counts.min())
    output = assigned.copy()
    for age in ages:
        members = np.flatnonzero(assigned == age)
        drop = rng.choice(members, size=len(members) - target, replace=False)
        output[drop] = np.nan
    return output


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def centered_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def partial_spearman(age: np.ndarray, stat: np.ndarray, control: np.ndarray) -> float:
    ranks = [rankdata(item) for item in (age, stat, control)]
    design = np.column_stack([np.ones(len(control)), ranks[2]])
    return centered_correlation(residualize(ranks[0], design), residualize(ranks[1], design))


def age_association(
    frame: pd.DataFrame, stat: str, rng: np.random.Generator, permutations: int, bootstrap: int = 0
) -> dict[str, float]:
    """Raw and size-adjusted Spearman association of ``stat`` with age.

    The test statistic is the partial rho (age and stat both residualised on the rank of
    log n_units).  Its null comes from shuffling ages within strata of n_units and recomputing
    the same statistic, so the null is centred on zero even when the statistic depends on n.
    """
    columns = ["age", stat, "n_units"]
    valid = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    empty = {"stat": stat, "n": len(valid), "rho": np.nan, "partial_rho": np.nan,
             "partial_ci_low": np.nan, "partial_ci_high": np.nan, "p_partial": np.nan}
    if len(valid) < 10 or valid["age"].nunique() < 3 or valid[stat].nunique() < 3:
        return empty
    age = valid["age"].to_numpy(dtype=float)
    values = valid[stat].to_numpy(dtype=float)
    size = valid["n_units"].to_numpy(dtype=float)
    age_rank, stat_rank, size_rank = rankdata(age), rankdata(values), rankdata(np.log(size))
    design = np.column_stack([np.ones(len(size)), size_rank])
    stat_residual = residualize(stat_rank, design)
    observed = centered_correlation(residualize(age_rank, design), stat_residual)
    strata = pd.qcut(pd.Series(size).rank(method="first"), q=min(5, max(2, len(valid) // 20)), labels=False).to_numpy()
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    exceed = 0
    for _ in range(permutations):
        permuted = age_rank.copy()
        for members in groups:
            permuted[members] = age_rank[rng.permutation(members)]
        statistic = centered_correlation(residualize(permuted, design), stat_residual)
        if np.isfinite(statistic) and abs(statistic) >= abs(observed) - 1e-12:
            exceed += 1
    low = high = float("nan")
    if bootstrap > 0:
        draws = []
        for _ in range(bootstrap):
            pick = rng.integers(0, len(valid), size=len(valid))
            value = partial_spearman(age[pick], values[pick], np.log(size[pick]))
            if np.isfinite(value):
                draws.append(value)
        if len(draws) > 20:
            low, high = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    return {
        "stat": stat, "n": len(valid),
        "rho": float(np.corrcoef(age_rank, stat_rank)[0, 1]),
        "partial_rho": observed, "partial_ci_low": low, "partial_ci_high": high,
        "p_partial": (1 + exceed) / (1 + permutations),
    }


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float("nan"), float("nan")
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def age_summary(frame: pd.DataFrame, stats: list[str], rng: np.random.Generator, draws: int) -> pd.DataFrame:
    rows = []
    for age, group in frame.groupby("age"):
        for stat in stats:
            values = group[stat].to_numpy(dtype=float)
            low, high = bootstrap_interval(values, rng, draws)
            rows.append({
                "age": age, "stat": stat, "n_stories": int(np.isfinite(values).sum()),
                "mean": float(np.nanmean(values)) if np.isfinite(values).any() else np.nan,
                "sd": float(np.nanstd(values)) if np.isfinite(values).any() else np.nan,
                "ci_low": low, "ci_high": high,
            })
    return pd.DataFrame(rows)


def hypothesis_check(tests: pd.DataFrame) -> dict[str, Any]:
    def partial(name: str) -> float:
        row = tests[tests["stat"] == name]
        return float(row["partial_rho"].iloc[0]) if len(row) else float("nan")

    k_rho, alpha_rho, head_rho = partial("kn90"), partial("alpha"), partial("head3")
    if not np.isfinite([k_rho, alpha_rho]).all():
        verdict = "undetermined"
    elif k_rho > 0 and alpha_rho > 0:
        verdict = "dimensionality and spectral steepness both rise with age (matches the compressibility hypothesis)"
    elif k_rho > 0 and alpha_rho < 0:
        verdict = "dimensionality rises but the spectrum flattens (opposes the hypothesis)"
    elif k_rho < 0 and alpha_rho > 0:
        verdict = "dimensionality falls while the spectrum steepens (rival: homogeneous young stories are flat in the head)"
    else:
        verdict = "dimensionality falls and the spectrum flattens with age"
    return {
        "partial_rho_kn90_with_age": k_rho,
        "partial_rho_alpha_with_age": alpha_rho,
        "partial_rho_head3_with_age": head_rho,
        "note": "partial rho controls log(number of units); normalized k is used, not raw k",
        "verdict": verdict,
    }


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def make_plots(frame: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ages = sorted(frame["age"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(ages), 2)))
    color_of = dict(zip(ages, colors))

    figure, axis = plt.subplots(figsize=(6, 4))
    for age in ages:
        group = frame[frame["age"] == age]
        curve = [group[f"kn{int(round(v * 100))}"].mean() for v in V_GRID]
        axis.plot(V_GRID, curve, color=color_of[age], label=f"age {age:g}")
    axis.set_xlabel("variance explained v")
    axis.set_ylabel("components needed / (n_units - 1)")
    axis.set_title(f"{title}: k(v) by age")
    axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "fig_k_curves.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [int(round(f * 100)) for f in FRACTIONS]
    for age in ages:
        group = frame[frame["age"] == age]
        axes[0].plot(labels, [group[f"pr_f{label}"].mean() for label in labels], color=color_of[age])
        axes[1].plot(labels, [group[f"prgap_f{label}"].mean() for label in labels], color=color_of[age], label=f"age {age:g}")
    axes[0].set_title("PR(t) of the first t units")
    axes[1].set_title("PR(t) minus shuffled-order null")
    for axis in axes:
        axis.set_xlabel("% of story units")
    axes[1].legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "fig_incremental_pr.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 4))
    scatter = axis.scatter(frame["kn90"], frame["head3"], c=frame["age"], cmap="viridis", s=8, alpha=0.6)
    axis.set_xlabel("components for 90% / (n_units - 1)")
    axis.set_ylabel("variance in first 3 PCs")
    figure.colorbar(scatter, label="age")
    figure.tight_layout()
    figure.savefig(output_dir / "fig_k_vs_head.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, stat, label in (
        (axes[0], "r2_pos_excess", "variance explained by position trend (excess over shuffle)"),
        (axes[1], "head3_drop", "head share lost after detrending"),
    ):
        rows = summary[summary["stat"] == stat].sort_values("age")
        axis.errorbar(rows["age"], rows["mean"], yerr=[rows["mean"] - rows["ci_low"], rows["ci_high"] - rows["mean"]], fmt="o-")
        axis.set_xlabel("age")
        axis.set_ylabel(label, fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "fig_position_drift.png", dpi=150)
    plt.close(figure)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isolated-dir", type=Path, default=Path(f"{DEFAULT_ROOT}/gpt_oss_20b_detail_controls"))
    parser.add_argument("--contextual-dir", type=Path, default=Path(f"{DEFAULT_ROOT}/gpt_oss_20b_global_current"))
    parser.add_argument("--views", default="isolated", help="comma list of: isolated, contextual")
    parser.add_argument("--vector-key", default=None, help="override the vector array name inside the shards")
    parser.add_argument("--layer", type=int, default=-1, help="model layer number stored in the shard, -1 = last stored")
    parser.add_argument("--subset-csv", type=Path, default=None, help="video_id, range_lower, range_upper (e.g. flat_videos_D_2-12.csv)")
    parser.add_argument("--age-min", type=int, default=2)
    parser.add_argument("--age-max", type=int, default=12)
    parser.add_argument("--per-age", type=int, default=200)
    parser.add_argument("--equalize", action="store_true", help="trim every age to the smallest age count")
    parser.add_argument("--n-assignments", type=int, default=1, help="independent random one-story-one-age assignments")
    parser.add_argument("--min-units", type=int, default=20)
    parser.add_argument("--max-units", type=int, default=300, help="thin longer stories evenly to this many units; 0 = no limit")
    parser.add_argument("--null-draws", type=int, default=10)
    parser.add_argument("--position-components", type=int, default=3)
    parser.add_argument("--pool-size", type=int, default=20000)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/representation/geometry/within_story_compression/gpt_oss_20b"))
    return parser.parse_args(argv)


def run_view(view_name: str, args: argparse.Namespace) -> None:
    directory = args.isolated_dir if view_name == "isolated" else args.contextual_dir
    output_dir = args.output_dir / view_name
    output_dir.mkdir(parents=True, exist_ok=True)

    details = load_metadata(directory, view_name)
    stories = build_story_table(details, args.subset_csv, args.age_min, args.age_max, args.min_units)
    if stories.empty:
        raise ValueError("No stories left after filtering; check --subset-csv, --min-units and the age window")
    details = details[details["story_key"].isin(stories["story_key"])].sort_values(["story_key", "detail_index"]).reset_index(drop=True)
    print(f"[{view_name}] candidate stories {len(stories)}, units {len(details)}", flush=True)

    vectors = normalize_rows(load_vectors(directory, view_name, details, args.layer, args.vector_key)).astype(np.float64)
    keys = details["story_key"].to_numpy()
    boundaries = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    story_slices = {keys[start]: slice(start, stop) for start, stop in zip(boundaries[:-1], boundaries[1:])}

    pool_rng = np.random.default_rng(args.seed)
    pool_index = pool_rng.choice(len(vectors), size=min(args.pool_size, len(vectors)), replace=False)
    pool_normalized = vectors[pool_index]
    dimension_scale = pool_normalized.std(axis=0) + 1e-6
    pool = pool_normalized / dimension_scale
    pool_story = keys[pool_index]

    rows = []
    for number, story_key in enumerate(stories["story_key"], start=1):
        matrix, thinned = thin_units(vectors[story_slices[story_key]], args.max_units)
        profile = story_profile(
            matrix, dimension_scale, pool, pool_story, story_key,
            stable_rng(args.seed, story_key),
            null_draws=args.null_draws, position_components=args.position_components,
        )
        profile["story_key"] = story_key
        profile["thinned"] = float(thinned)
        rows.append(profile)
        if number % 200 == 0:
            print(f"[{view_name}] profiled {number}/{len(stories)} stories", flush=True)
    profiles = stories.merge(pd.DataFrame(rows), on="story_key", how="inner")
    write_csv(output_dir / "story_profiles.csv", profiles)

    ages = np.arange(args.age_min, args.age_max + 1)
    assignment_rows = []
    association_frames = []
    for replicate in range(args.n_assignments):
        rng = np.random.default_rng(args.seed + 1000 + replicate)
        assigned = assign_ages(profiles["age_lo"].to_numpy(), profiles["age_hi"].to_numpy(), ages, args.per_age, rng)
        if args.equalize:
            assigned = equalize(assigned, rng)
        selected = profiles.assign(age=assigned).dropna(subset=["age"]).copy()
        counts = selected["age"].value_counts().sort_index()
        print(f"[{view_name}] assignment {replicate}: stories per age {counts.to_dict()}", flush=True)
        for story_key, age in zip(selected["story_key"], selected["age"]):
            assignment_rows.append({"replicate": replicate, "story_key": story_key, "age": age})
        stats = [name for name in TEST_STATS if name in selected.columns]
        summary = age_summary(selected, stats, rng, args.bootstrap)
        tests = pd.DataFrame([age_association(selected, name, rng, args.permutations, bootstrap=min(args.bootstrap, 500)) for name in stats])
        tests["q_bh"] = benjamini_hochberg(tests["p_partial"])
        tests["replicate"] = replicate
        write_csv(output_dir / f"age_summary_r{replicate:02d}.csv", summary)
        write_csv(output_dir / f"age_tests_r{replicate:02d}.csv", tests)
        write_json(output_dir / f"hypothesis_check_r{replicate:02d}.json", hypothesis_check(tests))
        association_frames.append(tests)
        if replicate == 0:
            make_plots(selected, summary, output_dir, f"{view_name}, layer {args.layer}")
    write_csv(output_dir / "assignments.csv", pd.DataFrame(assignment_rows))
    combined = pd.concat(association_frames, ignore_index=True)
    across = combined.groupby("stat").agg(
        median_rho=("rho", "median"), median_partial_rho=("partial_rho", "median"),
        median_ci_low=("partial_ci_low", "median"), median_ci_high=("partial_ci_high", "median"),
        median_p=("p_partial", "median"), replicates=("replicate", "nunique"),
    ).reset_index()
    write_csv(output_dir / "age_tests_across_assignments.csv", across)
    write_run_provenance(output_dir, args, extra={"view": view_name, "candidate_stories": int(len(stories))})


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for view_name in [item.strip() for item in args.views.split(",") if item.strip()]:
        if view_name not in VIEWS:
            raise ValueError(f"Unknown view {view_name!r}; choose from {sorted(VIEWS)}")
        run_view(view_name, args)


if __name__ == "__main__":
    main()
