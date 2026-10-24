"""Gist measures for one story's unit vectors (units x dimensions).

Everything here is held out or null-referenced, because in-sample variance thresholds are
misleading when a story has far fewer units than embedding dimensions (in-sample reconstruction
becomes perfect as k approaches n) and because LLM embeddings put a large share of variance on a
few shared directions (so k at 20% sits at 1-2 for almost every story).

* ``oos_pca_curve``: fit a PCA gist subspace on some of a story's details, measure the fraction of
  the left-out details' deviation that m components reconstruct.  What generalises is structure
  the details share, which is what a gist is.
* ``event_curve``: variance explained by the best k *contiguous* segments of the story (event
  structure), against a shuffled-order null.
* ``vq_curve``: held-out vector quantisation, a nonlinear "unique gists" version.
* every curve is compared with a column-permutation null: each embedding dimension is shuffled
  independently across the story's units, which keeps every per-dimension variance (anisotropy,
  rogue dimensions, noise level) and removes all covariance between dimensions.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

M_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40)
EVENT_KS = (2, 3, 4, 5, 6, 8, 10)
VQ_MS = (2, 3, 4, 6, 8, 12)
MIN_PEAK = 0.02


def fold_assignment(n: int, folds: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    folds = max(2, min(folds, n))
    order = rng.permutation(n)
    assignment = np.empty(n, dtype=int)
    assignment[order] = np.arange(n) % folds
    return assignment, folds


def oos_pca_curve(z: np.ndarray, rng: np.random.Generator, folds: int = 10, m_grid: Sequence[int] = M_GRID) -> np.ndarray:
    """Held-out fraction of detail deviation reconstructed by the top m PCs of the other details."""
    n = len(z)
    assignment, folds = fold_assignment(n, folds, rng)
    numerator = np.zeros(len(m_grid))
    denominator = 0.0
    for fold in range(folds):
        test = assignment == fold
        train = ~test
        if train.sum() < 4 or not test.any():
            continue
        mean = z[train].mean(axis=0)
        _, _, axes_all = np.linalg.svd(z[train] - mean, full_matrices=False)
        rank = min(axes_all.shape[0], int(train.sum()) - 1)
        axes = axes_all[:rank]
        held_out = z[test] - mean
        cumulative = np.cumsum((held_out @ axes.T) ** 2, axis=1).sum(axis=0)
        denominator += float((held_out**2).sum())
        for index, m in enumerate(m_grid):
            numerator[index] += cumulative[min(m, rank) - 1]
    if denominator <= 0:
        return np.full(len(m_grid), np.nan)
    return numerator / denominator


def vq_curve(z: np.ndarray, rng: np.random.Generator, folds: int = 5, ms: Sequence[int] = VQ_MS) -> np.ndarray:
    """Held-out gain of m centroids over the single mean: 1 - sum(min dist^2) / sum(dist to mean^2)."""
    from sklearn.cluster import KMeans

    n = len(z)
    assignment, folds = fold_assignment(n, folds, rng)
    numerator = np.zeros(len(ms))
    denominator = 0.0
    for fold in range(folds):
        test = assignment == fold
        train = ~test
        if train.sum() < max(ms) + 1 or not test.any():
            continue
        mean = z[train].mean(axis=0)
        baseline = float(((z[test] - mean) ** 2).sum())
        denominator += baseline
        for index, m in enumerate(ms):
            seed = int(rng.integers(0, 2**31 - 1))
            centroids = KMeans(n_clusters=m, n_init=2, max_iter=60, random_state=seed).fit(z[train]).cluster_centers_
            squared = ((z[test][:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2).min(axis=1)
            numerator[index] += float(squared.sum())
    if denominator <= 0:
        return np.full(len(ms), np.nan)
    return 1.0 - numerator / denominator


def excess_curve(
    curve: Callable[..., np.ndarray], z: np.ndarray, rng: np.random.Generator, null_draws: int, **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(real, column-permutation null, real - null) for any held-out curve function."""
    real = curve(z, rng, **kwargs)
    nulls = [curve(rng.permuted(z, axis=0), rng, **kwargs) for _ in range(max(null_draws, 1))]
    null = np.nanmean(nulls, axis=0)
    return real, null, real - null


def saturation_scalars(excess: np.ndarray, grid: Sequence[int], n: int, prefix: str, min_peak: float = MIN_PEAK) -> dict[str, float]:
    """Strength (peak), where the excess curve saturates (count, half), and low-m values.

    ``count`` is the smallest m reaching 90% of the peak: the estimated number of gists.  It is only
    defined when the peak itself is meaningful (>= min_peak), otherwise there is no shared structure
    to count.
    """
    grid_array = np.asarray(grid)
    values = np.asarray(excess, dtype=float)
    output = {f"{prefix}_peak": float("nan"), f"{prefix}_half": float("nan"), f"{prefix}_count": float("nan"),
              f"{prefix}_half_rel": float("nan"), f"{prefix}_count_rel": float("nan"), f"{prefix}_auc10": float("nan")}
    if not np.isfinite(values).any():
        return output
    peak = float(np.nanmax(values))
    output[f"{prefix}_peak"] = peak
    if peak >= min_peak:
        for fraction, name in ((0.5, "half"), (0.9, "count")):
            index = int(np.flatnonzero(values >= fraction * peak)[0])
            output[f"{prefix}_{name}"] = float(grid_array[index])
            output[f"{prefix}_{name}_rel"] = float(grid_array[index]) / n
    small = grid_array <= 10
    output[f"{prefix}_auc10"] = float(np.nanmean(values[small]))
    for m in (1, 3):
        if m in grid_array:
            output[f"{prefix}_E{m}"] = float(values[int(np.flatnonzero(grid_array == m)[0])])
    return output


def segment_cost_matrix(gram: np.ndarray) -> np.ndarray:
    """cost[i, j] = within-segment sum of squares of units i..j-1 (from an uncentred Gram matrix)."""
    n = len(gram)
    block = np.zeros((n + 1, n + 1))
    block[1:, 1:] = np.cumsum(np.cumsum(gram, axis=0), axis=1)
    diagonal = np.concatenate([[0.0], np.cumsum(np.diag(gram))])
    index = np.arange(n + 1)
    upper = index[None, :]
    lower = index[:, None]
    total = block[upper, upper] - block[lower, upper] - block[upper, lower] + block[lower, lower]
    length = np.maximum(upper - lower, 1)
    cost = (diagonal[upper] - diagonal[lower]) - total / length
    cost = np.where(upper > lower, np.maximum(cost, 0.0), np.inf)
    return cost


def event_curve(gram: np.ndarray, ks: Sequence[int] = EVENT_KS) -> np.ndarray:
    """Explained fraction 1 - SSE_k / SSE_1 for the best k contiguous segments, for each k."""
    n = len(gram)
    cost = segment_cost_matrix(gram)
    total = cost[0, n]
    if not np.isfinite(total) or total <= 0:
        return np.full(len(ks), np.nan)
    best = {1: cost[0, :].copy()}
    ceiling = max(ks)
    for k in range(2, ceiling + 1):
        previous = best[k - 1]
        combined = previous[:, None] + cost
        combined[~np.isfinite(combined)] = np.inf
        best[k] = combined.min(axis=0)
    return np.array([1.0 - best[k][n] / total if k <= n else np.nan for k in ks])


def event_excess(gram: np.ndarray, rng: np.random.Generator, null_draws: int, ks: Sequence[int] = EVENT_KS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Event curve against a shuffled-order null (same units, order destroyed)."""
    real = event_curve(gram, ks)
    nulls = []
    for _ in range(max(null_draws, 1)):
        order = rng.permutation(len(gram))
        nulls.append(event_curve(gram[np.ix_(order, order)], ks))
    null = np.nanmean(nulls, axis=0)
    return real, null, real - null


def duplicate_threshold(
    pool: np.ndarray, pool_story: np.ndarray, rng: np.random.Generator,
    quantile: float = 0.99, samples: int = 20000, floor: float = 0.8, ceiling: float = 0.98,
) -> float:
    """Cosine above which two details of one story count as near-duplicates (a refrain).

    Taken from the cosines between units of different stories, so it reflects the model's
    baseline similarity scale instead of a fixed number.
    """
    first = rng.integers(0, len(pool), samples)
    second = rng.integers(0, len(pool), samples)
    different = pool_story[first] != pool_story[second]
    cosine = np.einsum("ij,ij->i", pool[first[different]], pool[second[different]])
    return float(np.clip(np.quantile(cosine, quantile), floor, ceiling))


def dedupe_keep(normalized: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy near-duplicate removal in story order; returns a boolean keep mask."""
    kept: list[int] = []
    for index in range(len(normalized)):
        if kept and float((normalized[kept] @ normalized[index]).max()) >= threshold:
            continue
        kept.append(index)
    mask = np.zeros(len(normalized), dtype=bool)
    mask[kept] = True
    return mask
