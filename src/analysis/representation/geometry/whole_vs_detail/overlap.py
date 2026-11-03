"""Overlap of one story's whole-story vector with its own detail vectors (no PCA).

Everything is computed in *deviation space*: the corpus mean is removed from the whole-story vectors and, separately, from the detail
vectors, so the direction every text shares does not make all overlaps look high.  For one story, with ``w`` its centred whole-story
vector and ``D`` (n x d) its centred detail vectors:

* ``cos_mean``      cosine between ``w`` and the mean detail: does the story's deviation from the average story point where its details point?
* ``cos_detail``    mean cosine between ``w`` and each single detail.
* ``r2_span``       share of ``|w|^2`` inside the span of the n details; ``r2_span_excess`` subtracts chance (rank / d for a random direction).
* ``prefix_auc``    mean over m of the cosine between ``w`` and the mean of the first m details: how quickly the story's own order approaches the
                    whole.  ``prefix_auc_shuffled`` is the same with the details in random order, and ``order_gain`` their difference
                    (positive = the real order builds the whole faster than a random order of the same details).
* identification    ``auc_identification``: the chance that a story's own mean detail is closer (cosine) to its whole-story vector than another
                    story's mean detail is - 0.5 means the details say nothing specific about the whole.
"""
from __future__ import annotations

import numpy as np

EPSILON = 1e-12


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPSILON)


def prefix_cosines(whole: np.ndarray, details: np.ndarray) -> np.ndarray:
    """cos(w, mean of the first m details) for m = 1 .. n."""
    cumulative = np.cumsum(details, axis=0)
    return (cumulative @ whole) / np.maximum(np.linalg.norm(cumulative, axis=1) * np.linalg.norm(whole), EPSILON)


def shuffled_prefix(whole: np.ndarray, details: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Mean over ``draws`` random orders of the prefix-cosine curve."""
    curves = [prefix_cosines(whole, details[rng.permutation(len(details))]) for _ in range(draws)]
    return np.mean(curves, axis=0)


def span_r2(whole: np.ndarray, details: np.ndarray) -> tuple[float, int]:
    """Share of ``|w|^2`` that lies in the span of the detail vectors, and the rank of that span."""
    basis, singular, _ = np.linalg.svd(details.T, full_matrices=False)
    rank = int((singular > singular.max() * 1e-8).sum())
    basis = basis[:, :rank]
    projection = basis.T @ whole
    return float((projection @ projection) / max(whole @ whole, EPSILON)), rank


def resample(curve: np.ndarray, bins: int) -> np.ndarray:
    """A curve over m = 1..n reduced to ``bins`` points at equal fractions of the story (linear interpolation)."""
    fractions = np.arange(1, len(curve) + 1) / len(curve)
    return np.interp(np.linspace(1.0 / bins, 1.0, bins), fractions, curve)


def story_measures(whole: np.ndarray, details: np.ndarray, rng: np.random.Generator, draws: int = 20, bins: int = 10) -> tuple[dict[str, float], np.ndarray]:
    """Overlap measures of one story and its observed-minus-shuffled prefix curve at ``bins`` relative positions."""
    n, d = details.shape
    mean_detail = details.mean(axis=0)
    observed = prefix_cosines(whole, details)
    shuffled = shuffled_prefix(whole, details, draws, rng)
    r2, rank = span_r2(whole, details)
    cosines = unit(details) @ unit(whole)
    measures = {
        "n_details": float(n),
        "cos_mean": float(unit(mean_detail) @ unit(whole)),
        "cos_detail": float(cosines.mean()),
        "r2_span": r2,
        "r2_span_excess": r2 - rank / d,
        "prefix_auc": float(observed.mean()),
        "prefix_auc_shuffled": float(shuffled.mean()),
        "order_gain": float(observed.mean() - shuffled.mean()),
    }
    return measures, resample(observed - shuffled, bins)


def identification(wholes: np.ndarray, mean_details: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per story: AUC of its own mean detail against every other story's mean detail (cosine to its whole-story vector), and the mean
    cosine to the other stories' mean details (the chance level for ``cos_mean``)."""
    scores = unit(wholes) @ unit(mean_details).T          # scores[s, t] = cos(whole of s, mean detail of t)
    own = np.diag(scores)
    others = scores.copy()
    np.fill_diagonal(others, np.nan)
    auc = np.nanmean(own[:, None] > others, axis=1) + 0.5 * np.nanmean(own[:, None] == others, axis=1)
    return auc, np.nanmean(others, axis=1)
