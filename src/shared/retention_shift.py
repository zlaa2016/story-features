"""Within-story measures of how a detail's contextualization shift relates to whether a summary keeps it.

Details are retained or omitted by a summary. Each story contributes one value per measure, so
age can be put on a continuous axis and the story (not the detail) is the unit of inference.

  auc      probability that a randomly chosen *omitted* detail has a larger shift than a randomly chosen
           *retained* one (0.5 = no relation; above 0.5 = context-dependent details are dropped).
  log2 rr  log2 of retention among low-shift details over retention among high-shift details
           (split at the story's median shift; 0 = no relation, positive = low-shift details kept more).

Both compare details of the same story, so book length and book age do not leak in. Shift grows with
position in the text and varies with detail length, so `adjust_shift` removes the part of the shift
that position and length alone predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def adjust_shift(frame: pd.DataFrame, shift: str = "shift", position: str = "position_norm", length: str = "detail_len") -> pd.Series:
    """Shift minus its mean among details with similar position (deciles) and length (quartiles)."""
    cell = (
        pd.qcut(frame[position].rank(method="first"), 10, labels=False, duplicates="drop").astype(str)
        + "_" + pd.qcut(frame[length].rank(method="first"), 4, labels=False, duplicates="drop").astype(str)
    )
    return frame[shift] - frame.groupby(cell)[shift].transform("mean")


def _auc(retained: np.ndarray, shift: np.ndarray) -> float:
    ranks = rankdata(shift)
    n_ret, n_om = int(retained.sum()), int((~retained).sum())
    omitted_rank_sum = ranks[~retained].sum()
    return float((omitted_rank_sum - n_om * (n_om + 1) / 2) / (n_ret * n_om))


def story_measures(frame: pd.DataFrame, shift: str = "shift", min_each: int = 2, y: str = "y") -> pd.DataFrame:
    """One row per story: AUC and log2 retention ratio of the given shift column."""
    rows = []
    for story, group in frame.groupby("story_key"):
        retained = group[y].to_numpy().astype(bool)
        n_ret, n_om = int(retained.sum()), int((~retained).sum())
        if n_ret < min_each or n_om < min_each:
            continue
        values = group[shift].to_numpy(float)
        low = values <= np.median(values)
        kept_low, kept_high = retained[low].sum(), retained[~low].sum()
        n_low, n_high = low.sum(), (~low).sum()
        ratio = ((kept_low + 0.5) / (n_low + 1)) / ((kept_high + 0.5) / (n_high + 1))
        rows.append({
            "story_key": story, "source_age": float(group["source_age"].iloc[0]),
            "auc": _auc(retained, values), "log2_rr": float(np.log2(ratio)),
            "retained": n_ret, "omitted": n_om, "weight": n_ret * n_om / (n_ret + n_om),
        })
    return pd.DataFrame(rows)


def kernel_smooth(x: np.ndarray, y: np.ndarray, w: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """Weighted Gaussian-kernel mean of y at each grid point."""
    k = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bandwidth) ** 2) * w[None, :]
    return (k @ y) / np.maximum(k.sum(axis=1), 1e-12)


def smooth_with_band(table: pd.DataFrame, value: str, grid: np.ndarray, bandwidth: float = 0.75, n_boot: int = 400, seed: int = 0) -> pd.DataFrame:
    """Kernel-smoothed value against age with a story-level bootstrap 95% band."""
    x, y, w = table["source_age"].to_numpy(float), table[value].to_numpy(float), table["weight"].to_numpy(float)
    est = kernel_smooth(x, y, w, grid, bandwidth)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(grid)))
    for b in range(n_boot):
        pick = rng.integers(0, len(x), len(x))
        boots[b] = kernel_smooth(x[pick], y[pick], w[pick], grid, bandwidth)
    return pd.DataFrame({"age": grid, "estimate": est, "lo": np.percentile(boots, 2.5, axis=0), "hi": np.percentile(boots, 97.5, axis=0)})


def windowed_odds(frame: pd.DataFrame, grid: np.ndarray, bandwidth: float = 0.75, n_boot: int = 100, seed: int = 0,
                  shift: str = "shift", controls: tuple[str, ...] = ("centrality_pct", "position_norm", "log_len")) -> pd.DataFrame:
    """Odds ratio for retention per +1 SD of shift as a smooth function of source age.

    A logistic regression is fitted at each age on all details, weighting each by a Gaussian kernel of
    its story's age. Stories are resampled for the interval.
    """
    from sklearn.linear_model import LogisticRegression

    z = (frame[shift] - frame[shift].mean()) / frame[shift].std()
    X = np.column_stack([z.to_numpy()] + [frame[c].to_numpy(float) for c in controls])
    y = frame["y"].to_numpy().astype(int)
    age = frame["source_age"].to_numpy(float)
    groups = list(frame.groupby("story_key").indices.values())

    def fit(index: np.ndarray, centre: float) -> float:
        weights = np.exp(-0.5 * ((age[index] - centre) / bandwidth) ** 2)
        model = LogisticRegression(C=1e6, max_iter=300).fit(X[index], y[index], sample_weight=weights)
        return float(np.exp(model.coef_[0][0]))

    everything = np.arange(len(y))
    est = np.array([fit(everything, g) for g in grid])
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(grid)))
    for b in range(n_boot):
        index = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boots[b] = [fit(index, g) for g in grid]
    return pd.DataFrame({"age": grid, "estimate": est, "lo": np.percentile(boots, 2.5, axis=0), "hi": np.percentile(boots, 97.5, axis=0)})
