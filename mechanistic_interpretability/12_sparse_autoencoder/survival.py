from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mechanistic_interpretability.shared.sae import load_sae
from mechanistic_interpretability.shared.io import write_csv


class TopVarianceFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, limit: int = 512):
        self.limit = limit

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "TopVarianceFeatures":
        count = min(self.limit, X.shape[1])
        self.selected_ = np.argsort(np.var(X, axis=0))[-count:]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_]


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes"])


def grouped_predictions(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, folds: int, feature_limit: int) -> np.ndarray:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Grouped evaluation requires at least two stories")
    splits = list(GroupKFold(min(folds, len(unique_groups))).split(features, labels, groups))
    predictions = np.full(len(labels), np.nan)
    for fold, (train, test) in enumerate(splits):
        if len(np.unique(labels[train])) < 2:
            raise ValueError(f"Training fold {fold} has only one class")
        model = make_pipeline(TopVarianceFeatures(feature_limit), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1))
        model.fit(features[train], labels[train])
        predictions[test] = model.predict_proba(features[test])[:, 1]
    if np.isnan(predictions).any():
        raise RuntimeError("Out-of-fold predictions are incomplete")
    return predictions


def bootstrap_auc(labels: np.ndarray, predictions: np.ndarray, groups: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stories = np.unique(groups)
    rows = {story: np.flatnonzero(groups == story) for story in stories}
    values = []
    for _ in range(iterations):
        selected = rng.choice(stories, len(stories), replace=True)
        indices = np.concatenate([rows[story] for story in selected])
        if len(np.unique(labels[indices])) == 2:
            values.append(roc_auc_score(labels[indices], predictions[indices]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether SAE detail features predict behavioral detail survival.")
    parser.add_argument("--activation-root", type=Path, default=Path("mechanistic_interpretability/outputs/12_sparse_autoencoder/activations"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--retention-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-limit", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, payload = load_sae(str(args.checkpoint), device); layer = int(payload["config"]["layer"])
    mean = payload["mean"].to(device); scale = float(payload["scale"])
    detail_sums: dict[tuple[str, int], np.ndarray] = {}; detail_counts: dict[tuple[str, int], int] = {}
    for path_text in sorted(glob.glob(str(args.activation_root / f"layer_{layer:02d}" / "activations_*.npy"))):
        path = Path(path_text); values = np.load(path).astype(np.float32); metadata = pd.read_csv(path.with_name(path.name.replace("activations_", "metadata_").replace(".npy", ".csv")), low_memory=False)
        original_mask = metadata["record_type"].astype(str).eq("original").to_numpy()
        for start in range(0, len(values), args.batch_size):
            indices = np.arange(start, min(start + args.batch_size, len(values))); keep = original_mask[indices]
            if not keep.any(): continue
            batch = torch.from_numpy(values[indices[keep]]).to(device)
            with torch.inference_mode(): encoded = sae.encode((batch - mean) / scale).float().cpu().numpy()
            rows = metadata.iloc[indices[keep]]
            for vector, row in zip(encoded, rows.itertuples(index=False)):
                key = (str(row.story_key), int(row.detail_index)); detail_sums[key] = detail_sums.get(key, np.zeros_like(vector)) + vector; detail_counts[key] = detail_counts.get(key, 0) + 1
    keys = list(detail_sums); matrix = np.stack([detail_sums[key] / detail_counts[key] for key in keys])
    feature_frame = pd.DataFrame(matrix, columns=[f"sae_{index}" for index in range(matrix.shape[1])]); feature_frame["story_key"] = [key[0] for key in keys]; feature_frame["detail_index"] = [key[1] for key in keys]
    retention = pd.read_csv(args.retention_csv, low_memory=False); retention["detail_index"] = pd.to_numeric(retention["orig_idx"], errors="coerce").astype("Int64")
    outcome = "retained_structured" if "retained_structured" in retention else "retained_cosine"
    retention["survived"] = as_bool(retention[outcome]).astype(int)
    if "condition" not in retention:
        retention["condition"] = retention.get("target_age_label", retention.get("age_info_condition", "")).astype(str)
    label_keys = ["story_key", "condition", "detail_index"]
    conflicts = retention.groupby(label_keys, dropna=False)["survived"].nunique()
    if conflicts.gt(1).any():
        raise ValueError(f"Conflicting labels for {int(conflicts.gt(1).sum())} detail-condition keys")
    retention = retention.drop_duplicates(label_keys, keep="last")
    merged = retention.merge(feature_frame, on=["story_key", "detail_index"], how="inner", validate="many_to_one").dropna(subset=["survived"])
    if len(merged) < 100 or merged["survived"].nunique() < 2: raise ValueError("Too few matched retention labels")
    feature_columns = [column for column in merged if column.startswith("sae_")]; y = merged["survived"].to_numpy(); groups = merged["story_key"].astype(str).to_numpy()
    predictions = grouped_predictions(merged[feature_columns].to_numpy(np.float32), y, groups, args.folds, args.feature_limit)
    ci_low, ci_high = bootstrap_auc(y, predictions, groups, args.bootstrap_iterations, args.seed)
    result = pd.DataFrame([{"layer": layer, "rows": len(merged), "stories": len(np.unique(groups)), "selected_features_per_fold": min(args.feature_limit, len(feature_columns)), "grouped_roc_auc": roc_auc_score(y, predictions), "roc_auc_ci_low": ci_low, "roc_auc_ci_high": ci_high, "feature_selection": "training_fold_only"}])
    write_csv(args.output, result)
    predictions_frame = merged[["story_key", "condition", "detail_index", "survived"]].copy()
    predictions_frame["oof_probability"] = predictions
    write_csv(args.output.with_name(args.output.stem + "_oof_predictions.csv"), predictions_frame)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
