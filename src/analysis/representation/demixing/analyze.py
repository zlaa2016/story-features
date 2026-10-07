from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shared.io import write_csv, write_npz
from shared.research import load_split, train_story_keys, write_run_provenance


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes"])


class PcaSignedNmf(BaseEstimator, TransformerMixin):
    def __init__(self, pca_dim: int = 128, components: int = 32, seed: int = 42, max_iter: int = 1000):
        self.pca_dim = pca_dim
        self.components = components
        self.seed = seed
        self.max_iter = max_iter

    @staticmethod
    def _signed(values: np.ndarray) -> np.ndarray:
        return np.concatenate([np.maximum(values, 0), np.maximum(-values, 0)], axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PcaSignedNmf":
        pca_dim = min(self.pca_dim, X.shape[0] - 1, X.shape[1])
        if pca_dim < 2:
            raise ValueError("PCA+NMF requires at least three training rows")
        self.pca_ = PCA(n_components=pca_dim, random_state=self.seed)
        reduced = self.pca_.fit_transform(X)
        signed = self._signed(reduced)
        components = min(self.components, signed.shape[0], signed.shape[1])
        self.nmf_ = NMF(
            n_components=components,
            init="nndsvda",
            random_state=self.seed,
            max_iter=self.max_iter,
        )
        self.nmf_.fit(signed)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "pca_") or not hasattr(self, "nmf_"):
            raise RuntimeError("Transformer must be fitted before transform")
        return self.nmf_.transform(self._signed(self.pca_.transform(X)))


def load_features(input_dir: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    vectors: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    offset = 0
    layers: np.ndarray | None = None
    for feature_text in sorted(glob.glob(str(input_dir / "features_*.npz"))):
        feature_path = Path(feature_text)
        metadata_path = input_dir / feature_path.name.replace("features_", "metadata_").replace(".npz", ".csv")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for {feature_path}: {metadata_path}")
        payload = np.load(feature_path, allow_pickle=False)
        shard = payload["vectors"].astype(np.float32)
        shard_layers = payload["layers"].astype(int)
        frame = pd.read_csv(metadata_path, low_memory=False)
        if len(frame) != len(shard):
            raise ValueError(f"Feature/metadata row mismatch for {feature_path}: {len(shard)} != {len(frame)}")
        if layers is not None and not np.array_equal(layers, shard_layers):
            raise ValueError(f"Layer mismatch in {feature_path}")
        layers = shard_layers
        frame["vector_row"] = np.arange(len(frame)) + offset
        offset += len(frame)
        vectors.append(shard)
        frames.append(frame)
    if not vectors or layers is None:
        raise FileNotFoundError("No feature shards found")
    return np.concatenate(vectors), pd.concat(frames, ignore_index=True), layers


def prepare_labels(metadata: pd.DataFrame, path: Path) -> pd.DataFrame:
    retention = pd.read_csv(path, low_memory=False)
    if "condition" not in retention:
        for candidate in ["target_age_label", "age_info_condition"]:
            if candidate in retention:
                retention["condition"] = retention[candidate]
                break
    required = {"story_key", "condition", "orig_idx"}
    missing = required - set(retention.columns)
    if missing:
        raise ValueError(f"Retention table missing columns: {sorted(missing)}")
    retention["condition"] = retention["condition"].fillna("").astype(str)
    retention["detail_index"] = pd.to_numeric(retention["orig_idx"], errors="coerce").astype("Int64")
    outcome = "retained_structured" if "retained_structured" in retention else "retained_cosine"
    if outcome not in retention:
        raise ValueError("Retention table has neither retained_structured nor retained_cosine")
    retention["survived"] = as_bool(retention[outcome]).astype(int)
    key = ["story_key", "condition", "detail_index"]
    conflicts = retention.groupby(key, dropna=False)["survived"].nunique()
    if conflicts.gt(1).any():
        raise ValueError(f"Conflicting survival labels for {int(conflicts.gt(1).sum())} detail-condition keys")
    retention = retention.drop_duplicates(key, keep="last")
    original = metadata[metadata["record_type"].eq("original")].copy()
    original["detail_index"] = pd.to_numeric(original["detail_index"], errors="coerce").astype("Int64")
    columns = key + ["survived"]
    for optional in ["orig_centrality", "orig_position_norm", "target_age", "source_age"]:
        if optional in retention.columns:
            columns.append(optional)
    return retention[columns].merge(
        original[["story_key", "detail_index", "vector_row", "token_count"]],
        on=["story_key", "detail_index"],
        how="inner",
        validate="many_to_one",
    )


def checked_splits(labels: np.ndarray, groups: np.ndarray, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Grouped evaluation requires at least two stories")
    splits = list(GroupKFold(min(folds, len(unique_groups))).split(np.zeros(len(labels)), labels, groups))
    for fold, (train, test) in enumerate(splits):
        if len(np.unique(labels[train])) < 2:
            raise ValueError(f"Training fold {fold} has only one class")
        if not len(test):
            raise ValueError(f"Test fold {fold} is empty")
    return splits


def grouped_probabilities(
    estimator: Any,
    features: np.ndarray,
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    probabilities = np.full(len(labels), np.nan, dtype=float)
    for train, test in splits:
        estimator.fit(features[train], labels[train])
        probabilities[test] = estimator.predict_proba(features[test])[:, 1]
    if np.isnan(probabilities).any():
        raise RuntimeError("Out-of-fold predictions are incomplete")
    return probabilities


def story_bootstrap_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    values: list[float] = []
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    for _ in range(iterations):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        rows = np.concatenate([group_rows[group] for group in selected])
        if len(np.unique(labels[rows])) == 2:
            values.append(float(roc_auc_score(labels[rows], probabilities[rows])))
    if len(values) < max(100, iterations // 2):
        raise RuntimeError("Too few valid story-bootstrap replicates")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit leakage-free detail demixing and normalization models.")
    parser.add_argument("--input-dir", type=Path, default=Path("results/features/representation_extraction/causal_lm/detail_extraction"))
    parser.add_argument("--retention-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/representation/demixing"))
    parser.add_argument("--story-split", type=Path, default=Path("results/analysis/representation/probes/layerwise_detail/story_split.csv"))
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    vectors, metadata, layers = load_features(args.input_dir)
    layer_values = layers.astype(int).tolist()
    if args.layer not in layer_values:
        raise ValueError(f"Layer {args.layer} not available: {layer_values}")
    layer_index = layer_values.index(args.layer)
    labels = prepare_labels(metadata, args.retention_csv)
    row_indices = labels["vector_row"].astype(int).to_numpy()
    activation = vectors[row_indices, layer_index, :]
    labels["activation_energy"] = np.linalg.norm(activation, axis=1)
    group = labels.groupby(["story_key", "condition"])["activation_energy"]
    group_sum = group.transform("sum")
    group_count = group.transform("count")
    labels["story_pool_energy"] = np.where(
        group_count.gt(1),
        (group_sum - labels["activation_energy"]) / (group_count - 1),
        np.nan,
    )
    labels["normalized_energy"] = labels["activation_energy"] / (1.0 + labels["story_pool_energy"])
    labels["log_energy"] = np.log1p(labels["activation_energy"])
    labels["log_pool"] = np.log1p(labels["story_pool_energy"])
    target_age = pd.to_numeric(labels.get("target_age"), errors="coerce")
    labels["age_x_normalized_energy"] = target_age * labels["normalized_energy"]

    baseline = [column for column in ["orig_centrality", "orig_position_norm", "token_count", "target_age", "source_age"] if column in labels]
    if not baseline:
        raise ValueError("No behavioral baseline columns are available")
    normalization = baseline + ["log_energy", "log_pool", "normalized_energy", "age_x_normalized_energy"]
    valid = labels[normalization + ["survived"]].notna().all(axis=1)
    work = labels.loc[valid].copy()
    if work["survived"].nunique() != 2:
        raise ValueError("Survival outcome must contain both classes")
    y = work["survived"].astype(int).to_numpy()
    groups = work["story_key"].astype(str).to_numpy()
    splits = checked_splits(y, groups, args.folds)
    work_activation = vectors[work["vector_row"].astype(int), layer_index, :]
    estimators = {
        "behavioral_baseline": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=args.seed)),
            work[baseline].to_numpy(float),
            baseline,
        ),
        "divisive_normalization": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=args.seed)),
            work[normalization].to_numpy(float),
            normalization,
        ),
        "pca_signed_nmf_representation": (
            make_pipeline(
                PcaSignedNmf(args.pca_dim, args.components, args.seed),
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=args.seed),
            ),
            work_activation,
            [f"pca_signed_nmf_{args.components}"],
        ),
    }
    reports: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for model_name, (estimator, features, feature_names) in estimators.items():
        probabilities = grouped_probabilities(estimator, features, y, splits)
        ci_low, ci_high = story_bootstrap_auc(y, probabilities, groups, args.bootstrap_iterations, args.seed)
        reports.append({
            "model": model_name,
            "layer": args.layer,
            "n_rows": len(work),
            "n_stories": work["story_key"].nunique(),
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "roc_auc_ci_low": ci_low,
            "roc_auc_ci_high": ci_high,
            "features": ",".join(feature_names),
            "evaluation": "grouped_out_of_fold",
        })
        prediction_rows.append(pd.DataFrame({
            "story_key": work["story_key"].astype(str).to_numpy(),
            "condition": work["condition"].astype(str).to_numpy(),
            "detail_index": work["detail_index"].to_numpy(),
            "survived": y,
            "model": model_name,
            "probability": probabilities,
        }))

    split = load_split(args.story_split)
    artifact_stories = train_story_keys(args.story_split)
    artifact_rows = labels["story_key"].astype(str).isin(artifact_stories).to_numpy()
    unique_rows = np.unique(row_indices[artifact_rows])
    rng = np.random.default_rng(args.seed)
    if len(unique_rows) > args.sample_size:
        unique_rows = np.sort(rng.choice(unique_rows, args.sample_size, replace=False))
    demix_input = vectors[unique_rows, layer_index, :]
    demixer = PcaSignedNmf(args.pca_dim, args.components, args.seed).fit(demix_input)
    loadings = demixer.transform(demix_input)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "normalization_model_comparison.csv", pd.DataFrame(reports))
    write_csv(args.output_dir / "normalization_oof_predictions.csv", pd.concat(prediction_rows, ignore_index=True))
    write_csv(args.output_dir / "detail_normalization_features.csv", labels)
    write_npz(
        args.output_dir / "demixed_components.npz",
        source_vector_rows=unique_rows,
        loadings=loadings.astype(np.float32),
        nmf_components=demixer.nmf_.components_.astype(np.float32),
        pca_components=demixer.pca_.components_.astype(np.float32),
        pca_mean=demixer.pca_.mean_.astype(np.float32),
        layer=np.asarray(args.layer),
        split_seed=np.asarray(int(split["split_seed"].iloc[0])) if "split_seed" in split else np.asarray(args.seed),
    )
    write_run_provenance(args.output_dir, args, extra={"n_rows": len(work), "n_stories": int(work["story_key"].nunique())})
    print(pd.DataFrame(reports).to_string(index=False))


if __name__ == "__main__":
    main()
