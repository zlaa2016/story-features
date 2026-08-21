from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mechanistic_interpretability.shared.io import write_csv, write_npz
from mechanistic_interpretability.shared.research import assign_story_split, save_story_split, write_run_provenance


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes"])


def load_shards(input_dir: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    vectors = []
    metadata = []
    layers = None
    for path_text in sorted(glob.glob(str(input_dir / "features_*.npz"))):
        path = Path(path_text)
        meta_path = input_dir / path.name.replace("features_", "metadata_").replace(".npz", ".csv")
        if not meta_path.exists():
            continue
        payload = np.load(path, allow_pickle=False)
        shard_layers = payload["layers"]
        if layers is None:
            layers = shard_layers
        elif not np.array_equal(layers, shard_layers):
            raise ValueError(f"Layer mismatch in {path}")
        shard_vectors = payload["vectors"].astype(np.float32)
        shard_meta = pd.read_csv(meta_path, low_memory=False)
        if len(shard_vectors) != len(shard_meta):
            raise ValueError(f"Vector/metadata mismatch in {path}")
        shard_meta["vector_row"] = np.arange(len(shard_meta)) + sum(len(item) for item in vectors)
        vectors.append(shard_vectors)
        metadata.append(shard_meta)
    if not vectors or layers is None:
        raise FileNotFoundError(f"No complete feature shards in {input_dir}")
    return np.concatenate(vectors), pd.concat(metadata, ignore_index=True), layers


def aggregate_records(vectors: np.ndarray, metadata: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    rows = []
    pooled = []
    for record_id, group in metadata.groupby("record_id", sort=False):
        indices = group["vector_row"].astype(int).to_numpy()
        pooled.append(vectors[indices].mean(axis=0))
        first = group.iloc[0]
        rows.append(
            {
                "record_row": len(pooled) - 1,
                "record_id": record_id,
                "record_type": first["record_type"],
                "story_key": first["story_key"],
                "condition": first["condition"],
                "summary_id": first["summary_id"],
                "source_age": first.get("source_age"),
                "target_age": first.get("target_age"),
                "compression_ratio": first.get("compression_ratio"),
                "n_details": len(group),
            }
        )
    return np.stack(pooled), pd.DataFrame(rows)


def safe_splits(groups: pd.Series, requested: int) -> int:
    available = int(groups.nunique())
    if available < 2:
        raise ValueError("Grouped cross-validation requires at least two stories")
    return min(requested, available)


def checked_group_splits(
    features: np.ndarray,
    labels: np.ndarray,
    groups: pd.Series,
    folds: int,
    *,
    classification: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(safe_splits(groups, folds))
    splits = list(splitter.split(features, labels, groups))
    if classification:
        classes = set(np.unique(labels))
        for fold, (train, test) in enumerate(splits):
            if set(np.unique(labels[train])) != classes:
                raise ValueError(f"Training fold {fold} does not contain every class")
            if not len(test):
                raise ValueError(f"Evaluation fold {fold} is empty")
    return splits


def condition_probe(X: np.ndarray, frame: pd.DataFrame, layer: int, folds: int) -> dict[str, Any] | None:
    subset = frame[frame["condition"].isin(["younger", "actual", "older"])].copy()
    if subset["condition"].nunique() < 2:
        return None
    indices = subset["record_row"].astype(int).to_numpy()
    y = subset["condition"].astype(str).to_numpy()
    groups = subset["story_key"].astype(str)
    features = X[indices, layer, :]
    cv = checked_group_splits(features, y, groups, folds, classification=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    predictions = cross_val_predict(model, features, y, groups=groups, cv=cv, method="predict")
    return {
        "task": "condition_classification",
        "layer_index": layer,
        "n_rows": len(subset),
        "n_stories": groups.nunique(),
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
    }


def age_probe(X: np.ndarray, frame: pd.DataFrame, layer: int, folds: int) -> dict[str, Any] | None:
    subset = frame[(frame["record_type"] == "summary") & frame["target_age"].notna()].copy()
    if len(subset) < 20:
        return None
    indices = subset["record_row"].astype(int).to_numpy()
    subset["target_age_numeric"] = pd.to_numeric(subset["target_age"], errors="coerce")
    subset = subset.dropna(subset=["target_age_numeric"])
    if len(subset) < 20:
        return None
    indices = subset["record_row"].astype(int).to_numpy()
    y = subset["target_age_numeric"].to_numpy(float)
    groups = subset["story_key"].astype(str)
    features = X[indices, layer, :]
    cv = checked_group_splits(features, y, groups, folds, classification=False)
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    predictions = cross_val_predict(model, features, y, groups=groups, cv=cv)
    return {
        "task": "target_age_regression",
        "layer_index": layer,
        "n_rows": len(subset),
        "n_stories": groups.nunique(),
        "r2": r2_score(y, predictions),
        "mae": mean_absolute_error(y, predictions),
        "correlation": np.corrcoef(y, predictions)[0, 1],
    }


def survival_probe(
    vectors: np.ndarray,
    metadata: pd.DataFrame,
    retention: pd.DataFrame,
    layer: int,
    folds: int,
) -> dict[str, Any] | None:
    original = metadata[metadata["record_type"] == "original"].copy()
    original["detail_index"] = pd.to_numeric(original["detail_index"], errors="coerce").astype("Int64")
    labels = retention.copy()
    if "condition" not in labels:
        labels["condition"] = labels.get("target_age_label", labels.get("age_info_condition", "")).astype(str)
    labels["detail_index"] = pd.to_numeric(labels.get("orig_idx", labels.get("detail_index")), errors="coerce").astype("Int64")
    retention_column = "retained_structured" if "retained_structured" in labels else "retained_cosine"
    labels["survived"] = as_bool(labels[retention_column])
    label_keys = ["story_key", "condition", "detail_index"]
    conflicts = labels.groupby(label_keys, dropna=False)["survived"].nunique()
    if conflicts.gt(1).any():
        raise ValueError(f"Conflicting survival labels for {int(conflicts.gt(1).sum())} detail-condition keys")
    labels = labels[label_keys + ["survived"]].drop_duplicates()
    merged = labels.merge(
        original[["story_key", "detail_index", "vector_row"]],
        on=["story_key", "detail_index"],
        how="inner",
    )
    if len(merged) < 100 or merged["survived"].nunique() < 2:
        return None
    indices = merged["vector_row"].astype(int).to_numpy()
    y = merged["survived"].astype(int).to_numpy()
    groups = merged["story_key"].astype(str)
    features = vectors[indices, layer, :]
    cv = checked_group_splits(features, y, groups, folds, classification=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    probabilities = cross_val_predict(model, features, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return {
        "task": "detail_survival",
        "layer_index": layer,
        "n_rows": len(merged),
        "n_stories": groups.nunique(),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def derive_directions(record_vectors: np.ndarray, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    summary = frame[frame["record_type"] == "summary"].copy()
    vectors_by_key = {
        (row.story_key, row.condition, int(row.summary_id)): record_vectors[int(row.record_row)]
        for row in summary.itertuples(index=False)
    }
    directions: dict[str, np.ndarray] = {}
    for name, positive, negative in [
        ("older_minus_actual", "older", "actual"),
        ("actual_minus_younger", "actual", "younger"),
        ("older_minus_younger", "older", "younger"),
    ]:
        differences = []
        for story_key, condition, summary_id in vectors_by_key:
            if condition != positive:
                continue
            positive_vector = vectors_by_key[(story_key, positive, summary_id)]
            negative_vector = vectors_by_key.get((story_key, negative, summary_id))
            if negative_vector is not None:
                differences.append(positive_vector - negative_vector)
        if differences:
            direction = np.mean(differences, axis=0)
            norms = np.linalg.norm(direction, axis=1, keepdims=True)
            directions[name] = direction
            directions[f"{name}_unit"] = direction / np.maximum(norms, 1e-12)
    return directions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grouped layerwise probes and derive matched age directions.")
    parser.add_argument("--input-dir", type=Path, default=Path("mechanistic_interpretability/outputs/02_representation_extraction"))
    parser.add_argument("--output-dir", type=Path, default=Path("mechanistic_interpretability/outputs/04_layerwise_probes"))
    parser.add_argument("--retention-csv", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--direction-train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    vectors, metadata, layers = load_shards(args.input_dir)
    record_vectors, records = aggregate_records(vectors, metadata)
    results = []
    retention = pd.read_csv(args.retention_csv, low_memory=False) if args.retention_csv and args.retention_csv.exists() else None
    for layer_index, model_layer in enumerate(layers):
        for result in [
            condition_probe(record_vectors, records, layer_index, args.folds),
            age_probe(record_vectors, records, layer_index, args.folds),
            survival_probe(vectors, metadata, retention, layer_index, args.folds) if retention is not None else None,
        ]:
            if result is not None:
                result["model_layer"] = int(model_layer)
                results.append(result)
    story_split = assign_story_split(
        records["story_key"].astype(str),
        seed=args.seed,
        train_fraction=args.direction_train_fraction,
    )
    train_stories = set(story_split.loc[story_split["split"].eq("train"), "story_key"])
    direction_records = records[records["story_key"].astype(str).isin(train_stories)].copy()
    directions = derive_directions(record_vectors, direction_records)
    if not directions:
        raise RuntimeError("No age directions could be estimated from the training-story split")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "layerwise_probe_metrics.csv", pd.DataFrame(results))
    write_npz(
        args.output_dir / "age_directions.npz",
        layers=layers,
        **directions,
    )
    write_csv(args.output_dir / "record_feature_index.csv", records)
    save_story_split(args.output_dir / "story_split.csv", story_split)
    write_run_provenance(args.output_dir, args, extra={"layers": layers.astype(int).tolist()})
    print(pd.DataFrame(results).to_string(index=False))
    print(f"Directions: {sorted(directions)}")
    print(story_split["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
