from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shared.io import write_csv
from shared.research import benjamini_hochberg, write_run_provenance


def load_records(input_dir: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    vectors = []
    metadata = []
    layers = None
    offset = 0
    for feature_text in sorted(glob.glob(str(input_dir / "features_*.npz"))):
        feature_path = Path(feature_text)
        metadata_path = input_dir / feature_path.name.replace("features_", "metadata_").replace(".npz", ".csv")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for {feature_path}")
        payload = np.load(feature_path, allow_pickle=False)
        shard = payload["vectors"].astype(np.float32)
        frame = pd.read_csv(metadata_path, low_memory=False)
        if len(frame) != len(shard):
            raise ValueError(f"Feature/metadata row mismatch for {feature_path}")
        if layers is not None and not np.array_equal(layers, payload["layers"]):
            raise ValueError(f"Layer mismatch in {feature_path}")
        frame["vector_row"] = np.arange(len(frame)) + offset
        offset += len(frame)
        vectors.append(shard)
        metadata.append(frame)
        layers = payload["layers"] if layers is None else layers
    if not vectors or layers is None:
        raise FileNotFoundError("No feature shards found")
    detail_vectors = np.concatenate(vectors)
    detail_metadata = pd.concat(metadata, ignore_index=True)
    record_vectors = []
    records = []
    for record_id, group in detail_metadata.groupby("record_id", sort=False):
        indices = group["vector_row"].astype(int).to_numpy()
        record_vectors.append(detail_vectors[indices].mean(axis=0))
        first = group.iloc[0]
        records.append(
            {
                "record_row": len(record_vectors) - 1,
                "record_id": record_id,
                "record_type": first["record_type"],
                "story_key": first["story_key"],
                "condition": first["condition"],
                "source_age": first.get("source_age"),
                "target_age": first.get("target_age"),
                "compression_ratio": first.get("compression_ratio"),
                "n_details": len(group),
                "mean_detail_tokens": pd.to_numeric(group["token_count"], errors="coerce").mean(),
            }
        )
    return np.stack(record_vectors), pd.DataFrame(records), layers


def checked_splits(y: np.ndarray, groups: np.ndarray, folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Grouped evaluation requires at least two stories")
    splits = list(GroupKFold(min(folds, len(unique_groups))).split(np.zeros(len(y)), y, groups))
    for fold, (train, test) in enumerate(splits):
        if y.dtype.kind not in "f" and len(np.unique(y[train])) < 2:
            raise ValueError(f"Training fold {fold} has only one class")
        if not len(test):
            raise ValueError(f"Test fold {fold} is empty")
    return splits


def grouped_predictions(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    predictions = np.full(len(y), np.nan, dtype=object if y.dtype.kind in "OUS" else float)
    for train, test in splits:
        model.fit(X[train], y[train])
        predictions[test] = model.predict(X[test])
    if pd.isna(predictions).any():
        raise RuntimeError("Out-of-fold predictions are incomplete")
    return predictions.astype(y.dtype if y.dtype.kind in "OUS" else float)


def shuffle_within_story(values: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = values.copy()
    for story in np.unique(groups):
        rows = np.flatnonzero(groups == story)
        shuffled[rows] = rng.permutation(shuffled[rows])
    return shuffled


def story_bootstrap_difference(
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    groups: np.ndarray,
    metric: object,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stories = np.unique(groups)
    rows = {story: np.flatnonzero(groups == story) for story in stories}
    values = []
    for _ in range(iterations):
        sample = rng.choice(stories, len(stories), replace=True)
        selected = np.concatenate([rows[story] for story in sample])
        values.append(float(metric(target[selected], second[selected]) - metric(target[selected], first[selected])))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test behavioral-representational alignment and dissociation.")
    parser.add_argument("--input-dir", type=Path, default=Path("results/features/representation_extraction/causal_lm/detail_extraction"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/representation/behavior_vs_representation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()

    vectors, records, layers = load_records(args.input_dir)
    summaries = records[
        (records["record_type"] == "summary")
        & records["condition"].isin(["younger", "actual", "older"])
    ].copy()
    summaries = summaries.dropna(subset=["source_age", "target_age", "compression_ratio", "n_details"])
    indices = summaries["record_row"].astype(int).to_numpy()
    groups = summaries["story_key"].astype(str).to_numpy()
    condition_y = summaries["condition"].astype(str).to_numpy()
    age_y = pd.to_numeric(summaries["target_age"], errors="coerce").to_numpy(float)
    behavior_columns = ["source_age", "compression_ratio", "n_details", "mean_detail_tokens"]
    behavior_X = summaries[behavior_columns].to_numpy(float)
    classification_splits = checked_splits(condition_y, groups)
    regression_splits = checked_splits(age_y, groups)
    behavior_classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=args.seed),
    )
    behavior_predictions = grouped_predictions(behavior_classifier, behavior_X, condition_y, classification_splits)
    results = [{
        "task": "condition_classification",
        "feature_set": "behavior_only",
        "model_layer": -1,
        "n_rows": len(summaries),
        "n_stories": len(np.unique(groups)),
        "balanced_accuracy": balanced_accuracy_score(condition_y, behavior_predictions),
    }]

    behavior_regressor = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    behavior_age = grouped_predictions(behavior_regressor, behavior_X, age_y, regression_splits)
    results.append({
        "task": "target_age_regression",
        "feature_set": "behavior_only",
        "model_layer": -1,
        "n_rows": len(summaries),
        "n_stories": len(np.unique(groups)),
        "r2": r2_score(age_y, behavior_age),
        "mae": mean_absolute_error(age_y, behavior_age),
    })

    prediction_rows = []
    for row, prediction, age_prediction in zip(summaries.itertuples(index=False), behavior_predictions, behavior_age):
        prediction_rows.append({"record_id": row.record_id, "story_key": row.story_key, "model_layer": -1, "feature_set": "behavior_only", "condition": row.condition, "condition_prediction": prediction, "target_age": row.target_age, "target_age_prediction": age_prediction})
    for layer_index, model_layer in enumerate(layers.astype(int)):
        representation_X = vectors[indices, layer_index, :]
        representation_classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=args.seed),
        )
        representation_predictions = grouped_predictions(
            representation_classifier, representation_X, condition_y, classification_splits
        )
        combined_X = np.concatenate([behavior_X, representation_X], axis=1)
        combined_classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=args.seed),
        )
        combined_predictions = grouped_predictions(combined_classifier, combined_X, condition_y, classification_splits)
        representation_regressor = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        representation_age = grouped_predictions(representation_regressor, representation_X, age_y, regression_splits)
        combined_regressor = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        combined_age = grouped_predictions(combined_regressor, combined_X, age_y, regression_splits)
        rng = np.random.default_rng(args.seed + int(model_layer))
        null_scores = []
        for _ in range(args.permutations):
            shuffled = shuffle_within_story(condition_y, groups, rng)
            if any(len(np.unique(shuffled[train])) < 2 for train, _ in classification_splits):
                continue
            shuffled_predictions = grouped_predictions(representation_classifier, representation_X, shuffled, classification_splits)
            null_scores.append(float(balanced_accuracy_score(shuffled, shuffled_predictions)))
        for feature_set, predictions in [
            ("representation_only", representation_predictions),
            ("behavior_plus_representation", combined_predictions),
        ]:
            results.append({
                "task": "condition_classification",
                "feature_set": feature_set,
                "model_layer": int(model_layer),
                "n_rows": len(summaries),
                "n_stories": len(np.unique(groups)),
                "balanced_accuracy": balanced_accuracy_score(condition_y, predictions),
            })
        observed = float(balanced_accuracy_score(condition_y, representation_predictions))
        results.append({"task": "condition_classification", "feature_set": "within_story_permutation_null", "model_layer": int(model_layer), "n_rows": len(summaries), "n_stories": len(np.unique(groups)), "balanced_accuracy": float(np.mean(null_scores)) if null_scores else np.nan, "null_std": float(np.std(null_scores, ddof=1)) if len(null_scores) > 1 else np.nan, "empirical_p_value": float((1 + sum(score >= observed for score in null_scores)) / (len(null_scores) + 1)) if null_scores else np.nan, "n_permutations": len(null_scores)})
        for feature_set, predictions in [
            ("representation_only", representation_age),
            ("behavior_plus_representation", combined_age),
        ]:
            results.append({
                "task": "target_age_regression",
                "feature_set": feature_set,
                "model_layer": int(model_layer),
                "n_rows": len(summaries),
                "n_stories": len(np.unique(groups)),
                "r2": r2_score(age_y, predictions),
                "mae": mean_absolute_error(age_y, predictions),
            })
        condition_ci = story_bootstrap_difference(condition_y, behavior_predictions, combined_predictions, groups, balanced_accuracy_score, args.bootstrap_iterations, args.seed + int(model_layer))
        age_ci = story_bootstrap_difference(age_y, behavior_age, combined_age, groups, lambda truth, prediction: -mean_absolute_error(truth, prediction), args.bootstrap_iterations, args.seed + int(model_layer))
        results.append({"task": "incremental_condition_classification", "feature_set": "representation_over_behavior", "model_layer": int(model_layer), "n_rows": len(summaries), "n_stories": len(np.unique(groups)), "balanced_accuracy_gain": float(balanced_accuracy_score(condition_y, combined_predictions) - balanced_accuracy_score(condition_y, behavior_predictions)), "ci_low": condition_ci[0], "ci_high": condition_ci[1]})
        results.append({"task": "incremental_target_age_regression", "feature_set": "representation_over_behavior", "model_layer": int(model_layer), "n_rows": len(summaries), "n_stories": len(np.unique(groups)), "mae_reduction": float(mean_absolute_error(age_y, behavior_age) - mean_absolute_error(age_y, combined_age)), "ci_low": age_ci[0], "ci_high": age_ci[1]})
        for row_index, row in enumerate(summaries.itertuples(index=False)):
            for feature_set, condition_prediction, age_prediction in [("representation_only", representation_predictions[row_index], representation_age[row_index]), ("behavior_plus_representation", combined_predictions[row_index], combined_age[row_index])]:
                prediction_rows.append({"record_id": row.record_id, "story_key": row.story_key, "model_layer": int(model_layer), "feature_set": feature_set, "condition": row.condition, "condition_prediction": condition_prediction, "target_age": row.target_age, "target_age_prediction": age_prediction})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(results)
    output["empirical_q_value_bh"] = benjamini_hochberg(
        pd.to_numeric(output.get("empirical_p_value", pd.Series(np.nan, index=output.index)), errors="coerce")
    )
    write_csv(args.output_dir / "behavior_representation_comparison.csv", output)
    write_csv(args.output_dir / "behavior_representation_oof_predictions.csv", pd.DataFrame(prediction_rows))
    write_run_provenance(args.output_dir, args, extra={"n_rows": len(summaries), "n_stories": int(len(np.unique(groups)))})
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
