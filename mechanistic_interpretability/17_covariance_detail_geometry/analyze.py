from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from mechanistic_interpretability.shared.io import write_csv, write_npz
from mechanistic_interpretability.shared.research import benjamini_hochberg, write_run_provenance


@dataclass
class GeometryResult:
    covariance_cka: float
    mean_principal_angle_degrees: float
    maximum_principal_angle_degrees: float
    generalized_log_ratio_mean_abs: float
    retained_shrinkage: float
    omitted_shrinkage: float
    eigenvalues: np.ndarray
    retained_eigenvectors: np.ndarray
    omitted_eigenvectors: np.ndarray


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes"})


def covariance_cka(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first, ord="fro") * np.linalg.norm(second, ord="fro")
    return float(np.sum(first * second) / denominator) if denominator > 0 else np.nan


def leading_eigenspace(covariance: np.ndarray, rank: int) -> np.ndarray:
    values, vectors = np.linalg.eigh(covariance)
    usable = min(rank, vectors.shape[1])
    return vectors[:, np.argsort(values)[-usable:]]


def principal_angles(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    singular_values = np.linalg.svd(first.T @ second, compute_uv=False)
    return np.degrees(np.arccos(np.clip(singular_values, -1.0, 1.0)))


def fit_covariance(values: np.ndarray) -> tuple[np.ndarray, float]:
    estimator = LedoitWolf(assume_centered=False).fit(values)
    return estimator.covariance_.astype(np.float64), float(estimator.shrinkage_)


def compute_geometry(retained: np.ndarray, omitted: np.ndarray, subspace_rank: int) -> GeometryResult:
    retained_covariance, retained_shrinkage = fit_covariance(retained)
    omitted_covariance, omitted_shrinkage = fit_covariance(omitted)
    retained_space = leading_eigenspace(retained_covariance, subspace_rank)
    omitted_space = leading_eigenspace(omitted_covariance, subspace_rank)
    angles = principal_angles(retained_space, omitted_space)
    eigenvalues, eigenvectors = eigh(retained_covariance, omitted_covariance, check_finite=False)
    eigenvalues = np.clip(eigenvalues, np.finfo(float).tiny, None)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    rank = min(subspace_rank, eigenvectors.shape[1])
    return GeometryResult(
        covariance_cka=covariance_cka(retained_covariance, omitted_covariance),
        mean_principal_angle_degrees=float(np.mean(angles)),
        maximum_principal_angle_degrees=float(np.max(angles)),
        generalized_log_ratio_mean_abs=float(np.mean(np.abs(np.log(eigenvalues)))),
        retained_shrinkage=retained_shrinkage,
        omitted_shrinkage=omitted_shrinkage,
        eigenvalues=eigenvalues,
        retained_eigenvectors=eigenvectors[:, :rank],
        omitted_eigenvectors=eigenvectors[:, -rank:],
    )


def grouped_decoder_auc(
    values: np.ndarray,
    labels: np.ndarray,
    stories: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> float:
    class_counts = np.bincount(labels.astype(int), minlength=2)
    split_count = min(folds, len(np.unique(stories)), int(class_counts.min()))
    if split_count < 2:
        return np.nan
    splitter = StratifiedGroupKFold(n_splits=split_count, shuffle=True, random_state=seed)
    predictions = np.full(len(labels), np.nan, dtype=float)
    for train, test in splitter.split(values, labels, groups=stories):
        if len(np.unique(labels[train])) < 2 or len(np.unique(labels[test])) < 2:
            continue
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
        classifier.fit(values[train], labels[train])
        predictions[test] = classifier.predict_proba(values[test])[:, 1]
    usable = np.isfinite(predictions)
    if usable.sum() < 2 or len(np.unique(labels[usable])) < 2:
        return np.nan
    return float(roc_auc_score(labels[usable], predictions[usable]))


def load_deletion_shards(input_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    eos_parts: list[np.ndarray] = []
    mean_parts: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    reference_layers: np.ndarray | None = None
    offset = 0
    for feature_text in sorted(glob.glob(str(input_dir / "deletion_features_*.npz"))):
        feature_path = Path(feature_text)
        suffix = feature_path.stem.removeprefix("deletion_features_")
        metadata_path = input_dir / f"deletion_metadata_{suffix}.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for {feature_path}")
        payload = np.load(feature_path, allow_pickle=False)
        layers = payload["layers"]
        if reference_layers is None:
            reference_layers = layers
        elif not np.array_equal(reference_layers, layers):
            raise ValueError(f"Layer mismatch in {feature_path}")
        metadata = pd.read_csv(metadata_path, low_memory=False)
        eos = payload["eos_deltas"]
        mean = payload["mean_deltas"]
        if len(metadata) != len(eos) or len(metadata) != len(mean):
            raise ValueError(f"Row-count mismatch in {feature_path}")
        metadata = metadata.copy()
        metadata["deletion_global_row"] = np.arange(len(metadata), dtype=np.int64) + offset
        offset += len(metadata)
        eos_parts.append(eos)
        mean_parts.append(mean)
        metadata_parts.append(metadata)
    if not eos_parts or reference_layers is None:
        raise FileNotFoundError(f"No deletion feature shards found in {input_dir}")
    return (
        np.concatenate(eos_parts, axis=0),
        np.concatenate(mean_parts, axis=0),
        reference_layers,
        pd.concat(metadata_parts, ignore_index=True),
    )


def load_retention(path: Path, label_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"story_key", "orig_idx", label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Retention table is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["detail_index"] = pd.to_numeric(frame["orig_idx"], errors="raise").astype(int)
    frame["retained"] = as_bool(frame[label_column])
    if "summary_condition" not in frame:
        for candidate in ["age_info_condition", "target_age_label", "condition"]:
            if candidate in frame:
                frame["summary_condition"] = frame[candidate].fillna("").astype(str).str.strip()
                break
        else:
            frame["summary_condition"] = "all"
    if "target_age" in frame:
        frame["target_age"] = pd.to_numeric(frame["target_age"], errors="coerce")
    if "source_age" in frame:
        frame["source_age"] = pd.to_numeric(frame["source_age"], errors="coerce")
    identity = [
        column
        for column in [
            "story_key",
            "summary_condition",
            "target_age",
            "compression_condition",
            "age_info_condition",
            "summary_id",
            "detail_index",
        ]
        if column in frame
    ]
    return frame.drop_duplicates(identity)


def merge_labels(metadata: pd.DataFrame, retention: pd.DataFrame) -> pd.DataFrame:
    deletion = metadata.copy()
    deletion["story_key"] = deletion["story_key"].astype(str)
    deletion["detail_index"] = pd.to_numeric(deletion["detail_index"], errors="raise").astype(int)
    deletion = deletion.drop_duplicates(["story_key", "detail_index"])
    labels = retention.copy()
    labels["story_key"] = labels["story_key"].astype(str)
    merged = labels.merge(
        deletion[["story_key", "detail_index", "deletion_global_row", "deleted_token_count"]],
        on=["story_key", "detail_index"],
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise RuntimeError("No deletion vectors matched retention labels")
    return merged


def parse_groupings(value: str) -> list[list[str]]:
    return [[column.strip() for column in item.split("+") if column.strip()] for item in value.split(",") if item.strip()]


def group_name(columns: list[str], key: Any) -> tuple[str, dict[str, Any]]:
    values = key if isinstance(key, tuple) else (key,)
    metadata = dict(zip(columns, values))
    return "+".join(columns), metadata


def within_story_permutation(labels: np.ndarray, stories: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = labels.copy()
    for story in np.unique(stories):
        indices = np.flatnonzero(stories == story)
        shuffled[indices] = rng.permutation(shuffled[indices])
    return shuffled


def bootstrap_indices(stories: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(stories)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(stories == story) for story in sampled])


def infer_metric(
    values: np.ndarray,
    labels: np.ndarray,
    stories: np.ndarray,
    observed: GeometryResult,
    *,
    subspace_rank: int,
    bootstrap_iterations: int,
    permutation_iterations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    metrics = {
        "covariance_separation": 1.0 - observed.covariance_cka,
        "mean_principal_angle_degrees": observed.mean_principal_angle_degrees,
    }
    bootstrap: dict[str, list[float]] = {key: [] for key in metrics}
    for _ in range(bootstrap_iterations):
        indices = bootstrap_indices(stories, rng)
        sample_labels = labels[indices]
        if sample_labels.all() or (~sample_labels).all():
            continue
        result = compute_geometry(values[indices][sample_labels], values[indices][~sample_labels], subspace_rank)
        bootstrap["covariance_separation"].append(1.0 - result.covariance_cka)
        bootstrap["mean_principal_angle_degrees"].append(result.mean_principal_angle_degrees)
    null: dict[str, list[float]] = {key: [] for key in metrics}
    for _ in range(permutation_iterations):
        shuffled = within_story_permutation(labels, stories, rng)
        if shuffled.all() or (~shuffled).all():
            continue
        result = compute_geometry(values[shuffled], values[~shuffled], subspace_rank)
        null["covariance_separation"].append(1.0 - result.covariance_cka)
        null["mean_principal_angle_degrees"].append(result.mean_principal_angle_degrees)
    output: dict[str, float] = {}
    for metric, observed_value in metrics.items():
        boot = np.asarray(bootstrap[metric], dtype=float)
        permutation = np.asarray(null[metric], dtype=float)
        output[f"{metric}_ci_low"] = float(np.quantile(boot, 0.025)) if len(boot) else np.nan
        output[f"{metric}_ci_high"] = float(np.quantile(boot, 0.975)) if len(boot) else np.nan
        output[f"{metric}_permutation_p"] = (
            float((1 + np.sum(permutation >= observed_value)) / (len(permutation) + 1))
            if len(permutation)
            else np.nan
        )
        output[f"{metric}_null_mean"] = float(np.mean(permutation)) if len(permutation) else np.nan
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare shrinkage covariance geometry of retained versus omitted detail-deletion vectors."
    )
    parser.add_argument("--deletion-dir", type=Path, required=True)
    parser.add_argument("--retention-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-column", default="retained_structured")
    parser.add_argument("--poolings", default="eos,mean")
    parser.add_argument("--geometry-modes", default="raw,direction")
    parser.add_argument("--groupings", default="summary_condition,summary_condition+target_age")
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--pca-fit-rows", type=int, default=20000)
    parser.add_argument("--subspace-rank", type=int, default=10)
    parser.add_argument("--minimum-per-class", type=int, default=30)
    parser.add_argument("--decoder-folds", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--permutation-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eos, mean, layers, deletion_metadata = load_deletion_shards(args.deletion_dir)
    retention = load_retention(args.retention_csv, args.label_column)
    labels = merge_labels(deletion_metadata, retention)
    vector_rows = labels["deletion_global_row"].astype(int).to_numpy()
    groupings = parse_groupings(args.groupings)
    missing_group_columns = sorted({column for columns in groupings for column in columns} - set(labels.columns))
    if missing_group_columns:
        raise ValueError(f"Grouping columns missing from retention data: {missing_group_columns}")
    rng = np.random.default_rng(args.seed)
    geometry_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    score_parts: list[pd.DataFrame] = []
    direction_vectors: list[np.ndarray] = []
    direction_metadata: list[dict[str, Any]] = []
    arrays = {"eos": eos, "mean": mean}

    for pooling in [item.strip() for item in args.poolings.split(",") if item.strip()]:
        if pooling not in arrays:
            raise ValueError(f"Unknown pooling {pooling!r}")
        for layer_position, model_layer in enumerate(layers.tolist()):
            original_values = arrays[pooling][vector_rows, layer_position].astype(np.float32)
            for geometry_mode in [item.strip() for item in args.geometry_modes.split(",") if item.strip()]:
                if geometry_mode == "direction":
                    norms = np.linalg.norm(original_values, axis=1, keepdims=True)
                    values = original_values / np.maximum(norms, np.finfo(np.float32).eps)
                elif geometry_mode == "raw":
                    values = original_values
                else:
                    raise ValueError(f"Unknown geometry mode {geometry_mode!r}")
                fit_count = min(args.pca_fit_rows, len(values)) if args.pca_fit_rows > 0 else len(values)
                fit_indices = rng.choice(len(values), size=fit_count, replace=False)
                component_count = min(args.pca_components, values.shape[1], fit_count - 1)
                if component_count < args.subspace_rank:
                    raise ValueError("PCA components must be at least the requested subspace rank")
                pca = PCA(n_components=component_count, svd_solver="randomized", random_state=args.seed)
                pca.fit(values[fit_indices])
                projected = pca.transform(values).astype(np.float64)

                for grouping_columns in groupings:
                    grouped = labels.groupby(grouping_columns, dropna=False, sort=True)
                    for key, frame in grouped:
                        indices = frame.index.to_numpy(dtype=int)
                        retained_mask = labels.loc[indices, "retained"].to_numpy(dtype=bool)
                        retained_count = int(retained_mask.sum())
                        omitted_count = int((~retained_mask).sum())
                        if min(retained_count, omitted_count) < args.minimum_per_class:
                            continue
                        group_values = projected[indices]
                        result = compute_geometry(
                            group_values[retained_mask],
                            group_values[~retained_mask],
                            args.subspace_rank,
                        )
                        grouping_name, grouping_metadata = group_name(grouping_columns, key)
                        inference = infer_metric(
                            group_values,
                            retained_mask,
                            labels.loc[indices, "story_key"].astype(str).to_numpy(),
                            result,
                            subspace_rank=args.subspace_rank,
                            bootstrap_iterations=args.bootstrap_iterations,
                            permutation_iterations=args.permutation_iterations,
                            rng=rng,
                        )
                        decoder_auc = grouped_decoder_auc(
                            group_values,
                            retained_mask,
                            labels.loc[indices, "story_key"].astype(str).to_numpy(),
                            folds=args.decoder_folds,
                            seed=args.seed,
                        )
                        base = {
                            "pooling": pooling,
                            "model_layer": int(model_layer),
                            "geometry_mode": geometry_mode,
                            "grouping": grouping_name,
                            **grouping_metadata,
                            "n_rows": len(indices),
                            "n_stories": int(frame["story_key"].nunique()),
                            "n_retained": retained_count,
                            "n_omitted": omitted_count,
                            "retention_rate": retained_count / len(indices),
                            "pca_components": component_count,
                            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                            "subspace_rank": min(args.subspace_rank, component_count),
                            "covariance_cka": result.covariance_cka,
                            "covariance_separation": 1.0 - result.covariance_cka,
                            "mean_principal_angle_degrees": result.mean_principal_angle_degrees,
                            "maximum_principal_angle_degrees": result.maximum_principal_angle_degrees,
                            "generalized_log_ratio_mean_abs": result.generalized_log_ratio_mean_abs,
                            "retained_shrinkage": result.retained_shrinkage,
                            "omitted_shrinkage": result.omitted_shrinkage,
                            "retention_decoder_story_cv_auc_global_pca": decoder_auc,
                            **inference,
                        }
                        if "orig_centrality" in frame:
                            centrality = pd.to_numeric(frame["orig_centrality"], errors="coerce").to_numpy()
                            base["mean_centrality_retained"] = float(np.nanmean(centrality[retained_mask]))
                            base["mean_centrality_omitted"] = float(np.nanmean(centrality[~retained_mask]))
                            base["centrality_selection_difference"] = (
                                base["mean_centrality_retained"] - base["mean_centrality_omitted"]
                            )
                        geometry_rows.append(base)
                        for eigen_rank, eigenvalue in enumerate(result.eigenvalues, start=1):
                            spectrum_rows.append(
                                {
                                    **{key: value for key, value in base.items() if not key.endswith(("_ci_low", "_ci_high", "_permutation_p", "_null_mean"))},
                                    "eigen_rank": eigen_rank,
                                    "generalized_eigenvalue": float(eigenvalue),
                                    "log_generalized_eigenvalue": float(np.log(eigenvalue)),
                                }
                            )
                        if grouping_columns == ["summary_condition"]:
                            retained_directions = pca.components_.T @ result.retained_eigenvectors
                            omitted_directions = pca.components_.T @ result.omitted_eigenvectors
                            direction_start = len(direction_vectors)
                            for direction_type, direction_matrix in [
                                ("retained", retained_directions),
                                ("omitted", omitted_directions),
                            ]:
                                for rank_index in range(direction_matrix.shape[1]):
                                    vector = direction_matrix[:, rank_index]
                                    vector = vector / max(np.linalg.norm(vector), np.finfo(float).eps)
                                    direction_vectors.append(vector.astype(np.float32))
                                    direction_metadata.append(
                                        {
                                            "direction_row": len(direction_vectors) - 1,
                                            "pooling": pooling,
                                            "model_layer": int(model_layer),
                                            "geometry_mode": geometry_mode,
                                            "summary_condition": grouping_metadata.get("summary_condition"),
                                            "direction_type": direction_type,
                                            "direction_rank": rank_index + 1,
                                        }
                                    )
                            centered_values = values[indices] - pca.mean_
                            retained_scores = np.sum((centered_values @ retained_directions) ** 2, axis=1)
                            omitted_scores = np.sum((centered_values @ omitted_directions) ** 2, axis=1)
                            score_frame = frame[[column for column in ["story_key", "detail_index", "summary_condition", "summary_id", "source_age", "target_age", "retained", "orig_centrality", "orig_position_norm", "best_cosine", "transport_mass"] if column in frame]].copy()
                            score_frame["pooling"] = pooling
                            score_frame["model_layer"] = int(model_layer)
                            score_frame["geometry_mode"] = geometry_mode
                            score_frame["deletion_norm"] = np.linalg.norm(original_values[indices], axis=1)
                            score_frame["retained_subspace_energy"] = retained_scores
                            score_frame["omitted_subspace_energy"] = omitted_scores
                            score_frame["relative_retained_geometry"] = np.log(
                                (retained_scores + np.finfo(float).eps) / (omitted_scores + np.finfo(float).eps)
                            )
                            score_frame["direction_start_row"] = direction_start
                            score_parts.append(score_frame)

    geometry = pd.DataFrame(geometry_rows)
    if geometry.empty:
        raise RuntimeError("No groups met the minimum retained/omitted sample requirements")
    p_columns = [column for column in geometry if column.endswith("_permutation_p")]
    for column in p_columns:
        geometry[f"{column.removesuffix('_p')}_q_bh"] = benjamini_hochberg(geometry[column])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "covariance_geometry.csv", geometry)
    write_csv(args.output_dir / "generalized_eigenspectrum.csv", pd.DataFrame(spectrum_rows))
    if score_parts:
        pd.concat(score_parts, ignore_index=True).to_csv(
            args.output_dir / "detail_covariance_scores.csv.gz", index=False, compression="gzip"
        )
    if direction_vectors:
        write_npz(
            args.output_dir / "generalized_directions.npz",
            directions=np.stack(direction_vectors),
        )
        write_csv(args.output_dir / "generalized_direction_metadata.csv", pd.DataFrame(direction_metadata))
    write_run_provenance(
        args.output_dir,
        args,
        extra={
            "matched_retention_rows": len(labels),
            "stories": int(labels["story_key"].nunique()),
            "geometry_rows": len(geometry),
            "deletion_layers": layers.tolist(),
        },
    )
    print(f"Matched retention rows: {len(labels)}")
    print(f"Stories: {labels['story_key'].nunique()}")
    print(f"Geometry comparisons: {len(geometry)}")
    print(f"Wrote: {args.output_dir / 'covariance_geometry.csv'}")


if __name__ == "__main__":
    main()
