from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from mechanistic_interpretability.shared.io import write_csv
from mechanistic_interpretability.shared.research import benjamini_hochberg, write_run_provenance


covariance_analysis = importlib.import_module(
    "mechanistic_interpretability.17_covariance_detail_geometry.analyze"
)


def linear_cka(first: np.ndarray, second: np.ndarray) -> float:
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    cross = first.T @ second
    denominator = np.linalg.norm(first.T @ first, ord="fro") * np.linalg.norm(second.T @ second, ord="fro")
    return float(np.linalg.norm(cross, ord="fro") ** 2 / denominator) if denominator > 0 else np.nan


def cosine_rsm_correlation(first: np.ndarray, second: np.ndarray, maximum_rows: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    if len(first) > maximum_rows:
        indices = rng.choice(len(first), size=maximum_rows, replace=False)
        first = first[indices]
        second = second[indices]
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), np.finfo(float).eps)
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), np.finfo(float).eps)
    triangle = np.triu_indices(len(first), k=1)
    first_rsm = (first @ first.T)[triangle]
    second_rsm = (second @ second.T)[triangle]
    if np.std(first_rsm) == 0 or np.std(second_rsm) == 0:
        return np.nan
    return float(np.corrcoef(first_rsm, second_rsm)[0, 1])


def symmetric_ridge_predictivity(
    first: np.ndarray,
    second: np.ndarray,
    stories: np.ndarray,
    *,
    folds: int,
    alpha: float,
) -> tuple[float, float]:
    split_count = min(folds, len(np.unique(stories)))
    if split_count < 2:
        return np.nan, np.nan
    splitter = GroupKFold(n_splits=split_count)
    first_scores: list[float] = []
    second_scores: list[float] = []
    for train, test in splitter.split(first, groups=stories):
        first_to_second = Ridge(alpha=alpha).fit(first[train], second[train])
        second_to_first = Ridge(alpha=alpha).fit(second[train], first[train])
        first_scores.append(
            r2_score(second[test], first_to_second.predict(first[test]), multioutput="variance_weighted")
        )
        second_scores.append(
            r2_score(first[test], second_to_first.predict(second[test]), multioutput="variance_weighted")
        )
    return float(np.mean(first_scores)), float(np.mean(second_scores))


def shuffle_rows_within_story(values: np.ndarray, stories: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = values.copy()
    for story in np.unique(stories):
        indices = np.flatnonzero(stories == story)
        shuffled[indices] = values[rng.permutation(indices)]
    return shuffled


def align_models(
    metadata_a: pd.DataFrame,
    metadata_b: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["story_key", "detail_index"]
    first = metadata_a.copy()
    second = metadata_b.copy()
    first["story_key"] = first["story_key"].astype(str)
    second["story_key"] = second["story_key"].astype(str)
    first["detail_index"] = pd.to_numeric(first["detail_index"], errors="raise").astype(int)
    second["detail_index"] = pd.to_numeric(second["detail_index"], errors="raise").astype(int)
    first = first.drop_duplicates(keys).rename(columns={"deletion_global_row": "row_a"})
    second = second.drop_duplicates(keys).rename(columns={"deletion_global_row": "row_b"})
    aligned = first[keys + ["row_a"]].merge(second[keys + ["row_b"]], on=keys, how="inner", validate="one_to_one")
    if aligned.empty:
        raise RuntimeError("The two model outputs have no aligned story details")
    return aligned


def fit_projected(values: np.ndarray, components: int, seed: int) -> np.ndarray:
    count = min(components, values.shape[1], len(values) - 1)
    if count < 2:
        raise ValueError("At least two PCA components are required")
    return PCA(n_components=count, svd_solver="randomized", random_state=seed).fit_transform(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dissociate cross-model representational alignment from shared detail-retention behavior."
    )
    parser.add_argument("--model-a-dir", type=Path, required=True)
    parser.add_argument("--model-b-dir", type=Path, required=True)
    parser.add_argument("--model-a-name", required=True)
    parser.add_argument("--model-b-name", required=True)
    parser.add_argument("--retention-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-column", default="retained_structured")
    parser.add_argument("--poolings", default="eos,mean")
    parser.add_argument("--geometry-modes", default="raw,direction")
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--decoder-folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--rsm-rows", type=int, default=1000)
    parser.add_argument("--permutation-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eos_a, mean_a, layers_a, metadata_a = covariance_analysis.load_deletion_shards(args.model_a_dir)
    eos_b, mean_b, layers_b, metadata_b = covariance_analysis.load_deletion_shards(args.model_b_dir)
    aligned = align_models(metadata_a, metadata_b)
    rows_a = aligned["row_a"].astype(int).to_numpy()
    rows_b = aligned["row_b"].astype(int).to_numpy()
    stories = aligned["story_key"].astype(str).to_numpy()
    retention = covariance_analysis.load_retention(args.retention_csv, args.label_column)
    labeled = retention.merge(aligned, on=["story_key", "detail_index"], how="inner", validate="many_to_one")
    arrays_a = {"eos": eos_a, "mean": mean_a}
    arrays_b = {"eos": eos_b, "mean": mean_b}
    rng = np.random.default_rng(args.seed)
    representation_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []

    layer_pairs = list(zip(range(len(layers_a)), range(len(layers_b))))
    for pooling in [item.strip() for item in args.poolings.split(",") if item.strip()]:
        for layer_position_a, layer_position_b in layer_pairs:
            raw_a = arrays_a[pooling][rows_a, layer_position_a].astype(np.float32)
            raw_b = arrays_b[pooling][rows_b, layer_position_b].astype(np.float32)
            for mode in [item.strip() for item in args.geometry_modes.split(",") if item.strip()]:
                if mode == "direction":
                    first = raw_a / np.maximum(np.linalg.norm(raw_a, axis=1, keepdims=True), np.finfo(float).eps)
                    second = raw_b / np.maximum(np.linalg.norm(raw_b, axis=1, keepdims=True), np.finfo(float).eps)
                elif mode == "raw":
                    first, second = raw_a, raw_b
                else:
                    raise ValueError(f"Unknown geometry mode {mode!r}")
                first_projected = fit_projected(first, args.pca_components, args.seed)
                second_projected = fit_projected(second, args.pca_components, args.seed)
                observed_cka = linear_cka(first_projected, second_projected)
                null_cka = np.asarray(
                    [
                        linear_cka(
                            first_projected,
                            shuffle_rows_within_story(second_projected, stories, rng),
                        )
                        for _ in range(args.permutation_iterations)
                    ]
                )
                predict_a_b, predict_b_a = symmetric_ridge_predictivity(
                    first_projected,
                    second_projected,
                    stories,
                    folds=args.decoder_folds,
                    alpha=args.ridge_alpha,
                )
                base = {
                    "model_a": args.model_a_name,
                    "model_b": args.model_b_name,
                    "pooling": pooling,
                    "geometry_mode": mode,
                    "layer_position": layer_position_a,
                    "model_a_layer": int(layers_a[layer_position_a]),
                    "model_b_layer": int(layers_b[layer_position_b]),
                    "n_details": len(aligned),
                    "n_stories": int(aligned["story_key"].nunique()),
                    "linear_cka": observed_cka,
                    "linear_cka_within_story_null_mean": float(np.mean(null_cka)),
                    "linear_cka_within_story_permutation_p": float(
                        (1 + np.sum(null_cka >= observed_cka)) / (len(null_cka) + 1)
                    ),
                    "cosine_rsm_correlation": cosine_rsm_correlation(
                        first_projected, second_projected, args.rsm_rows, args.seed
                    ),
                    "ridge_predictivity_a_to_b": predict_a_b,
                    "ridge_predictivity_b_to_a": predict_b_a,
                }
                representation_rows.append(base)

                for condition, frame in labeled.groupby("summary_condition", dropna=False, sort=True):
                    aligned_lookup = aligned.reset_index().set_index(["story_key", "detail_index"])["index"]
                    positions = np.asarray(
                        [aligned_lookup.loc[(str(row.story_key), int(row.detail_index))] for row in frame.itertuples()],
                        dtype=int,
                    )
                    labels = frame["retained"].to_numpy(dtype=bool)
                    condition_stories = frame["story_key"].astype(str).to_numpy()
                    readout_rows.append(
                        {
                            **base,
                            "summary_condition": condition,
                            "n_behavior_rows": len(frame),
                            "retention_rate": float(labels.mean()),
                            "model_a_retention_decoder_auc": covariance_analysis.grouped_decoder_auc(
                                first_projected[positions], labels, condition_stories,
                                folds=args.decoder_folds, seed=args.seed
                            ),
                            "model_b_retention_decoder_auc": covariance_analysis.grouped_decoder_auc(
                                second_projected[positions], labels, condition_stories,
                                folds=args.decoder_folds, seed=args.seed
                            ),
                        }
                    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    representation = pd.DataFrame(representation_rows)
    representation["linear_cka_within_story_q_bh"] = benjamini_hochberg(
        representation["linear_cka_within_story_permutation_p"]
    )
    write_csv(args.output_dir / "cross_model_representation.csv", representation)
    write_csv(args.output_dir / "cross_model_behavior_readout.csv", pd.DataFrame(readout_rows))
    write_run_provenance(
        args.output_dir,
        args,
        extra={"aligned_details": len(aligned), "aligned_stories": int(aligned["story_key"].nunique())},
    )
    print(f"Aligned details: {len(aligned)}")
    print(f"Aligned stories: {aligned['story_key'].nunique()}")
    print(f"Wrote: {args.output_dir / 'cross_model_representation.csv'}")


if __name__ == "__main__":
    main()
