from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau, rankdata

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.io import write_csv


IDENTITY_COLUMNS = [
    "record_id",
    "story_key",
    "condition",
    "summary_id",
    "source_age",
    "target_age",
]


def write_gzip_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".csv.gz")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, compression="gzip")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def centrality(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.full(len(values), np.nan, dtype=np.float32)
    similarities = values @ values.T
    return ((similarities.sum(axis=1) - np.diag(similarities)) / (len(values) - 1)).astype(np.float32)


def percentile(values: np.ndarray) -> np.ndarray:
    if not len(values):
        return np.asarray([], dtype=np.float32)
    if len(values) == 1 or not np.isfinite(values).any():
        return np.full(len(values), 0.5, dtype=np.float32)
    filled = np.where(np.isfinite(values), values, np.nanmedian(values))
    return ((rankdata(filled, method="average") - 1) / (len(filled) - 1)).astype(np.float32)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    usable = np.isfinite(left) & np.isfinite(right)
    if usable.sum() < 3 or np.std(left[usable]) == 0 or np.std(right[usable]) == 0:
        return np.nan
    return float(np.corrcoef(left[usable], right[usable])[0, 1])


def cosine_to_vector(values: np.ndarray, vector: np.ndarray) -> np.ndarray:
    normalized_vector = vector.astype(np.float32, copy=False)
    normalized_vector /= max(float(np.linalg.norm(normalized_vector)), 1e-12)
    return values @ normalized_vector


def selected_layer_position(layers: np.ndarray, requested_layer: int) -> tuple[int, int]:
    if requested_layer == -1:
        return len(layers) - 1, int(layers[-1])
    matches = np.flatnonzero(layers.astype(int) == requested_layer)
    if not len(matches):
        raise ValueError(f"Layer {requested_layer} is not available; found {layers.tolist()}")
    return int(matches[0]), requested_layer


def shard_suffixes(input_dir: Path) -> list[str]:
    suffixes = []
    for path_text in sorted(glob.glob(str(input_dir / "contextual_detail_features_*.npz"))):
        path = Path(path_text)
        suffix = path.stem.removeprefix("contextual_detail_features_")
        required = [
            input_dir / f"contextual_detail_metadata_{suffix}.csv",
            input_dir / f"global_features_{suffix}.npz",
            input_dir / f"global_metadata_{suffix}.csv",
        ]
        if all(path.exists() for path in required):
            suffixes.append(suffix)
    if not suffixes:
        raise FileNotFoundError(f"No complete global/contextual-detail shards in {input_dir}")
    return suffixes


def load_shard(
    input_dir: Path,
    suffix: str,
    representation: str,
    requested_layer: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, int]:
    global_payload = np.load(input_dir / f"global_features_{suffix}.npz", allow_pickle=False)
    detail_payload = np.load(input_dir / f"contextual_detail_features_{suffix}.npz", allow_pickle=False)
    if not np.array_equal(global_payload["layers"], detail_payload["layers"]):
        raise ValueError(f"Global/detail layer mismatch in shard {suffix}")
    layer_position, layer = selected_layer_position(global_payload["layers"], requested_layer)
    if representation == "last":
        global_vectors = global_payload["eos_vectors"][:, layer_position]
        detail_vectors = detail_payload["last_token_vectors"][:, layer_position]
    else:
        global_vectors = global_payload["mean_vectors"][:, layer_position]
        detail_vectors = detail_payload["mean_control_vectors"][:, layer_position]
    global_metadata = pd.read_csv(input_dir / f"global_metadata_{suffix}.csv", low_memory=False)
    detail_metadata = pd.read_csv(input_dir / f"contextual_detail_metadata_{suffix}.csv", low_memory=False)
    if len(global_metadata) != len(global_vectors) or len(detail_metadata) != len(detail_vectors):
        raise ValueError(f"Feature/metadata row mismatch in shard {suffix}")
    return (
        global_metadata,
        normalize_rows(global_vectors),
        detail_metadata,
        normalize_rows(detail_vectors),
        layer,
    )


def load_isolated_shard(
    input_dir: Path | None,
    suffix: str,
    representation: str,
    requested_layer: int,
) -> tuple[pd.DataFrame, np.ndarray] | None:
    if input_dir is None:
        return None
    feature_path = input_dir / f"isolated_detail_features_{suffix}.npz"
    metadata_path = input_dir / f"isolated_detail_metadata_{suffix}.csv"
    if not feature_path.exists() or not metadata_path.exists():
        return None
    payload = np.load(feature_path, allow_pickle=False)
    layer_position, _ = selected_layer_position(payload["layers"], requested_layer)
    key = "eos_vectors" if representation == "last" else "mean_vectors"
    vectors = normalize_rows(payload[key][:, layer_position])
    metadata = pd.read_csv(metadata_path, low_memory=False)
    if len(metadata) != len(vectors):
        raise ValueError(f"Isolated feature/metadata row mismatch in shard {suffix}")
    return metadata, vectors


def process_summary(
    original_metadata: pd.DataFrame,
    original_vectors: np.ndarray,
    original_global: np.ndarray,
    summary_metadata: pd.DataFrame,
    summary_vectors: np.ndarray,
    summary_global: np.ndarray,
    layer: int,
    representation: str,
    original_isolated: np.ndarray | None = None,
    summary_isolated: np.ndarray | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    original_centrality = centrality(original_vectors)
    summary_centrality = centrality(summary_vectors)
    original_percentile = percentile(original_centrality)
    summary_percentile = percentile(summary_centrality)
    cross_similarity = summary_vectors @ original_vectors.T
    summary_to_original = cross_similarity.argmax(axis=1)
    original_to_summary = cross_similarity.argmax(axis=0)
    grounding_cosine = cross_similarity[np.arange(len(summary_vectors)), summary_to_original]
    coverage_cosine = cross_similarity[original_to_summary, np.arange(len(original_vectors))]
    selected_count = np.bincount(summary_to_original, minlength=len(original_vectors))
    summary_rows, original_rows = linear_sum_assignment(-cross_similarity)
    unique_original_for_summary = np.full(len(summary_vectors), -1, dtype=int)
    unique_cosine_for_summary = np.full(len(summary_vectors), np.nan, dtype=np.float32)
    unique_original_for_summary[summary_rows] = original_rows
    unique_cosine_for_summary[summary_rows] = cross_similarity[summary_rows, original_rows]

    original_to_original_global = cosine_to_vector(original_vectors, original_global)
    original_to_summary_global = cosine_to_vector(original_vectors, summary_global)
    summary_to_original_global = cosine_to_vector(summary_vectors, original_global)
    summary_to_summary_global = cosine_to_vector(summary_vectors, summary_global)
    isolated_original_to_original_global = (
        cosine_to_vector(original_isolated, original_global)
        if original_isolated is not None
        else np.full(len(original_vectors), np.nan)
    )
    isolated_original_to_summary_global = (
        cosine_to_vector(original_isolated, summary_global)
        if original_isolated is not None
        else np.full(len(original_vectors), np.nan)
    )
    isolated_summary_to_original_global = (
        cosine_to_vector(summary_isolated, original_global)
        if summary_isolated is not None
        else np.full(len(summary_vectors), np.nan)
    )
    isolated_summary_to_summary_global = (
        cosine_to_vector(summary_isolated, summary_global)
        if summary_isolated is not None
        else np.full(len(summary_vectors), np.nan)
    )
    original_context_cosine = (
        np.sum(original_vectors * original_isolated, axis=1)
        if original_isolated is not None
        else np.full(len(original_vectors), np.nan)
    )
    summary_context_cosine = (
        np.sum(summary_vectors * summary_isolated, axis=1)
        if summary_isolated is not None
        else np.full(len(summary_vectors), np.nan)
    )
    original_texts = original_metadata["detail_text"].fillna("").astype(str).tolist()
    summary_texts = summary_metadata["detail_text"].fillna("").astype(str).tolist()
    identity = {column: summary_metadata.iloc[0].get(column) for column in IDENTITY_COLUMNS}
    identity.update({"layer": layer, "representation": representation})

    summary_output: list[dict[str, object]] = []
    for summary_position, original_position in enumerate(summary_to_original):
        unique_position = unique_original_for_summary[summary_position]
        summary_output.append({
            **identity,
            "summary_detail_index": int(summary_metadata.iloc[summary_position]["detail_index"]),
            "summary_detail_text": summary_texts[summary_position],
            "summary_centrality": float(summary_centrality[summary_position]),
            "summary_centrality_percentile": float(summary_percentile[summary_position]),
            "grounded_original_detail_index": int(original_metadata.iloc[original_position]["detail_index"]),
            "grounded_original_detail_text": original_texts[original_position],
            "grounding_cosine": float(grounding_cosine[summary_position]),
            "grounded_original_centrality": float(original_centrality[original_position]),
            "grounded_original_centrality_percentile": float(original_percentile[original_position]),
            "centrality_percentile_change": float(summary_percentile[summary_position] - original_percentile[original_position]),
            "summary_detail_to_original_global_cosine": float(summary_to_original_global[summary_position]),
            "summary_detail_to_summary_global_cosine": float(summary_to_summary_global[summary_position]),
            "summary_detail_global_weight_change": float(summary_to_summary_global[summary_position] - summary_to_original_global[summary_position]),
            "isolated_summary_detail_to_original_global_cosine": float(isolated_summary_to_original_global[summary_position]),
            "isolated_summary_detail_to_summary_global_cosine": float(isolated_summary_to_summary_global[summary_position]),
            "isolated_summary_detail_global_weight_change": float(
                isolated_summary_to_summary_global[summary_position]
                - isolated_summary_to_original_global[summary_position]
            ),
            "contextual_isolated_cosine": float(summary_context_cosine[summary_position]),
            "contextualization_shift": float(1 - summary_context_cosine[summary_position]),
            "unique_original_detail_index": int(original_metadata.iloc[unique_position]["detail_index"]) if unique_position >= 0 else -1,
            "unique_match_cosine": float(unique_cosine_for_summary[summary_position]),
        })

    original_output: list[dict[str, object]] = []
    for original_position, summary_position in enumerate(original_to_summary):
        original_output.append({
            **identity,
            "original_detail_index": int(original_metadata.iloc[original_position]["detail_index"]),
            "original_detail_text": original_texts[original_position],
            "original_centrality": float(original_centrality[original_position]),
            "original_centrality_percentile": float(original_percentile[original_position]),
            "nearest_summary_detail_index": int(summary_metadata.iloc[summary_position]["detail_index"]),
            "nearest_summary_detail_text": summary_texts[summary_position],
            "coverage_cosine": float(coverage_cosine[original_position]),
            "selected_by_summary_count": int(selected_count[original_position]),
            "selected_by_any_summary_detail": bool(selected_count[original_position] > 0),
            "nearest_summary_centrality": float(summary_centrality[summary_position]),
            "nearest_summary_centrality_percentile": float(summary_percentile[summary_position]),
            "original_detail_to_original_global_cosine": float(original_to_original_global[original_position]),
            "original_detail_to_summary_global_cosine": float(original_to_summary_global[original_position]),
            "original_detail_global_weight_change": float(original_to_summary_global[original_position] - original_to_original_global[original_position]),
            "isolated_original_detail_to_original_global_cosine": float(
                isolated_original_to_original_global[original_position]
            ),
            "isolated_original_detail_to_summary_global_cosine": float(
                isolated_original_to_summary_global[original_position]
            ),
            "isolated_original_detail_global_weight_change": float(
                isolated_original_to_summary_global[original_position]
                - isolated_original_to_original_global[original_position]
            ),
            "contextual_minus_isolated_summary_alignment": float(
                original_to_summary_global[original_position]
                - isolated_original_to_summary_global[original_position]
            ),
            "contextual_isolated_cosine": float(original_context_cosine[original_position]),
            "contextualization_shift": float(1 - original_context_cosine[original_position]),
        })

    chronology = kendalltau(np.arange(len(summary_to_original)), summary_to_original).statistic if len(summary_to_original) > 1 else np.nan
    story_output = {
        **identity,
        "n_original_details": len(original_vectors),
        "n_summary_details": len(summary_vectors),
        "mean_grounding_cosine": float(np.mean(grounding_cosine)),
        "mean_coverage_cosine": float(np.mean(coverage_cosine)),
        "proportion_original_details_selected": float(np.mean(selected_count > 0)),
        "centrality_coverage_correlation": safe_correlation(original_centrality, coverage_cosine),
        "centrality_selection_correlation": safe_correlation(original_centrality, (selected_count > 0).astype(float)),
        "mean_grounded_original_centrality": float(np.nanmean(original_centrality[summary_to_original])),
        "mean_absolute_centrality_percentile_change": float(np.nanmean(np.abs(summary_percentile - original_percentile[summary_to_original]))),
        "mean_original_detail_global_weight_change": float(np.mean(original_to_summary_global - original_to_original_global)),
        "centrality_global_weight_change_correlation": safe_correlation(
            original_centrality, original_to_summary_global - original_to_original_global
        ),
        "mean_isolated_original_detail_global_weight_change": (
            float(np.nanmean(isolated_original_to_summary_global - isolated_original_to_original_global))
            if np.isfinite(isolated_original_to_summary_global).any()
            else np.nan
        ),
        "centrality_isolated_global_weight_change_correlation": safe_correlation(
            original_centrality,
            isolated_original_to_summary_global - isolated_original_to_original_global,
        ),
        "mean_contextual_minus_isolated_summary_alignment": (
            float(np.nanmean(original_to_summary_global - isolated_original_to_summary_global))
            if np.isfinite(isolated_original_to_summary_global).any()
            else np.nan
        ),
        "chronology_kendall_tau": float(chronology),
        "mean_unique_match_cosine": float(np.nanmean(unique_cosine_for_summary)),
        "mean_original_contextualization_shift": float(np.nanmean(1 - original_context_cosine)) if np.isfinite(original_context_cosine).any() else np.nan,
        "mean_summary_contextualization_shift": float(np.nanmean(1 - summary_context_cosine)) if np.isfinite(summary_context_cosine).any() else np.nan,
        "centrality_contextualization_correlation": safe_correlation(original_centrality, 1 - original_context_cosine),
    }
    return summary_output, original_output, story_output


def analyze_shard(
    input_dir: Path,
    isolated_input_dir: Path | None,
    suffix: str,
    representation: str,
    requested_layer: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    global_metadata, global_vectors, detail_metadata, detail_vectors, layer = load_shard(
        input_dir, suffix, representation, requested_layer
    )
    global_lookup = {str(row.record_id): index for index, row in enumerate(global_metadata.itertuples(index=False))}
    detail_groups = {
        str(record_id): np.asarray(indices, dtype=int)
        for record_id, indices in detail_metadata.groupby("record_id", sort=False).indices.items()
    }
    isolated = load_isolated_shard(isolated_input_dir, suffix, representation, requested_layer)
    isolated_lookup: dict[tuple[str, int], int] = {}
    isolated_vectors: np.ndarray | None = None
    if isolated is not None:
        isolated_metadata, isolated_vectors = isolated
        isolated_lookup = {
            (str(row.record_id), int(row.detail_index)): index
            for index, row in enumerate(isolated_metadata.itertuples(index=False))
        }
    summary_output: list[dict[str, object]] = []
    original_output: list[dict[str, object]] = []
    story_output: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for story_key, story_records in global_metadata.groupby("story_key", sort=False):
        originals = story_records[story_records["record_type"].eq("original")]
        summaries = story_records[story_records["record_type"].eq("summary")]
        if len(originals) != 1 or summaries.empty:
            continue
        original_record = originals.iloc[0]
        original_record_id = str(original_record["record_id"])
        original_indices = detail_groups.get(original_record_id)
        if original_indices is None or not len(original_indices):
            errors.append({"shard": suffix, "story_key": story_key, "error": "missing original details"})
            continue
        for _, summary_record in summaries.iterrows():
            summary_record_id = str(summary_record["record_id"])
            summary_indices = detail_groups.get(summary_record_id)
            if summary_indices is None or not len(summary_indices):
                errors.append({"shard": suffix, "story_key": story_key, "record_id": summary_record_id, "error": "missing summary details"})
                continue
            original_isolated = None
            summary_isolated = None
            if isolated_vectors is not None:
                original_control_rows = [
                    isolated_lookup.get((original_record_id, int(index)))
                    for index in detail_metadata.iloc[original_indices]["detail_index"]
                ]
                summary_control_rows = [
                    isolated_lookup.get((summary_record_id, int(index)))
                    for index in detail_metadata.iloc[summary_indices]["detail_index"]
                ]
                if all(index is not None for index in original_control_rows):
                    original_isolated = isolated_vectors[np.asarray(original_control_rows, dtype=int)]
                if all(index is not None for index in summary_control_rows):
                    summary_isolated = isolated_vectors[np.asarray(summary_control_rows, dtype=int)]
            summary_rows, original_rows, story_row = process_summary(
                detail_metadata.iloc[original_indices].reset_index(drop=True),
                detail_vectors[original_indices],
                global_vectors[global_lookup[original_record_id]],
                detail_metadata.iloc[summary_indices].reset_index(drop=True),
                detail_vectors[summary_indices],
                global_vectors[global_lookup[summary_record_id]],
                layer,
                representation,
                original_isolated,
                summary_isolated,
            )
            summary_output.extend(summary_rows)
            original_output.extend(original_rows)
            story_output.append(story_row)
    return summary_output, original_output, story_output, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure age-steered detail compression from contextual representations.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--isolated-input-dir", type=Path, default=None)
    parser.add_argument("--representation", choices=["last", "mean"], default="last")
    parser.add_argument("--layer", type=int, default=-1, help="Stored layer number; -1 selects the final stored layer.")
    args = parser.parse_args()

    summary_rows: list[dict[str, object]] = []
    original_rows: list[dict[str, object]] = []
    story_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    suffixes = shard_suffixes(args.input_dir)
    for shard_number, suffix in enumerate(suffixes, start=1):
        shard_summary, shard_original, shard_story, shard_errors = analyze_shard(
            args.input_dir, args.isolated_input_dir, suffix, args.representation, args.layer
        )
        summary_rows.extend(shard_summary)
        original_rows.extend(shard_original)
        story_rows.extend(shard_story)
        errors.extend(shard_errors)
        print(
            f"[{shard_number}/{len(suffixes)}] shard={suffix} "
            f"summaries={len(shard_story)} summary_details={len(shard_summary)}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_gzip_csv(args.output_dir / "summary_detail_representation_alignment.csv.gz", pd.DataFrame(summary_rows))
    write_gzip_csv(args.output_dir / "original_detail_representation_coverage.csv.gz", pd.DataFrame(original_rows))
    write_csv(args.output_dir / "story_representation_compression_metrics.csv", pd.DataFrame(story_rows))
    error_frame = pd.DataFrame(errors)
    if error_frame.empty:
        error_frame = pd.DataFrame(columns=["shard", "story_key", "record_id", "error"])
    write_csv(args.output_dir / "errors.csv", error_frame)
    print(f"Wrote {len(story_rows)} summary comparisons from {len(set(row['story_key'] for row in story_rows))} stories")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
