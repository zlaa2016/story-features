from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from mechanistic_interpretability.shared.io import write_csv


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def load_global_shards(input_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    eos_parts = []
    mean_parts = []
    metadata_parts = []
    layers = None
    offset = 0
    for path_text in sorted(glob.glob(str(input_dir / "global_features_*.npz"))):
        path = Path(path_text)
        metadata_path = input_dir / path.name.replace("global_features_", "global_metadata_").replace(".npz", ".csv")
        if not metadata_path.exists():
            continue
        payload = np.load(path, allow_pickle=False)
        shard_layers = payload["layers"]
        if layers is None:
            layers = shard_layers
        elif not np.array_equal(layers, shard_layers):
            raise ValueError(f"Layer mismatch in {path}")
        eos = payload["eos_vectors"].astype(np.float32)
        mean = payload["mean_vectors"].astype(np.float32)
        metadata = pd.read_csv(metadata_path, low_memory=False)
        if len(eos) != len(mean) or len(eos) != len(metadata):
            raise ValueError(f"Vector/metadata mismatch in {path}")
        metadata["global_row"] = np.arange(len(metadata)) + offset
        offset += len(metadata)
        eos_parts.append(eos)
        mean_parts.append(mean)
        metadata_parts.append(metadata)
    if not eos_parts or layers is None:
        raise FileNotFoundError(f"No complete global feature shards in {input_dir}")
    return np.concatenate(eos_parts), np.concatenate(mean_parts), pd.concat(metadata_parts, ignore_index=True), layers


def paired_geometry(
    eos: np.ndarray,
    mean: np.ndarray,
    metadata: pd.DataFrame,
    layers: np.ndarray,
) -> pd.DataFrame:
    originals = metadata[metadata["record_type"].eq("original")].drop_duplicates("story_key").copy()
    summaries = metadata[metadata["record_type"].eq("summary")].copy()
    pairs = summaries.merge(
        originals[["story_key", "global_row", "token_count"]].rename(
            columns={"global_row": "original_row", "token_count": "original_token_count"}
        ),
        on="story_key",
        how="inner",
        validate="many_to_one",
    )
    rows = []
    for layer_position, layer in enumerate(layers):
        original_indices = pairs["original_row"].astype(int).to_numpy()
        summary_indices = pairs["global_row"].astype(int).to_numpy()
        eos_original = eos[original_indices, layer_position]
        eos_summary = eos[summary_indices, layer_position]
        mean_original = mean[original_indices, layer_position]
        mean_summary = mean[summary_indices, layer_position]
        eos_cosine = cosine_rows(eos_original, eos_summary)
        mean_cosine = cosine_rows(mean_original, mean_summary)
        eos_displacement = np.linalg.norm(eos_original - eos_summary, axis=-1)
        mean_displacement = np.linalg.norm(mean_original - mean_summary, axis=-1)
        for pair_index, (_, pair) in enumerate(pairs.iterrows()):
            rows.append({
                "record_id": pair["record_id"],
                "story_key": pair["story_key"],
                "condition": pair["condition"],
                "summary_id": pair["summary_id"],
                "source_age": pair["source_age"],
                "target_age": pair["target_age"],
                "age_shift": pd.to_numeric(pair["target_age"], errors="coerce") - pd.to_numeric(pair["source_age"], errors="coerce"),
                "compression_ratio": pair["compression_ratio"],
                "original_token_count": pair["original_token_count"],
                "summary_token_count": pair["token_count"],
                "layer": int(layer),
                "eos_cosine": float(eos_cosine[pair_index]),
                "mean_control_cosine": float(mean_cosine[pair_index]),
                "eos_minus_mean_cosine": float(eos_cosine[pair_index] - mean_cosine[pair_index]),
                "eos_displacement_norm": float(eos_displacement[pair_index]),
                "mean_control_displacement_norm": float(mean_displacement[pair_index]),
            })
    return pd.DataFrame(rows)


def load_deletion_shards(input_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray] | None:
    eos_parts = []
    mean_parts = []
    metadata_parts = []
    layers = None
    offset = 0
    for path_text in sorted(glob.glob(str(input_dir / "deletion_features_*.npz"))):
        path = Path(path_text)
        metadata_path = input_dir / path.name.replace("deletion_features_", "deletion_metadata_").replace(".npz", ".csv")
        if not metadata_path.exists():
            continue
        payload = np.load(path, allow_pickle=False)
        shard_layers = payload["layers"]
        if layers is None:
            layers = shard_layers
        elif not np.array_equal(layers, shard_layers):
            raise ValueError(f"Layer mismatch in {path}")
        eos_part = payload["eos_deltas"].astype(np.float32)
        mean_part = payload["mean_deltas"].astype(np.float32)
        metadata = pd.read_csv(metadata_path, low_memory=False)
        metadata["deletion_global_row"] = np.arange(len(metadata)) + offset
        offset += len(metadata)
        eos_parts.append(eos_part)
        mean_parts.append(mean_part)
        metadata_parts.append(metadata)
    if not eos_parts or layers is None:
        return None
    return np.concatenate(eos_parts), np.concatenate(mean_parts), pd.concat(metadata_parts, ignore_index=True), layers


def deletion_alignment(
    deletion: tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray],
    eos: np.ndarray,
    mean: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    deletion_eos, deletion_mean, deletion_metadata, layers = deletion
    originals = metadata[metadata["record_type"].eq("original")][["story_key", "global_row"]].drop_duplicates("story_key")
    summaries = metadata[metadata["record_type"].eq("summary")].copy()
    pairs = deletion_metadata.merge(originals, on="story_key", how="inner", validate="many_to_one").merge(
        summaries,
        on="story_key",
        how="inner",
        suffixes=("_detail", "_summary"),
        validate="many_to_many",
    )
    rows = []
    for layer_position, layer in enumerate(layers):
        deletion_indices = pairs["deletion_global_row"].astype(int).to_numpy()
        original_indices = pairs["global_row_detail"].astype(int).to_numpy()
        summary_indices = pairs["global_row_summary"].astype(int).to_numpy()
        eos_compression = eos[original_indices, layer_position] - eos[summary_indices, layer_position]
        mean_compression = mean[original_indices, layer_position] - mean[summary_indices, layer_position]
        eos_alignment = cosine_rows(deletion_eos[deletion_indices, layer_position], eos_compression)
        mean_alignment = cosine_rows(deletion_mean[deletion_indices, layer_position], mean_compression)
        for pair_index, (_, pair) in enumerate(pairs.iterrows()):
            rows.append({
                "story_key": pair["story_key"],
                "detail_index": pair["detail_index"],
                "detail_text": pair["detail_text"],
                "summary_record_id": pair["record_id_summary"],
                "condition": pair["condition"],
                "summary_id": pair["summary_id"],
                "source_age": pair["source_age"],
                "target_age": pair["target_age"],
                "layer": int(layer),
                "eos_detail_compression_alignment": float(eos_alignment[pair_index]),
                "mean_control_detail_compression_alignment": float(mean_alignment[pair_index]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one-pass EOS story-summary geometry against mean-pooling controls.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--deletion-input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    eos, mean, metadata, layers = load_global_shards(args.input_dir)
    geometry = paired_geometry(eos, mean, metadata, layers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "story_summary_global_geometry.csv", geometry)
    summary = geometry.groupby(["layer", "condition"], dropna=False).agg(
        n_pairs=("record_id", "size"),
        n_stories=("story_key", "nunique"),
        mean_eos_cosine=("eos_cosine", "mean"),
        mean_pooling_control_cosine=("mean_control_cosine", "mean"),
        mean_eos_minus_control=("eos_minus_mean_cosine", "mean"),
    ).reset_index()
    write_csv(args.output_dir / "story_summary_global_geometry_summary.csv", summary)
    deletion = load_deletion_shards(args.deletion_input_dir or args.input_dir)
    if deletion is not None:
        detail_frame = deletion_alignment(deletion, eos, mean, metadata)
        write_csv(args.output_dir / "detail_deletion_compression_alignment.csv", detail_frame)
    print(f"Wrote {len(geometry)} story-summary-layer comparisons")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
