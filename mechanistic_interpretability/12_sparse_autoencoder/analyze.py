from __future__ import annotations

import argparse
import glob
import heapq
from pathlib import Path

import numpy as np
import pandas as pd

from mechanistic_interpretability.shared.sae import load_sae
from mechanistic_interpretability.shared.io import write_csv, write_json
from mechanistic_interpretability.shared.research import train_story_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and interpret SAE features by developmental condition.")
    parser.add_argument("--activation-root", type=Path, default=Path("mechanistic_interpretability/outputs/12_sparse_autoencoder/activations"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-features", type=int, default=200)
    parser.add_argument("--examples-per-feature", type=int, default=10)
    parser.add_argument(
        "--story-split",
        type=Path,
        default=Path("mechanistic_interpretability/outputs/04_layerwise_probes/story_split.csv"),
    )
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, payload = load_sae(str(args.checkpoint), device)
    config = payload["config"]; layer = int(config["layer"]); n_features = int(config["n_features"])
    mean = payload["mean"].to(device); scale = float(payload["scale"])
    sums = {condition: np.zeros(n_features, dtype=np.float64) for condition in ["younger", "actual", "older"]}
    counts = {condition: 0 for condition in sums}; firing = np.zeros(n_features, dtype=np.int64)
    mse_sum = cosine_sum = l0_sum = total = 0.0
    paths = [Path(path) for path in sorted(glob.glob(str(args.activation_root / f"layer_{layer:02d}" / "activations_*.npy")))]
    for path in paths:
        values = np.load(path).astype(np.float32); metadata = pd.read_csv(path.with_name(path.name.replace("activations_", "metadata_").replace(".npy", ".csv")), low_memory=False)
        for start in range(0, len(values), args.batch_size):
            batch = torch.from_numpy(values[start : start + args.batch_size]).to(device)
            normalized = (batch - mean) / scale
            with torch.inference_mode(): reconstruction, features = sae(normalized)
            feature_np = features.float().cpu().numpy(); conditions = metadata.iloc[start : start + len(batch)]["condition"].fillna("").astype(str).to_numpy()
            for condition in sums:
                mask = conditions == condition
                if mask.any(): sums[condition] += feature_np[mask].sum(axis=0); counts[condition] += int(mask.sum())
            firing += (feature_np > 0).sum(axis=0)
            mse_sum += float(F.mse_loss(reconstruction, normalized, reduction="sum")); cosine_sum += float(F.cosine_similarity(reconstruction, normalized, dim=-1).sum()); l0_sum += float((features > 0).sum()); total += len(batch)
    training_stories = train_story_keys(args.story_split)
    training_sums = {condition: np.zeros(n_features, dtype=np.float64) for condition in sums}
    training_counts = {condition: 0 for condition in sums}
    for path in paths:
        values = np.load(path).astype(np.float32); metadata = pd.read_csv(path.with_name(path.name.replace("activations_", "metadata_").replace(".npy", ".csv")), low_memory=False)
        for start in range(0, len(values), args.batch_size):
            batch = torch.from_numpy(values[start : start + args.batch_size]).to(device)
            with torch.inference_mode(): features = sae.encode((batch - mean) / scale).float().cpu().numpy()
            batch_metadata = metadata.iloc[start : start + len(batch)]
            batch_training = batch_metadata["story_key"].astype(str).isin(training_stories).to_numpy()
            batch_conditions = batch_metadata["condition"].fillna("").astype(str).to_numpy()
            for condition in training_sums:
                mask = batch_training & (batch_conditions == condition)
                if mask.any():
                    training_sums[condition] += features[mask].sum(axis=0)
                    training_counts[condition] += int(mask.sum())
    missing_conditions = [condition for condition in ["younger", "older"] if training_counts[condition] == 0]
    if missing_conditions:
        raise ValueError(f"No training-split activations for conditions: {missing_conditions}")
    training_means = {
        key: value / max(training_counts[key], 1) for key, value in training_sums.items()
    }
    training_effect = training_means["older"] - training_means["younger"]
    selected = np.argsort(np.abs(training_effect))[-args.top_features:][::-1]
    heaps = {int(feature): [] for feature in selected}
    for path in paths:
        values = np.load(path).astype(np.float32)
        metadata = pd.read_csv(
            path.with_name(path.name.replace("activations_", "metadata_").replace(".npy", ".csv")),
            low_memory=False,
        )
        for start in range(0, len(values), args.batch_size):
            batch = torch.from_numpy(values[start : start + args.batch_size]).to(device)
            with torch.inference_mode():
                features = sae.encode((batch - mean) / scale).float().cpu().numpy()
            batch_metadata = metadata.iloc[start : start + len(batch)]
            batch_training = batch_metadata["story_key"].astype(str).isin(training_stories).to_numpy()
            for local_row, global_row in enumerate(range(start, start + len(batch))):
                if not batch_training[local_row]:
                    continue
                for feature in selected:
                    value = float(features[local_row, feature])
                    if value <= 0:
                        continue
                    row = metadata.iloc[global_row]
                    item = (
                        value,
                        path.name,
                        int(global_row),
                        str(row.get("story_key", "")),
                        str(row.get("token", "")),
                        str(row.get("condition", "")),
                    )
                    heap = heaps[int(feature)]
                    if len(heap) < args.examples_per_feature:
                        heapq.heappush(heap, item)
                    elif value > heap[0][0]:
                        heapq.heapreplace(heap, item)
    feature_frame = pd.DataFrame({"feature": np.arange(n_features), "younger_mean": training_means["younger"], "actual_mean": training_means["actual"], "older_mean": training_means["older"], "older_minus_younger": training_effect, "firing_rate": firing / max(int(total), 1), "selected_for_intervention": np.isin(np.arange(n_features), selected)})
    examples = [{"feature": feature, "activation": item[0], "shard": item[1], "row": item[2], "story_key": item[3], "token": item[4], "condition": item[5]} for feature, heap in heaps.items() for item in sorted(heap, reverse=True)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "feature_statistics.csv", feature_frame)
    write_csv(args.output_dir / "top_activating_examples.csv", pd.DataFrame(examples))
    summary = {"layer": layer, "tokens": int(total), "mse_per_dimension": mse_sum / (total * config["d_model"]), "mean_cosine": cosine_sum / total, "mean_l0": l0_sum / total, "dead_feature_fraction": float((firing == 0).mean()), "condition_counts": counts, "feature_selection_split": "train", "feature_selection_condition_counts": training_counts, "story_split": str(args.story_split)}
    write_json(args.output_dir / "evaluation.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
