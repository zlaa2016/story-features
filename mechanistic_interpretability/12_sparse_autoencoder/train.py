from __future__ import annotations

import argparse
import glob
import os
import random
import tempfile
from pathlib import Path

import numpy as np

from mechanistic_interpretability.shared.sae import TopKSAE
from mechanistic_interpretability.shared.io import write_json
from mechanistic_interpretability.shared.research import train_story_keys


def activation_paths(root: Path, layer: int) -> list[Path]:
    paths = [Path(path) for path in sorted(glob.glob(str(root / f"layer_{layer:02d}" / "activations_*.npy")))]
    if not paths:
        raise FileNotFoundError(f"No activation shards for layer {layer}")
    return paths


def training_values(path: Path, stories: set[str]) -> np.ndarray:
    metadata_path = path.with_name(path.name.replace("activations_", "metadata_").replace(".npy", ".csv"))
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata for {path}")
    import pandas as pd
    metadata = pd.read_csv(metadata_path, low_memory=False)
    values = np.load(path, mmap_mode="r")
    if len(values) != len(metadata):
        raise ValueError(f"Activation/metadata row mismatch for {path}")
    return np.asarray(values[metadata["story_key"].astype(str).isin(stories).to_numpy()], dtype=np.float32)


def normalization(paths: list[Path], stories: set[str]) -> tuple[np.ndarray, float, int]:
    total = None
    squared = 0.0
    count = 0
    for path in paths:
        values = training_values(path, stories)
        batch_sum = values.astype(np.float64).sum(axis=0)
        total = batch_sum if total is None else total + batch_sum
        count += len(values)
    if total is None or count == 0:
        raise ValueError("No training-split activations are available")
    mean = total / count
    for path in paths:
        values = training_values(path, stories)
        centered = values.astype(np.float32) - mean.astype(np.float32)
        squared += float(np.square(centered).sum())
    scale = float(np.sqrt(squared / (count * len(mean))))
    return mean.astype(np.float32), max(scale, 1e-6), count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TopK sparse autoencoder on residual activations.")
    parser.add_argument("--activation-root", type=Path, default=Path("mechanistic_interpretability/outputs/12_sparse_autoencoder/activations"))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--story-split", type=Path, default=Path("mechanistic_interpretability/outputs/04_layerwise_probes/story_split.csv"))
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    paths = activation_paths(args.activation_root, args.layer)
    training_stories = train_story_keys(args.story_split)
    mean, scale, n_tokens = normalization(paths, training_stories)
    if n_tokens < args.batch_size:
        raise ValueError(f"Only {n_tokens} training-split tokens are available")
    d_model = int(mean.shape[0]); n_features = d_model * args.expansion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = TopKSAE.build(d_model, n_features, args.k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    mean_tensor = torch.from_numpy(mean).to(device)
    history = []
    for epoch in range(args.epochs):
        shuffled = paths[:]; random.shuffle(shuffled)
        loss_total = cosine_total = l0_total = seen = 0.0
        for path in shuffled:
            values = training_values(path, training_stories)
            if not len(values):
                continue
            order = np.random.permutation(len(values))
            for start in range(0, len(values), args.batch_size):
                batch = torch.from_numpy(values[order[start : start + args.batch_size]]).to(device)
                normalized = (batch - mean_tensor) / scale
                reconstruction, features = sae(normalized)
                loss = F.mse_loss(reconstruction, normalized)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                with torch.no_grad(): sae.normalize_decoder()
                batch_size = len(batch)
                loss_total += float(loss.detach()) * batch_size
                cosine_total += float(F.cosine_similarity(reconstruction.detach(), normalized, dim=-1).sum())
                l0_total += float((features.detach() > 0).sum())
                seen += batch_size
        row = {"epoch": epoch + 1, "mse": loss_total / seen, "cosine": cosine_total / seen, "l0": l0_total / seen}
        history.append(row); print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(dir=args.output.parent, suffix=".pt")
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save({"state_dict": sae.state_dict(), "mean": torch.from_numpy(mean), "scale": scale, "config": {"layer": args.layer, "d_model": d_model, "n_features": n_features, "k": args.k, "relu": True, "expansion": args.expansion, "n_tokens": n_tokens, "source": "domain_trained", "fit_split": "train", "story_split": str(args.story_split), "seed": args.seed}}, temporary)
        temporary.replace(args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    write_json(args.output.with_suffix(".history.json"), history)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
