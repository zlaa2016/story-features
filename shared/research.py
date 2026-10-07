from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

import numpy as np
import pandas as pd

from shared.io import write_csv, write_json


Row = TypeVar("Row")


def select_work_shard(
    rows: Sequence[Row],
    *,
    start: int = 0,
    limit: int = 0,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> list[Row]:
    if shard_index is not None or num_shards is not None:
        if shard_index is None or num_shards is None:
            raise ValueError("shard_index and num_shards must be provided together")
        if num_shards < 1 or not 0 <= shard_index < num_shards:
            raise ValueError("Invalid work-shard coordinates")
        return list(rows[shard_index::num_shards])
    stop = None if limit <= 0 else start + limit
    return list(rows[start:stop])


def stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}::{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_story_split(story_keys: Iterable[str], *, seed: int, train_fraction: float) -> pd.DataFrame:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between zero and one")
    unique = sorted({str(value) for value in story_keys if str(value).strip()})
    if len(unique) < 2:
        raise ValueError("At least two stories are required for a train/evaluation split")
    rows = [
        {
            "story_key": story,
            "split": "train" if stable_fraction(story, seed) < train_fraction else "evaluation",
            "split_seed": seed,
            "train_fraction": train_fraction,
        }
        for story in unique
    ]
    frame = pd.DataFrame(rows)
    if frame["split"].nunique() != 2:
        ordered = sorted(unique, key=lambda story: stable_fraction(story, seed))
        cutoff = min(max(1, round(len(ordered) * train_fraction)), len(ordered) - 1)
        train = set(ordered[:cutoff])
        frame["split"] = frame["story_key"].map(lambda story: "train" if story in train else "evaluation")
    return frame


def load_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"story_key", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Story split is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["story_key"] = frame["story_key"].astype(str)
    if frame["story_key"].duplicated().any():
        raise ValueError("Story split contains duplicate story_key rows")
    invalid = set(frame["split"].dropna().astype(str)) - {"train", "evaluation"}
    if invalid:
        raise ValueError(f"Story split contains invalid labels: {sorted(invalid)}")
    return frame


def evaluation_story_keys(path: Path) -> set[str]:
    frame = load_split(path)
    return set(frame.loc[frame["split"].eq("evaluation"), "story_key"])


def train_story_keys(path: Path) -> set[str]:
    frame = load_split(path)
    return set(frame.loc[frame["split"].eq("train"), "story_key"])


def benjamini_hochberg(values: Iterable[float]) -> np.ndarray:
    p_values = np.asarray(list(values), dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p_values))
    if not len(finite):
        return adjusted
    order = finite[np.argsort(p_values[finite])]
    ranked = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def save_story_split(path: Path, frame: pd.DataFrame) -> None:
    counts = frame["split"].value_counts()
    if not {"train", "evaluation"}.issubset(counts.index):
        raise ValueError("Story split must contain train and evaluation stories")
    write_csv(path, frame.sort_values("story_key").reset_index(drop=True))


def write_run_provenance(output_dir: Path, arguments: Namespace, *, extra: dict[str, object] | None = None) -> None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    packages = {}
    for package in ["numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "torch", "transformers"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    serialized_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(arguments).items()
    }
    payload: dict[str, object] = {
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "arguments": serialized_arguments,
        "packages": packages,
    }
    if extra:
        payload["extra"] = extra
    write_json(output_dir / "run_provenance.json", payload)
