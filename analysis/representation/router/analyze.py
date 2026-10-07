from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from shared.io import write_csv


def parse_array(value: object) -> list[float]:
    try:
        result = json.loads(str(value))
        return result if isinstance(result, list) else []
    except Exception:
        return []


def js_divergence(left: Counter[int], right: Counter[int]) -> float:
    experts = sorted(set(left) | set(right))
    if not experts:
        return float("nan")
    p = np.asarray([left[item] for item in experts], dtype=float)
    q = np.asarray([right[item] for item in experts], dtype=float)
    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    midpoint = 0.5 * (p + q)
    p_term = np.where(p > 0, p * np.log2(p / np.maximum(midpoint, 1e-12)), 0.0)
    q_term = np.where(q > 0, q * np.log2(q / np.maximum(midpoint, 1e-12)), 0.0)
    return float(0.5 * (p_term.sum() + q_term.sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MoE routing by age condition and detail.")
    parser.add_argument("--input-dir", type=Path, default=Path("results/features/representation_extraction/causal_lm/detail_extraction"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/representation/router"))
    args = parser.parse_args()

    frames = [pd.read_csv(path, low_memory=False) for path in sorted(glob.glob(str(args.input_dir / "metadata_*.csv")))]
    if not frames:
        raise FileNotFoundError("No representation metadata shards found")
    metadata = pd.concat(frames, ignore_index=True)
    rows = []
    for record in metadata.itertuples(index=False):
        entropies = parse_array(record.router_entropy)
        experts = parse_array(record.router_top_expert)
        probabilities = parse_array(record.router_top_probability)
        for router_layer, entropy in enumerate(entropies):
            rows.append(
                {
                    "story_key": record.story_key,
                    "record_id": record.record_id,
                    "record_type": record.record_type,
                    "condition": record.condition,
                    "detail_index": record.detail_index,
                    "router_layer": router_layer,
                    "entropy": entropy,
                    "top_expert": int(experts[router_layer]) if router_layer < len(experts) else -1,
                    "top_probability": probabilities[router_layer] if router_layer < len(probabilities) else np.nan,
                }
            )
    routing = pd.DataFrame(rows)
    if routing.empty:
        raise RuntimeError("The model did not return router logits; inspect the smoke-test report")
    detail_weighted_summary = (
        routing.groupby(["record_type", "condition", "router_layer"], as_index=False)
        .agg(
            n_details=("detail_index", "size"),
            n_stories=("story_key", "nunique"),
            mean_entropy=("entropy", "mean"),
            mean_top_probability=("top_probability", "mean"),
            unique_top_experts=("top_expert", "nunique"),
        )
    )
    story_level = (
        routing.groupby(["story_key", "record_type", "condition", "router_layer"], as_index=False)
        .agg(mean_entropy=("entropy", "mean"), mean_top_probability=("top_probability", "mean"), n_details=("detail_index", "size"))
    )
    summary = (
        story_level.groupby(["record_type", "condition", "router_layer"], as_index=False)
        .agg(n_stories=("story_key", "nunique"), mean_entropy=("mean_entropy", "mean"), mean_top_probability=("mean_top_probability", "mean"), story_sd_entropy=("mean_entropy", "std"))
    )
    divergences = []
    summary_routing = routing[routing["record_type"] == "summary"]
    for layer, group in summary_routing.groupby("router_layer"):
        distributions = {}
        for condition, values in group.groupby("condition"):
            story_probabilities = []
            experts = sorted(values["top_expert"].astype(int).unique())
            for _story, story_rows in values.groupby("story_key"):
                counts = story_rows["top_expert"].astype(int).value_counts(normalize=True)
                story_probabilities.append({expert: float(counts.get(expert, 0.0)) for expert in experts})
            distributions[condition] = Counter({expert: float(np.mean([row[expert] for row in story_probabilities])) for expert in experts})
        for left, right in [("actual", "older"), ("younger", "actual"), ("younger", "older")]:
            if left in distributions and right in distributions:
                divergences.append(
                    {
                        "router_layer": layer,
                        "condition_left": left,
                        "condition_right": right,
                        "expert_js_divergence": js_divergence(distributions[left], distributions[right]),
                    }
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "detail_router_metrics.csv", routing)
    write_csv(args.output_dir / "story_router_metrics.csv", story_level)
    write_csv(args.output_dir / "condition_router_summary.csv", summary)
    write_csv(args.output_dir / "detail_weighted_condition_router_summary.csv", detail_weighted_summary)
    write_csv(args.output_dir / "expert_distribution_divergence.csv", pd.DataFrame(divergences))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
