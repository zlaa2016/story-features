from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from mechanistic_interpretability.shared.io import read_jsonl
from mechanistic_interpretability.shared.io import write_csv, write_json
from mechanistic_interpretability.shared.research import train_story_keys


def distribution(rows: list[dict], temperature: float = 1.0) -> dict[int, float]:
    scores: Counter[int] = Counter()
    for row in rows:
        logits = np.asarray(row["logits"], dtype=float) / temperature
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        for token_id, weight in zip(row["token_ids"], weights):
            scores[int(token_id)] += float(weight)
    total = sum(scores.values()) or 1.0
    return {key: value / total for key, value in scores.items()}


def js_divergence(left: dict[int, float], right: dict[int, float]) -> float:
    keys = sorted(set(left) | set(right))
    p = np.asarray([left.get(key, 0.0) for key in keys]) + 1e-12
    q = np.asarray([right.get(key, 0.0) for key in keys]) + 1e-12
    p /= p.sum(); q /= q.sum(); middle = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / middle)) + 0.5 * np.sum(q * np.log(q / middle)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare verbalizable J-space content across age conditions.")
    parser.add_argument("--input-glob", default="mechanistic_interpretability/outputs/11_jacobian_lens/readouts_*.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("mechanistic_interpretability/outputs/11_jacobian_lens/analysis"))
    parser.add_argument("--story-split", type=Path, default=Path("mechanistic_interpretability/outputs/04_layerwise_probes/story_split.csv"))
    args = parser.parse_args()
    import glob

    rows = [row for path in sorted(glob.glob(args.input_glob)) for row in read_jsonl(Path(path))]
    if not rows:
        raise FileNotFoundError(f"No readouts match {args.input_glob}")
    grouped: dict[tuple, list[dict]] = {}
    token_names: dict[int, str] = {}
    for row in rows:
        token_names.update({int(token_id): str(token) for token_id, token in zip(row["token_ids"], row["tokens"])})
        key = (row["story_key"], row["record_id"], row["record_type"], row.get("condition", ""), int(row["layer"]))
        grouped.setdefault(key, []).append(row)
    records = []
    distributions = {}
    for key, group in grouped.items():
        dist = distribution(group)
        distributions[key] = dist
        records.append({"story_key": key[0], "record_id": key[1], "record_type": key[2], "condition": key[3], "layer": key[4], "jspace_entropy": -sum(value * np.log(value + 1e-12) for value in dist.values()), "jspace_unique_top_tokens": len(dist)})
    frame = pd.DataFrame(records)
    comparisons = []
    for row in frame[frame["record_type"] == "summary"].itertuples(index=False):
        original_candidates = frame[(frame.story_key == row.story_key) & (frame.record_type == "original") & (frame.layer == row.layer)]
        if original_candidates.empty:
            continue
        original = original_candidates.iloc[0]
        left = distributions[(row.story_key, original.record_id, "original", original.condition, row.layer)]
        right = distributions[(row.story_key, row.record_id, "summary", row.condition, row.layer)]
        comparisons.append({"story_key": row.story_key, "condition": row.condition, "layer": row.layer, "original_summary_js": js_divergence(left, right), "summary_jspace_entropy": row.jspace_entropy, "original_jspace_entropy": original.jspace_entropy})
    condition_counts: Counter[tuple[int, str]] = Counter()
    condition_token_sums: Counter[tuple[int, str, int]] = Counter()
    training = train_story_keys(args.story_split)
    for key, dist in distributions.items():
        story_key, _record_id, record_type, condition, layer = key
        if story_key not in training or record_type != "summary" or condition not in {"younger", "actual", "older"}:
            continue
        condition_counts[(layer, condition)] += 1
        for token_id, value in dist.items():
            condition_token_sums[(layer, condition, token_id)] += value
    token_effects = []
    all_layer_tokens = sorted({(layer, token_id) for layer, _condition, token_id in condition_token_sums})
    for layer, token_id in all_layer_tokens:
        means = {
            condition: condition_token_sums[(layer, condition, token_id)] / max(condition_counts[(layer, condition)], 1)
            for condition in ["younger", "actual", "older"]
        }
        token_effects.append({"layer": layer, "token_id": token_id, "token": token_names.get(token_id, ""), **{f"{condition}_mean": value for condition, value in means.items()}, "older_minus_younger": means["older"] - means["younger"]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "jspace_record_metrics.csv", frame)
    write_csv(args.output_dir / "jspace_original_summary_comparisons.csv", pd.DataFrame(comparisons))
    write_csv(args.output_dir / "jspace_token_condition_effects.csv", pd.DataFrame(token_effects))
    write_json(args.output_dir / "summary.json", {"readout_rows": len(rows), "record_layer_rows": len(frame), "comparison_rows": len(comparisons), "feature_selection_split": "train"})
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
