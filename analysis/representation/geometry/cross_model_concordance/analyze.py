from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from shared.io import write_csv
from shared.research import write_run_provenance


METRICS = [
    "whole_text_cosine", "mean_alignment_cosine", "summary_grounded_rate", "mean_source_span_length",
    "multi_detail_summary_rate", "source_detail_coverage_rate", "source_centrality_coverage",
    "retained_minus_omitted_centrality", "retained_minus_omitted_semantic_match",
    "centrality_weighted_semantic_preservation", "centrality_semantic_match_correlation",
    "order_inversion_rate",
]


def load_metrics(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame["condition"].isin(["younger", "actual", "older"])].copy()
    frame["model"] = label
    return frame.groupby(["story_key", "condition", "model"], as_index=False).mean(numeric_only=True)


def bootstrap_correlation(frame: pd.DataFrame, left: str, right: str, iterations: int, seed: int) -> tuple[float, float]:
    stories = frame["story_key"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sampled = rng.choice(stories, len(stories), replace=True)
        pieces = [frame[frame["story_key"].eq(story)] for story in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        correlation = spearmanr(boot[left], boot[right]).statistic
        if np.isfinite(correlation):
            values.append(correlation)
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))) if values else (np.nan, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether open models reproduce GPT age-conditioned compression behavior.")
    parser.add_argument("--reference", type=Path, default=Path("results/analysis/natural_language/detail_alignment/story_alignment_metrics.csv"))
    parser.add_argument("--open-model", action="append", nargs=2, metavar=("LABEL", "CSV"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/representation/geometry/cross_model_concordance"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reference = load_metrics(args.reference, "gpt_generated")
    rows = []
    for label, path in args.open_model:
        candidate = load_metrics(Path(path), label)
        merged = reference.merge(candidate, on=["story_key", "condition"], suffixes=("_gpt", "_open"))
        for metric in METRICS:
            left = f"{metric}_gpt"
            right = f"{metric}_open"
            if left not in merged or right not in merged:
                continue
            work = merged.dropna(subset=[left, right]).copy()
            if len(work) < 20:
                continue
            work[f"within_{left}"] = work[left] - work.groupby("story_key")[left].transform("mean")
            work[f"within_{right}"] = work[right] - work.groupby("story_key")[right].transform("mean")
            rho = float(spearmanr(work[f"within_{left}"], work[f"within_{right}"]).statistic)
            ci_low, ci_high = bootstrap_correlation(work, f"within_{left}", f"within_{right}", args.bootstrap, args.seed)
            rows.append({
                "open_model": label,
                "metric": metric,
                "within_story_spearman": rho,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_rows": len(work),
                "n_stories": work["story_key"].nunique(),
                "concordance_supported": bool(np.isfinite(ci_low) and ci_low > 0),
            })
    results = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "behavioral_concordance.csv", results)
    write_run_provenance(args.output_dir, args, extra={"tests": len(results)})
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
