from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mechanistic_interpretability.shared.io import write_csv
from mechanistic_interpretability.shared.research import benjamini_hochberg, write_run_provenance


def paired_inference(difference: np.ndarray, iterations: int, rng: np.random.Generator) -> dict[str, float]:
    difference = difference[np.isfinite(difference)]
    if not len(difference):
        return {"ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    bootstrap = np.asarray([
        rng.choice(difference, size=len(difference), replace=True).mean()
        for _ in range(iterations)
    ])
    null = np.asarray([
        (difference * rng.choice([-1.0, 1.0], size=len(difference))).mean()
        for _ in range(iterations)
    ])
    observed = float(difference.mean())
    return {
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "p_value": float((1 + np.sum(np.abs(null) >= abs(observed))) / (iterations + 1)),
    }


def central_derivative(group: pd.DataFrame, outcome: str) -> dict[str, float] | None:
    values = group.groupby("alpha")[outcome].mean().dropna().sort_index()
    negative = values.index[values.index < 0]
    positive = values.index[values.index > 0]
    if not len(negative) or not len(positive):
        return None
    lower = float(negative.max())
    upper = float(positive.min())
    derivative = (float(values.loc[upper]) - float(values.loc[lower])) / (upper - lower)
    baseline = float(values.loc[0.0]) if 0.0 in values.index else float(np.interp(0.0, values.index, values.values))
    return {"derivative": derivative, "baseline": baseline, "lower_alpha": lower, "upper_alpha": upper}


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate directional Jacobians and pullback Fisher cost in detail-retention space.")
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probability-epsilon", type=float, default=1e-4)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    details = pd.read_csv(args.alignment_dir / "original_detail_retention.csv", low_memory=False)
    required = {"story_key", "orig_idx", "control", "alpha", "transport_mass"}
    missing = required - set(details.columns)
    if missing:
        raise ValueError(f"Aligned interventions are missing columns: {sorted(missing)}")
    details["alpha"] = pd.to_numeric(details["alpha"], errors="coerce")
    outcomes = [column for column in ["transport_mass", "best_cosine", "whole_summary_lexical_token_recall", "whole_summary_lexical_lcs_recall"] if column in details]
    derivative_rows = []
    keys = ["story_key", "orig_idx", "control"]
    for key, group in details.groupby(keys, dropna=False):
        for outcome in outcomes:
            result = central_derivative(group, outcome)
            if result is None:
                continue
            derivative_rows.append({
                "story_key": key[0],
                "orig_idx": key[1],
                "control": key[2],
                "outcome": outcome,
                **result,
            })
    derivatives = pd.DataFrame(derivative_rows)
    if derivatives.empty:
        raise RuntimeError("No central derivatives could be estimated; include negative and positive alpha values")
    derivatives["squared_sensitivity"] = derivatives["derivative"] ** 2
    probability = derivatives["baseline"].clip(args.probability_epsilon, 1 - args.probability_epsilon)
    derivatives["bernoulli_pullback_fisher"] = np.where(
        derivatives["outcome"].eq("transport_mass"),
        derivatives["squared_sensitivity"] / (probability * (1 - probability)),
        np.nan,
    )
    story = derivatives.groupby(["story_key", "control", "outcome"], dropna=False).agg(
        mean_directional_derivative=("derivative", "mean"),
        mean_absolute_derivative=("derivative", lambda value: float(np.mean(np.abs(value)))),
        total_squared_sensitivity=("squared_sensitivity", "sum"),
        pullback_fisher_cost=("bernoulli_pullback_fisher", "sum"),
        n_details=("orig_idx", "nunique"),
    ).reset_index()
    contrasts = []
    rng = np.random.default_rng(args.seed)
    for outcome, outcome_frame in story.groupby("outcome"):
        pivot = outcome_frame.pivot_table(index="story_key", columns="control", values=["mean_absolute_derivative", "pullback_fisher_cost"])
        for metric in ["mean_absolute_derivative", "pullback_fisher_cost"]:
            if (metric, "age_direction") not in pivot or (metric, "random_direction") not in pivot:
                continue
            difference = (pivot[(metric, "age_direction")] - pivot[(metric, "random_direction")]).dropna()
            inference = paired_inference(difference.to_numpy(float), args.iterations, rng)
            contrasts.append({
                "outcome": outcome,
                "metric": metric,
                "age_minus_random": float(difference.mean()),
                "median_age_minus_random": float(difference.median()),
                "positive_fraction": float((difference > 0).mean()),
                "n_stories": len(difference),
                "model": "paired_story_bootstrap_sign_flip",
                **inference,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "detail_directional_derivatives.csv", derivatives)
    write_csv(args.output_dir / "story_causal_geometry.csv", story)
    contrast_frame = pd.DataFrame(contrasts)
    if len(contrast_frame):
        contrast_frame["q_value_bh"] = benjamini_hochberg(contrast_frame["p_value"])
    write_csv(args.output_dir / "age_direction_random_controls.csv", contrast_frame)
    write_run_provenance(args.output_dir, args, extra={"detail_derivatives": len(derivatives), "stories": int(derivatives["story_key"].nunique())})
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__ == "__main__":
    main()
