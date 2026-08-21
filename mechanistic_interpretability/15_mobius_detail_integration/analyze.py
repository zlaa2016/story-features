from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from mechanistic_interpretability.shared.io import write_csv
from mechanistic_interpretability.shared.research import benjamini_hochberg, write_run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Test within-story age changes in combinatorial detail integration.")
    parser.add_argument("--input-dir", type=Path, default=Path("mechanistic_interpretability/outputs/15_mobius_detail_integration"))
    args = parser.parse_args()

    paths = sorted(glob.glob(str(args.input_dir / "mobius_interactions_*.csv")))
    frames = [pd.read_csv(path, low_memory=False) for path in paths if Path(path).stat().st_size]
    if not frames:
        raise FileNotFoundError(f"No interaction shards in {args.input_dir}")
    details = pd.concat(frames, ignore_index=True)
    group_keys = ["story_key", "condition", "summary_id", "source_age", "target_age", "summary_detail_index", "null_similarity"]
    by_order = details.groupby(group_keys + ["interaction_order"], dropna=False).agg(
        absolute_mass=("absolute_mobius", "sum"),
        signed_mass=("mobius_coefficient", "sum"),
        positive_synergy=("positive_synergy", "sum"),
        negative_redundancy=("negative_redundancy", "sum"),
    ).reset_index()
    wide = by_order.pivot_table(index=group_keys, columns="interaction_order", values="absolute_mass", fill_value=0).reset_index()
    wide.columns = [f"order_{int(column)}_absolute_mass" if column in {1, 2, 3} else column for column in wide.columns]
    for order in [1, 2, 3]:
        column = f"order_{order}_absolute_mass"
        if column not in wide:
            wide[column] = 0.0
    total = wide[[f"order_{order}_absolute_mass" for order in [1, 2, 3]]].sum(axis=1).replace(0, np.nan)
    wide["higher_order_share"] = (wide["order_2_absolute_mass"] + wide["order_3_absolute_mass"]) / total
    summary = wide.groupby(["story_key", "condition", "summary_id", "source_age", "target_age", "null_similarity"], dropna=False).mean(numeric_only=True).reset_index()
    write_csv(args.input_dir / "mobius_summary_metrics.csv", summary)

    outcomes = ["order_1_absolute_mass", "order_2_absolute_mass", "order_3_absolute_mass", "higher_order_share"]
    results = []
    for null_similarity, null_frame in summary.groupby("null_similarity"):
        for outcome in outcomes:
            work = null_frame.dropna(subset=[outcome, "target_age"]).copy()
            if len(work) < 20 or work["story_key"].nunique() < 5:
                continue
            work["within_outcome"] = work[outcome] - work.groupby("story_key")[outcome].transform("mean")
            work["within_target_age"] = work["target_age"] - work.groupby("story_key")["target_age"].transform("mean")
            if np.isclose(work["within_target_age"].var(), 0):
                continue
            fit = sm.OLS(work["within_outcome"], work[["within_target_age"]]).fit(
                cov_type="cluster", cov_kwds={"groups": work["story_key"]}
            )
            results.append({
                "outcome": outcome,
                "null_similarity": null_similarity,
                "estimate_per_target_year": float(fit.params["within_target_age"]),
                "std_error": float(fit.bse["within_target_age"]),
                "p_value": float(fit.pvalues["within_target_age"]),
                "n_rows": len(work),
                "n_stories": work["story_key"].nunique(),
                "model": "within_story_ols_story_clustered",
            })
    models = pd.DataFrame(results)
    if len(models):
        models["q_value_bh"] = benjamini_hochberg(models["p_value"])
    write_csv(args.input_dir / "mobius_age_models.csv", models)
    write_run_provenance(args.input_dir, args, extra={"interaction_rows": len(details), "summary_rows": len(summary)})
    print(f"Interactions: {len(details)}; summary records: {len(summary)}; models: {len(models)}")


if __name__ == "__main__":
    main()
