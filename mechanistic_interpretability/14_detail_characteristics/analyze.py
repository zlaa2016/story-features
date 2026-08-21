from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit

from mechanistic_interpretability.shared.io import write_csv
from mechanistic_interpretability.shared.research import benjamini_hochberg, write_run_provenance


IDENTIFIERS = {"story_key", "video_id", "source_age", "source_age_range", "detail_index", "detail_text"}
OUTCOMES = [
    "best_cosine", "transport_mass", "whole_summary_lexical_token_recall", "whole_summary_lexical_lcs_recall",
]


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.lower().isin(["1", "true", "t", "yes"])


def standardized_within_story(frame: pd.DataFrame, column: str) -> pd.Series:
    mean = frame.groupby("story_key")[column].transform("mean")
    std = frame.groupby("story_key")[column].transform("std").replace(0, np.nan)
    return ((frame[column] - mean) / std).fillna(0.0)


def binary_model(frame: pd.DataFrame, feature: str) -> dict[str, float | str | int] | None:
    work = frame.dropna(subset=[feature, "target_age"]).copy()
    work["retained"] = as_bool(work["retained_structured"]).astype(int)
    work["group"] = work[["story_key", "condition", "summary_id", "summary_variant"]].astype(str).agg("::".join, axis=1)
    work = work[work.groupby("group")["retained"].transform("nunique").gt(1)].copy()
    if len(work) < 100 or work[feature].nunique() < 2:
        return None
    work["z_feature"] = standardized_within_story(work, feature)
    age_std = work["target_age"].std()
    work["z_target_age"] = (work["target_age"] - work["target_age"].mean()) / (age_std if age_std else 1.0)
    work["feature_x_age"] = work["z_feature"] * work["z_target_age"]
    design = work[["z_feature", "feature_x_age", "detail_position"]].astype(float)
    try:
        fit = ConditionalLogit(work["retained"], design, groups=work["group"]).fit(method="bfgs", maxiter=300, disp=False)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return {
        "feature": feature,
        "outcome": "retained_structured",
        "term": "feature_x_target_age",
        "estimate": float(fit.params["feature_x_age"]),
        "std_error": float(fit.bse["feature_x_age"]),
        "p_value": float(fit.pvalues["feature_x_age"]),
        "n_rows": len(work),
        "n_stories": work["story_key"].nunique(),
        "model": "conditional_logit_story_summary_strata",
    }


def continuous_model(frame: pd.DataFrame, feature: str, outcome: str) -> dict[str, float | str | int] | None:
    work = frame.dropna(subset=[feature, outcome, "target_age"]).copy()
    if len(work) < 100 or work[feature].nunique() < 2:
        return None
    work["group"] = work[["story_key", "condition", "summary_id", "summary_variant"]].astype(str).agg("::".join, axis=1)
    work["z_feature"] = standardized_within_story(work, feature)
    age_std = work["target_age"].std()
    work["z_target_age"] = (work["target_age"] - work["target_age"].mean()) / (age_std if age_std else 1.0)
    work["feature_x_age"] = work["z_feature"] * work["z_target_age"]
    for column in [outcome, "z_feature", "feature_x_age", "detail_position"]:
        work[f"within_{column}"] = work[column] - work.groupby("group")[column].transform("mean")
    design = work[["within_z_feature", "within_feature_x_age", "within_detail_position"]].astype(float)
    if np.linalg.matrix_rank(design.to_numpy()) < 2:
        return None
    fit = sm.OLS(work[f"within_{outcome}"].astype(float), design).fit(
        cov_type="cluster", cov_kwds={"groups": work["story_key"]}
    )
    return {
        "feature": feature,
        "outcome": outcome,
        "term": "feature_x_target_age",
        "estimate": float(fit.params["within_feature_x_age"]),
        "std_error": float(fit.bse["within_feature_x_age"]),
        "p_value": float(fit.pvalues["within_feature_x_age"]),
        "n_rows": len(work),
        "n_stories": work["story_key"].nunique(),
        "model": "within_summary_ols_story_clustered",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test which detail characteristics predict preservation and whether selection changes with target age.")
    parser.add_argument("--input-dir", type=Path, default=Path("mechanistic_interpretability/outputs/14_detail_characteristics"))
    parser.add_argument("--retention-csv", type=Path, default=Path("mechanistic_interpretability/outputs/13_structured_detail_alignment/original_detail_retention.csv"))
    args = parser.parse_args()

    features = pd.read_csv(args.input_dir / "detail_characteristics.csv", low_memory=False)
    retention = pd.read_csv(args.retention_csv, low_memory=False).rename(columns={"orig_idx": "detail_index"})
    frame = retention.merge(features, on=["story_key", "detail_index"], how="inner", suffixes=("", "_feature"), validate="many_to_one")
    excluded = IDENTIFIERS | {"detail_position", "word_count", "sentence_count"}
    feature_columns = [
        column for column in features.select_dtypes(include=[np.number]).columns
        if column not in excluded and not column.endswith("_coverage") and features[column].notna().sum() >= 100
    ]
    rows = []
    for feature in feature_columns:
        binary = binary_model(frame, feature)
        if binary:
            rows.append(binary)
        for outcome in OUTCOMES:
            if outcome in frame:
                result = continuous_model(frame, feature, outcome)
                if result:
                    rows.append(result)
    results = pd.DataFrame(rows)
    if len(results):
        results["q_value_bh"] = benjamini_hochberg(results["p_value"])
    write_csv(args.input_dir / "detail_characteristic_preservation_models.csv", results)
    write_run_provenance(args.input_dir, args, extra={"merged_rows": len(frame), "features_tested": feature_columns})
    print(f"Merged rows: {len(frame)}; stories: {frame['story_key'].nunique()}; tests: {len(results)}")


if __name__ == "__main__":
    main()
