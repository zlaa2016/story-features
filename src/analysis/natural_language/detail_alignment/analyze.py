from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr
from statsmodels.discrete.conditional_models import ConditionalLogit

from shared.io import write_csv
from shared.research import benjamini_hochberg, write_run_provenance


METRICS = [
    "whole_text_cosine",
    "mean_alignment_cosine",
    "mean_span_gain_over_single",
    "summary_grounded_rate",
    "mean_source_span_length",
    "multi_detail_summary_rate",
    "nonconsecutive_fusion_rate",
    "source_detail_coverage_rate",
    "source_centrality_coverage",
    "retained_minus_omitted_centrality",
    "mean_semantic_match_retained",
    "mean_semantic_match_omitted",
    "retained_minus_omitted_semantic_match",
    "centrality_weighted_semantic_preservation",
    "centrality_semantic_match_correlation",
    "mean_centrality_rank_shift",
    "mean_absolute_centrality_rank_shift",
    "source_summary_centrality_rank_correlation",
    "order_inversion_rate",
]

REGRESSION_COLUMNS = [
    "outcome", "term", "estimate", "std_error", "p_value", "ci_low", "ci_high",
    "n_rows", "n_stories", "model", "r_squared_within", "q_value_bh",
]
CONTRAST_COLUMNS = [
    "outcome", "contrast", "estimate", "ci_low", "ci_high", "p_value", "n_stories",
    "n_younger", "n_older", "n_both", "model", "q_value_bh",
]
CENTRALITY_COLUMNS = [
    "term", "log_odds", "odds_ratio", "std_error", "p_value", "ci_low_odds_ratio",
    "ci_high_odds_ratio", "n_rows", "n_groups", "model", "converged", "q_value_bh",
]


def clustered_fixed_effect(frame: pd.DataFrame, outcome: str) -> dict[str, Any] | None:
    columns = [outcome, "story_key", "target_age", "age_distance", "age_matched"]
    if "compression_ratio" in frame:
        columns.append("compression_ratio")
    work = frame[columns].dropna().copy()
    if len(work) < 20 or work["story_key"].nunique() < 5:
        return None
    work["age_matched"] = work["age_matched"].astype(float)
    work["target_x_distance_raw"] = work["target_age"] * work["age_distance"]
    centered_columns = [outcome, "target_age", "age_distance", "age_matched", "target_x_distance_raw"]
    if "compression_ratio" in work:
        centered_columns.append("compression_ratio")
    for column in centered_columns:
        work[f"within_{column}"] = work[column] - work.groupby("story_key")[column].transform("mean")
    proposed = [
        "within_target_age",
        "within_age_distance",
        "within_target_x_distance_raw",
        "within_age_matched",
    ]
    if "within_compression_ratio" in work:
        proposed.append("within_compression_ratio")
    raw_design = work[proposed].astype(float)
    _, triangular, pivots = qr(raw_design.to_numpy(), mode="economic", pivoting=True)
    rank = int(np.sum(np.abs(np.diag(triangular)) > 1e-10))
    predictors = [proposed[index] for index in sorted(pivots[:rank])]
    if not predictors:
        return None
    design = raw_design[predictors]
    fit = sm.OLS(work[f"within_{outcome}"].astype(float), design).fit(
        cov_type="cluster", cov_kwds={"groups": work["story_key"]}
    )
    rows = []
    for term in predictors:
        rows.append(
            {
                "outcome": outcome,
                "term": term,
                "estimate": float(fit.params[term]),
                "std_error": float(fit.bse[term]),
                "p_value": float(fit.pvalues[term]),
                "ci_low": float(fit.conf_int().loc[term, 0]),
                "ci_high": float(fit.conf_int().loc[term, 1]),
                "n_rows": len(work),
                "n_stories": work["story_key"].nunique(),
                "model": "within_story_ols_cluster_robust",
            }
        )
    return {"rows": rows, "r_squared_within": float(fit.rsquared)}


def matched_contrast(frame: pd.DataFrame, outcome: str, iterations: int, seed: int) -> dict[str, Any] | None:
    pivot = frame.pivot_table(index="story_key", columns="condition", values=outcome, aggfunc="mean")
    if "actual" not in pivot or not ({"younger", "older"} & set(pivot.columns)):
        return None
    alternative_columns = [column for column in ["younger", "older"] if column in pivot]
    alternatives = pivot[alternative_columns].mean(axis=1)
    differences = (pivot["actual"] - alternatives).dropna().to_numpy(float)
    if len(differences) < 5:
        return None
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([
        np.mean(rng.choice(differences, size=len(differences), replace=True)) for _ in range(iterations)
    ])
    signs = rng.choice([-1.0, 1.0], size=(iterations, len(differences)))
    null = np.mean(signs * differences[None, :], axis=1)
    observed = float(np.mean(differences))
    return {
        "outcome": outcome,
        "contrast": "actual_minus_available_age_mismatch_mean",
        "estimate": observed,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "p_value": float((1 + np.sum(np.abs(null) >= abs(observed))) / (iterations + 1)),
        "n_stories": len(differences),
        "n_younger": int(pivot[["actual", "younger"]].dropna().shape[0]) if "younger" in pivot else 0,
        "n_older": int(pivot[["actual", "older"]].dropna().shape[0]) if "older" in pivot else 0,
        "n_both": int(pivot[["actual", "younger", "older"]].dropna().shape[0]) if {"younger", "older"}.issubset(pivot.columns) else 0,
        "model": "paired_story_bootstrap_sign_flip",
    }


def centrality_selection(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.dropna(
        subset=["retained_structured", "orig_centrality", "orig_position_norm", "source_age", "target_age"]
    ).copy()
    work["group"] = (
        work["story_key"].astype(str)
        + "::"
        + work["condition"].astype(str)
        + "::"
        + work["summary_id"].astype(str)
        + "::"
        + work["summary_variant"].astype(str)
    )
    varying = work.groupby("group")["retained_structured"].transform("nunique").gt(1)
    work = work[varying].copy()
    if len(work) < 100:
        return pd.DataFrame()
    centrality_mean = work.groupby("story_key")["orig_centrality"].transform("mean")
    centrality_std = work.groupby("story_key")["orig_centrality"].transform("std").replace(0, np.nan)
    work["z_orig_centrality"] = ((work["orig_centrality"] - centrality_mean) / centrality_std).fillna(0)
    work["age_shift"] = work["target_age"] - work["source_age"]
    for column in ["orig_position_norm", "source_age", "target_age", "age_shift"]:
        std = work[column].std()
        work[f"z_{column}"] = (work[column] - work[column].mean()) / (std if std else 1.0)
    work["centrality_x_source_age"] = work["z_orig_centrality"] * work["z_source_age"]
    work["centrality_x_target_age"] = work["z_orig_centrality"] * work["z_target_age"]
    work["centrality_x_age_shift"] = work["z_orig_centrality"] * work["z_age_shift"]

    def fit_model(data: pd.DataFrame, predictors: list[str], label: str) -> list[dict[str, Any]]:
        varying_groups = data.groupby("group")["retained_structured"].transform("nunique").gt(1)
        data = data[varying_groups].copy()
        if len(data) < 100 or data["group"].nunique() < 10:
            return []
        model = ConditionalLogit(
            data["retained_structured"].astype(int),
            data[predictors].astype(float),
            groups=data["group"],
        )
        try:
            fit = model.fit(method="bfgs", maxiter=300, disp=False)
        except (ValueError, np.linalg.LinAlgError):
            return []
        confidence = fit.conf_int()
        converged = bool(getattr(fit, "mle_retvals", {}).get("converged", True))
        return [
            {
                "term": term,
                "log_odds": float(fit.params[term]),
                "odds_ratio": float(np.exp(fit.params[term])),
                "std_error": float(fit.bse[term]),
                "p_value": float(fit.pvalues[term]),
                "ci_low_odds_ratio": float(np.exp(confidence.loc[term, 0])),
                "ci_high_odds_ratio": float(np.exp(confidence.loc[term, 1])),
                "n_rows": len(data),
                "n_groups": data["group"].nunique(),
                "model": label,
                "converged": converged,
            }
            for term in predictors
        ]

    rows = fit_model(
        work,
        ["z_orig_centrality", "z_orig_position_norm"],
        "conditional_logit_centrality_baseline",
    )
    rows.extend(
        fit_model(
            work,
            [
                "z_orig_centrality",
                "centrality_x_source_age",
                "centrality_x_age_shift",
                "z_orig_position_norm",
            ],
            "conditional_logit_developmental_compression",
        )
    )
    for condition, condition_frame in work.groupby("condition", dropna=False):
        rows.extend(
            fit_model(
                condition_frame,
                ["z_orig_centrality", "centrality_x_source_age", "z_orig_position_norm"],
                f"conditional_logit_condition_{condition}",
            )
        )
    return pd.DataFrame(rows)


def retention_set_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "story_key", "condition", "summary_id", "orig_idx", "retained_structured", "centrality_pct"
    }
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    work = frame.copy()
    if "summary_variant" not in work:
        work["summary_variant"] = 1
    record_columns = ["story_key", "condition", "summary_id", "summary_variant"]
    records: list[dict[str, Any]] = []
    output: list[dict[str, Any]] = []
    for key, detail_frame in work.groupby(record_columns, dropna=False, sort=False):
        detail_frame = detail_frame.sort_values("orig_idx")
        indices = detail_frame["orig_idx"].astype(int).to_numpy()
        retained_mask = detail_frame["retained_structured"].astype(bool).to_numpy()
        centrality = detail_frame["centrality_pct"].astype(float).to_numpy()
        n_details = len(indices)
        retained = set(indices[retained_mask].tolist())
        retained_count = len(retained)
        top_central = set(indices[np.argsort(centrality)[-retained_count:]].tolist()) if retained_count else set()
        first = set(indices[:retained_count].tolist())
        denominator = max(retained_count, 1)
        metadata = dict(zip(record_columns, key if isinstance(key, tuple) else (key,)))
        output.append(
            {
                "row_type": "retention_baseline",
                **metadata,
                "n_details": n_details,
                "n_retained": retained_count,
                "centrality_top_k_precision": len(retained & top_central) / denominator,
                "first_k_precision": len(retained & first) / denominator,
                "random_k_expected_precision": retained_count / n_details if n_details else np.nan,
                "retained_mean_centrality_pct": float(np.mean(centrality[retained_mask])) if retained_count else np.nan,
                "omitted_mean_centrality_pct": float(np.mean(centrality[~retained_mask])) if (~retained_mask).any() else np.nan,
            }
        )
        records.append(
            {
                **metadata,
                "retained": retained,
                "centrality": dict(zip(indices.tolist(), centrality.tolist())),
                "n_details": n_details,
            }
        )

    record_frame = pd.DataFrame(records)
    for (story_key, summary_id), story_records in record_frame.groupby(["story_key", "summary_id"], dropna=False):
        for (_, left), (_, right) in combinations(story_records.iterrows(), 2):
            if left["condition"] == right["condition"]:
                continue
            left_set = left["retained"]
            right_set = right["retained"]
            union = left_set | right_set
            intersection = left_set & right_set
            centrality = left["centrality"]
            weighted_union = sum(centrality[index] for index in union)
            weighted_intersection = sum(centrality[index] for index in intersection)
            expected_intersection = (
                len(left_set) * len(right_set) / left["n_details"] if left["n_details"] else np.nan
            )
            expected_union = len(left_set) + len(right_set) - expected_intersection
            output.append(
                {
                    "row_type": "condition_pair",
                    "story_key": story_key,
                    "summary_id": summary_id,
                    "condition_left": left["condition"],
                    "condition_right": right["condition"],
                    "n_details": left["n_details"],
                    "n_retained_left": len(left_set),
                    "n_retained_right": len(right_set),
                    "retention_jaccard": len(intersection) / len(union) if union else 1.0,
                    "centrality_weighted_jaccard": (
                        weighted_intersection / weighted_union if weighted_union else 1.0
                    ),
                    "random_expected_jaccard": (
                        expected_intersection / expected_union if expected_union else np.nan
                    ),
                }
            )
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test contingent age hypotheses on structured detail alignments.")
    parser.add_argument("--input-dir", type=Path, default=Path("results/analysis/natural_language/detail_alignment"))
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stories = pd.read_csv(args.input_dir / "story_alignment_metrics.csv", low_memory=False)
    details = pd.read_csv(args.input_dir / "original_detail_retention.csv", low_memory=False)
    regression_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        if metric not in stories:
            continue
        regression = clustered_fixed_effect(stories, metric)
        if regression:
            for row in regression["rows"]:
                row["r_squared_within"] = regression["r_squared_within"]
                regression_rows.append(row)
        contrast = matched_contrast(stories, metric, args.iterations, args.seed)
        if contrast:
            contrast_rows.append(contrast)

    regressions = pd.DataFrame(regression_rows, columns=REGRESSION_COLUMNS)
    contrasts = pd.DataFrame(contrast_rows, columns=CONTRAST_COLUMNS)
    centrality = centrality_selection(details)
    retention_sets = retention_set_analysis(details)
    if centrality.empty:
        centrality = pd.DataFrame(columns=CENTRALITY_COLUMNS)
    for frame in [regressions, contrasts, centrality]:
        if len(frame) and "p_value" in frame:
            frame["q_value_bh"] = benjamini_hochberg(frame["p_value"].to_numpy(float))
    write_csv(args.input_dir / "within_story_age_models.csv", regressions)
    write_csv(args.input_dir / "matched_age_contrasts.csv", contrasts)
    write_csv(args.input_dir / "centrality_selection_model.csv", centrality)
    write_csv(args.input_dir / "retention_set_comparison.csv", retention_sets)
    write_run_provenance(args.input_dir, args, extra={"story_metric_rows": len(stories), "detail_rows": len(details)})
    print(f"Wrote {len(regressions)} fixed-effect coefficients")
    print(f"Wrote {len(contrasts)} matched-age contrasts")
    print(f"Wrote {len(centrality)} conditional-logit coefficients")
    print(f"Wrote {len(retention_sets)} retention baseline/set-comparison rows")


if __name__ == "__main__":
    main()
