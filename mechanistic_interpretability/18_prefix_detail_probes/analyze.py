#!/usr/bin/env python3
"""Analyze controlled prefix/detail probes and actual-summary retention labels."""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AGE_CENTER = 8.0


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("...d,...d->...", normalized(left), normalized(right))


def load_probe_metrics(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks: list[pd.DataFrame] = []
    audits: list[dict] = []
    for metadata_path_text in sorted(glob.glob(str(input_dir / "probe_metadata_*.csv"))):
        metadata_path = Path(metadata_path_text)
        suffix = metadata_path.stem.removeprefix("probe_metadata_")
        feature_path = input_dir / f"probe_features_{suffix}.npz"
        if not feature_path.exists():
            audits.append({"shard": suffix, "status": "missing_features", "rows": 0})
            continue
        metadata = pd.read_csv(metadata_path, low_memory=False)
        with np.load(feature_path, allow_pickle=False) as payload:
            arrays = {
                name: payload[name].astype(np.float32)
                for name in (
                    "first_detail_vectors", "last_detail_vectors", "mean_detail_vectors",
                    "prefix_readout_vectors", "post_detail_readout_vectors",
                )
            }
            layers = payload["layers"].astype(int)
        if any(len(values) != len(metadata) for values in arrays.values()):
            raise ValueError(f"Row mismatch in shard {suffix}")
        isolated_lookup = {
            (str(row.story_key), int(row.target_detail_index)): index
            for index, row in enumerate(metadata.itertuples(index=False))
            if str(row.prefix_condition) == "isolated"
        }
        missing = [
            index for index, row in enumerate(metadata.itertuples(index=False))
            if (str(row.story_key), int(row.target_detail_index)) not in isolated_lookup
        ]
        if missing:
            raise ValueError(f"Shard {suffix} has {len(missing)} rows without isolated baselines")
        baseline_indices = np.asarray([
            isolated_lookup[(str(row.story_key), int(row.target_detail_index))]
            for row in metadata.itertuples(index=False)
        ])
        final_mean = normalized(arrays["mean_detail_vectors"][:, -1])
        centrality_for_isolated: dict[int, float] = {}
        for story_key, story_frame in metadata.groupby("story_key", sort=False):
            target_to_isolated = {
                int(row.target_detail_index): index
                for index, row in zip(story_frame.index, story_frame.itertuples(index=False))
                if str(row.prefix_condition) == "isolated"
            }
            ordered = sorted(target_to_isolated)
            indices = np.asarray([target_to_isolated[target] for target in ordered], dtype=int)
            if len(indices) < 2:
                values = np.full(len(indices), np.nan)
            else:
                similarity = final_mean[indices] @ final_mean[indices].T
                values = (similarity.sum(axis=1) - np.diag(similarity)) / (len(indices) - 1)
            centrality_for_isolated.update({int(index): float(value) for index, value in zip(indices, values)})
        centrality = np.asarray([centrality_for_isolated[int(index)] for index in baseline_indices])
        centrality_group = np.full(len(metadata), "peripheral", dtype=object)
        for _story_key, story_indices in metadata.groupby("story_key", sort=False).groups.items():
            positions = np.asarray(list(story_indices), dtype=int)
            median = np.nanmedian(centrality[positions])
            centrality_group[positions] = np.where(centrality[positions] >= median, "central", "peripheral")
        for layer_position, layer in enumerate(layers):
            block = metadata.copy()
            block["layer"] = int(layer)
            block["isolated_semantic_centrality"] = centrality
            block["centrality_group"] = centrality_group
            for label, array_name in (
                ("first", "first_detail_vectors"),
                ("last", "last_detail_vectors"),
                ("mean", "mean_detail_vectors"),
            ):
                values = arrays[array_name][:, layer_position]
                baseline = values[baseline_indices]
                block[f"{label}_context_isolated_cosine"] = cosine_rows(values, baseline)
                block[f"{label}_contextualization_shift"] = 1.0 - block[f"{label}_context_isolated_cosine"]
            prefix = arrays["prefix_readout_vectors"][:, layer_position]
            post = arrays["post_detail_readout_vectors"][:, layer_position]
            block["readout_hidden_cosine"] = cosine_rows(prefix, post)
            block["readout_hidden_shift"] = 1.0 - block["readout_hidden_cosine"]
            block["readout_relative_l2"] = (
                np.linalg.norm(post - prefix, axis=1)
                / np.maximum(np.linalg.norm(prefix, axis=1), 1e-12)
            )
            blocks.append(block)
        audits.append({"shard": suffix, "status": "complete", "rows": len(metadata), "layers": len(layers)})
    if not blocks:
        raise FileNotFoundError(f"No complete probe shards under {input_dir}")
    metrics = pd.concat(blocks, ignore_index=True)
    metrics["target_position_norm"] = metrics["target_detail_index"] / np.maximum(
        metrics["n_details_in_story"] - 1, 1
    )
    return metrics, pd.DataFrame(audits)


def input_csvs(directory: Path, name: str) -> list[Path]:
    direct = directory / name
    if direct.exists():
        return [direct]
    paths = sorted(directory.glob(f"shard_*/{name}"))
    if not paths:
        raise FileNotFoundError(f"No {name} under {directory}")
    return paths


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    output = series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if output.isna().any():
        raise ValueError("Could not parse retained_cosine")
    return output.astype(bool)


def load_actual_retention(directory: Path) -> pd.DataFrame:
    frame = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in input_csvs(directory, "original_detail_retention.csv")],
        ignore_index=True,
    )
    required = {"story_key", "condition", "summary_id", "source_age", "orig_idx", "retained_cosine"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Retention table lacks {sorted(missing)}")
    frame = frame.loc[frame["condition"].astype(str).eq("actual")].copy()
    frame["target_detail_index"] = pd.to_numeric(frame["orig_idx"], errors="raise").astype(int)
    frame["source_age"] = pd.to_numeric(frame["source_age"], errors="coerce")
    frame["retention_status"] = np.where(parse_bool(frame["retained_cosine"]), "retained", "omitted")
    frame["summary_key"] = (
        frame["story_key"].astype(str) + "::" +
        pd.to_numeric(frame["summary_id"], errors="coerce").astype("Int64").astype(str)
    )
    return frame


def holm(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    adjusted = np.full(len(p), np.nan)
    running = 0.0
    finite_count = int(np.isfinite(p).sum())
    rank = 0
    for index in order:
        if not np.isfinite(p[index]):
            continue
        value = min(1.0, (finite_count - rank) * p[index])
        running = max(running, value)
        adjusted[index] = running
        rank += 1
    return adjusted


def fixed_effect_table(result: object, metric: str, model_name: str) -> pd.DataFrame:
    names = list(result.fe_params.index) if hasattr(result, "fe_params") else list(result.params.index)
    estimates = result.fe_params if hasattr(result, "fe_params") else result.params
    confidence = result.conf_int().loc[names]
    return pd.DataFrame({
        "metric": metric,
        "model": model_name,
        "term": names,
        "estimate": estimates.loc[names].to_numpy(float),
        "std_error": result.bse.loc[names].to_numpy(float),
        "statistic": result.tvalues.loc[names].to_numpy(float),
        "p_value": result.pvalues.loc[names].to_numpy(float),
        "ci95_low": confidence[0].to_numpy(float),
        "ci95_high": confidence[1].to_numpy(float),
    })


def linear_test(result: object, contrast: np.ndarray, metric: str, test: str) -> dict:
    from scipy.stats import norm

    names = list(result.fe_params.index)
    values = result.fe_params.to_numpy(float)
    covariance = result.cov_params().loc[names, names].to_numpy(float)
    estimate = float(contrast @ values)
    se = float(np.sqrt(max(contrast @ covariance @ contrast, 0.0)))
    z = estimate / se if se > 0 else np.nan
    return {
        "metric": metric, "test": test, "estimate": estimate, "std_error": se, "z": z,
        "p_value": 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan,
        "ci95_low": estimate - 1.96 * se, "ci95_high": estimate + 1.96 * se,
    }


def retention_models(joined: pd.DataFrame, metrics_to_test: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import statsmodels.formula.api as smf

    aggregate_blocks: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    tests: list[dict] = []
    final_layer = int(joined["layer"].max())
    native = joined.loc[
        joined["layer"].eq(final_layer)
        & joined["prefix_condition"].eq("true")
        & joined["native_prefix"].astype(bool)
    ].copy()
    native["source_age_centered"] = native["source_age"] - AGE_CENTER
    native["retention_status"] = pd.Categorical(
        native["retention_status"], categories=["retained", "omitted"], ordered=True
    )
    keys = ["story_key", "summary_key", "source_age", "source_age_centered", "retention_status"]
    for metric in metrics_to_test:
        if metric not in native or native[metric].notna().sum() < 20:
            continue
        aggregate = native.groupby(keys, observed=True, as_index=False).agg(
            value=(metric, "mean"),
            details=("target_detail_index", "size"),
            mean_log_detail_tokens=("detail_tokens", lambda values: float(np.log1p(values).mean())),
            mean_log_prefix_tokens=("prefix_tokens", lambda values: float(np.log1p(values).mean())),
        )
        complete = aggregate.groupby("summary_key")["retention_status"].nunique()
        aggregate = aggregate.loc[aggregate["summary_key"].isin(complete[complete.eq(2)].index)].copy()
        aggregate["metric"] = metric
        aggregate_blocks.append(aggregate)
        if aggregate["story_key"].nunique() < 10:
            continue
        formula = (
            'value ~ source_age_centered * C(retention_status, Treatment(reference="retained")) '
            '+ mean_log_detail_tokens + mean_log_prefix_tokens'
        )
        model = smf.mixedlm(formula, aggregate, groups=aggregate["story_key"], re_formula="1")
        try:
            fit = model.fit(reml=False, method="lbfgs")
        except Exception as first_error:
            try:
                fit = model.fit(reml=False, method="powell")
            except Exception as second_error:
                tests.append({
                    "metric": metric,
                    "test": "MODEL_FAILED",
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "z": np.nan,
                    "p_value": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "error": f"lbfgs={first_error}; powell={second_error}",
                })
                continue
        coefficients.append(fixed_effect_table(fit, metric, "actual-summary two-row mixed model"))
        names = list(fit.fe_params.index)
        age_name = "source_age_centered"
        retention_name = 'C(retention_status, Treatment(reference="retained"))[T.omitted]'
        interaction_name = f"{age_name}:{retention_name}"
        basis = {name: index for index, name in enumerate(names)}
        vector = lambda: np.zeros(len(names), dtype=float)
        retained_slope = vector(); retained_slope[basis[age_name]] = 1
        omitted_slope = retained_slope.copy(); omitted_slope[basis[interaction_name]] += 1
        gap_at_eight = vector(); gap_at_eight[basis[retention_name]] = -1  # retained minus omitted
        slope_gap = vector(); slope_gap[basis[interaction_name]] = -1  # retained minus omitted
        tests.extend([
            linear_test(fit, retained_slope, metric, "retained: source-age slope"),
            linear_test(fit, omitted_slope, metric, "omitted: source-age slope"),
            linear_test(fit, slope_gap, metric, "retained minus omitted: slope difference"),
            linear_test(fit, gap_at_eight, metric, "retained minus omitted at source age 8"),
        ])
        pivot = aggregate.pivot(index="summary_key", columns="retention_status", values="value").dropna()
        difference = pivot["retained"] - pivot["omitted"]
        dz = float(difference.mean() / difference.std(ddof=1)) if len(difference) > 1 and difference.std(ddof=1) else np.nan
        tests.append({
            "metric": metric, "test": "paired Cohen dz: retained minus omitted", "estimate": dz,
            "std_error": np.nan, "z": np.nan, "p_value": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
        })
    aggregate_frame = pd.concat(aggregate_blocks, ignore_index=True) if aggregate_blocks else pd.DataFrame()
    coefficient_frame = pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame()
    test_frame = pd.DataFrame(tests)
    if not test_frame.empty:
        mask = test_frame["p_value"].notna()
        test_frame.loc[mask, "p_holm"] = holm(test_frame.loc[mask, "p_value"])
        test_frame["significant_holm_05"] = test_frame["p_holm"] < 0.05
    return aggregate_frame, coefficient_frame, test_frame


def pooling_equivalence(metrics: pd.DataFrame, margin: float) -> pd.DataFrame:
    """Paired TOST of mean-pooled versus last-token contextualization shift."""
    from scipy.stats import t

    final = metrics.loc[metrics["layer"].eq(metrics["layer"].max())].copy()
    rows: list[dict] = []
    for condition, frame in final.groupby("prefix_condition", observed=True):
        difference = (
            frame["mean_contextualization_shift"] - frame["last_contextualization_shift"]
        ).dropna().to_numpy(float)
        n = len(difference)
        mean = float(difference.mean()) if n else np.nan
        sd = float(difference.std(ddof=1)) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        lower_t = (mean + margin) / se if se and np.isfinite(se) else np.nan
        upper_t = (mean - margin) / se if se and np.isfinite(se) else np.nan
        p_lower = 1 - t.cdf(lower_t, n - 1) if np.isfinite(lower_t) else np.nan
        p_upper = t.cdf(upper_t, n - 1) if np.isfinite(upper_t) else np.nan
        rows.append({
            "prefix_condition": condition,
            "n": n,
            "mean_difference_mean_minus_last": mean,
            "sd_difference": sd,
            "paired_cohen_dz": mean / sd if sd else np.nan,
            "pearson_r": frame[["mean_contextualization_shift", "last_contextualization_shift"]].corr().iloc[0, 1],
            "equivalence_margin": margin,
            "p_tost": max(p_lower, p_upper) if np.isfinite(p_lower) and np.isfinite(p_upper) else np.nan,
            "equivalent_at_05": bool(max(p_lower, p_upper) < 0.05) if np.isfinite(p_lower) and np.isfinite(p_upper) else False,
        })
    return pd.DataFrame(rows)


def centrality_models(joined: pd.DataFrame, outcomes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Four rows per summary: retained/omitted crossed with central/peripheral."""
    import statsmodels.formula.api as smf

    final_layer = int(joined["layer"].max())
    native = joined.loc[
        joined["layer"].eq(final_layer)
        & joined["prefix_condition"].eq("true")
        & joined["native_prefix"].astype(bool)
    ].copy()
    native["source_age_centered"] = native["source_age"] - AGE_CENTER
    native["retention_status"] = pd.Categorical(
        native["retention_status"], categories=["retained", "omitted"], ordered=True
    )
    native["centrality_group"] = pd.Categorical(
        native["centrality_group"], categories=["central", "peripheral"], ordered=True
    )
    keys = [
        "story_key", "summary_key", "source_age", "source_age_centered",
        "retention_status", "centrality_group",
    ]
    aggregates: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    for outcome in outcomes:
        if outcome not in native:
            continue
        data = native.groupby(keys, observed=True, as_index=False).agg(
            value=(outcome, "mean"),
            details=("target_detail_index", "size"),
            mean_log_detail_tokens=("detail_tokens", lambda values: float(np.log1p(values).mean())),
            mean_log_prefix_tokens=("prefix_tokens", lambda values: float(np.log1p(values).mean())),
        )
        data["metric"] = outcome
        aggregates.append(data)
        if len(data) < 40 or data["story_key"].nunique() < 10:
            continue
        formula = (
            'value ~ source_age_centered * '
            'C(retention_status, Treatment(reference="retained")) * '
            'C(centrality_group, Treatment(reference="central")) '
            '+ mean_log_detail_tokens + mean_log_prefix_tokens'
        )
        model = smf.mixedlm(formula, data, groups=data["story_key"], re_formula="1")
        try:
            fit = model.fit(reml=False, method="lbfgs")
        except Exception:
            try:
                fit = model.fit(reml=False, method="powell")
            except Exception:
                continue
        coefficients.append(fixed_effect_table(fit, outcome, "actual-summary retention x centrality mixed model"))
    return (
        pd.concat(aggregates, ignore_index=True) if aggregates else pd.DataFrame(),
        pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame(),
    )


def prefix_trajectory_model(metrics: pd.DataFrame, outcome: str) -> pd.DataFrame:
    final_layer = int(metrics["layer"].max())
    data = metrics.loc[
        metrics["layer"].eq(final_layer)
        & metrics["prefix_condition"].isin(["true", "shuffled", "foreign"])
    ].dropna(subset=[outcome, "prefix_fraction"]).copy()
    if "prefix_mode" in data:
        data = data.loc[data["prefix_mode"].astype(str).eq("all")].copy()
    if data["prefix_index"].nunique() < 3 or data["story_key"].nunique() < 2:
        return pd.DataFrame()
    import statsmodels.formula.api as smf
    formula = (
        f'{outcome} ~ prefix_fraction * C(prefix_condition, Treatment(reference="true")) '
        "+ target_position_norm + np.log1p(detail_tokens) + np.log1p(prefix_tokens)"
    )
    fit = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["story_key"]})
    return fixed_effect_table(fit, outcome, "all-prefix OLS; story-clustered SE")


def plot_distributions(metrics: pd.DataFrame, output: Path) -> None:
    final = metrics.loc[metrics["layer"].eq(metrics["layer"].max())].copy()
    outcomes = ["mean_contextualization_shift", "detail_nll_delta_from_isolated", "readout_output_js"]
    conditions = [item for item in ["true", "shuffled", "foreign"] if item in set(final["prefix_condition"])]
    colors = {"true": "#4C78A8", "shuffled": "#F28E2B", "foreign": "#9C755F"}
    fig, axes = plt.subplots(1, len(outcomes), figsize=(16, 4.5))
    for ax, outcome in zip(axes, outcomes):
        for condition in conditions:
            values = final.loc[final["prefix_condition"].eq(condition), outcome].dropna().to_numpy()
            if len(values):
                ax.hist(values, bins=40, density=True, histtype="step", linewidth=1.8,
                        label=condition, color=colors[condition])
        ax.set_title(outcome.replace("_", " "))
        ax.set_ylabel("Density")
    axes[-1].legend(title="prefix")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_retention_age(aggregate: pd.DataFrame, output: Path) -> None:
    if aggregate.empty:
        return
    preferred = ["mean_contextualization_shift", "detail_nll_delta_from_isolated", "readout_output_js"]
    available = [metric for metric in preferred if metric in set(aggregate["metric"])]
    fig, axes = plt.subplots(1, len(available), figsize=(5.2 * len(available), 4.5), squeeze=False)
    colors = {"retained": "#4C78A8", "omitted": "#E15759"}
    for ax, metric in zip(axes[0], available):
        data = aggregate.loc[aggregate["metric"].eq(metric)].copy()
        for status in ["retained", "omitted"]:
            group = data.loc[data["retention_status"].astype(str).eq(status)]
            means = group.groupby("source_age", as_index=False)["value"].mean()
            ax.plot(means["source_age"], means["value"], marker="o", label=status, color=colors[status])
        ax.set(xlabel="Source age", ylabel=metric.replace("_", " "), title="Actual summaries")
    axes[0, -1].legend(title="detail status")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_prefix_trajectory(metrics: pd.DataFrame, output: Path) -> None:
    final = metrics.loc[
        metrics["layer"].eq(metrics["layer"].max())
        & metrics["prefix_condition"].isin(["true", "shuffled", "foreign"])
    ].copy()
    if "prefix_mode" in final:
        final = final.loc[final["prefix_mode"].astype(str).eq("all")].copy()
    if final["prefix_index"].nunique() < 3:
        return
    final["prefix_bin"] = pd.cut(final["prefix_fraction"], bins=np.linspace(0, 1, 11), include_lowest=True)
    grouped = final.groupby(["prefix_condition", "prefix_bin"], observed=True, as_index=False).agg(
        prefix_fraction=("prefix_fraction", "mean"),
        mean_shift=("mean_contextualization_shift", "mean"),
        nll_delta=("detail_nll_delta_from_isolated", "mean"),
        readout_js=("readout_output_js", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {"true": "#4C78A8", "shuffled": "#F28E2B", "foreign": "#9C755F"}
    for condition, data in grouped.groupby("prefix_condition", observed=True):
        for ax, outcome in zip(axes, ["mean_shift", "nll_delta", "readout_js"]):
            ax.plot(data["prefix_fraction"], data[outcome], marker="o", label=condition,
                    color=colors.get(str(condition), None))
    for ax, title in zip(axes, ["Mean hidden shift", "Detail NLL change", "Readout output JS"]):
        ax.set(xlabel="Prefix boundary / story details", ylabel=title)
    axes[-1].legend(title="prefix")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--retention-dir", type=Path, default=Path("mechanistic_interpretability/outputs/13_structured_detail_alignment"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pooling-equivalence-margin", type=float, default=0.02,
        help="TOST margin on 1-cosine units for mean-pooling versus last-token shift.",
    )
    parser.add_argument("--skip-statistics", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics, audit = load_probe_metrics(args.input_dir)
    retention = load_actual_retention(args.retention_dir)
    joined = retention.merge(
        metrics,
        on=["story_key", "target_detail_index"],
        how="inner",
        suffixes=("_retention", ""),
        validate="many_to_many",
    )
    if "source_age_retention" in joined:
        joined["source_age"] = joined["source_age_retention"].combine_first(joined["source_age"])
    metrics_to_test = [
        "first_contextualization_shift", "last_contextualization_shift",
        "mean_contextualization_shift", "readout_hidden_shift", "readout_relative_l2",
        "detail_nll_delta_from_isolated", "mean_token_js_from_isolated",
        "readout_output_js", "readout_entropy_change",
    ]
    pooling = pooling_equivalence(metrics, args.pooling_equivalence_margin)
    if args.skip_statistics:
        aggregate = pd.DataFrame()
        coefficients = pd.DataFrame()
        tests = pd.DataFrame()
        centrality_aggregate = pd.DataFrame()
        centrality_coefficients = pd.DataFrame()
        trajectory = pd.DataFrame()
    else:
        aggregate, coefficients, tests = retention_models(joined, metrics_to_test)
        centrality_aggregate, centrality_coefficients = centrality_models(
            joined,
            ["mean_contextualization_shift", "detail_nll_delta_from_isolated", "readout_output_js"],
        )
        trajectory = prefix_trajectory_model(metrics, "mean_contextualization_shift")

    metrics.to_csv(args.output_dir / "probe_metrics.csv.gz", index=False)
    joined.to_csv(args.output_dir / "actual_retention_probe_metrics.csv.gz", index=False)
    audit.to_csv(args.output_dir / "shard_audit.csv", index=False)
    aggregate.to_csv(args.output_dir / "actual_two_row_retention_means.csv", index=False)
    coefficients.to_csv(args.output_dir / "actual_retention_mixed_model_coefficients.csv", index=False)
    tests.to_csv(args.output_dir / "actual_retention_hypothesis_tests.csv", index=False)
    pooling.to_csv(args.output_dir / "mean_vs_last_pooling_equivalence.csv", index=False)
    centrality_aggregate.to_csv(args.output_dir / "actual_retention_centrality_means.csv", index=False)
    centrality_coefficients.to_csv(args.output_dir / "actual_retention_centrality_model_coefficients.csv", index=False)
    trajectory.to_csv(args.output_dir / "prefix_trajectory_coefficients.csv", index=False)
    plot_distributions(metrics, args.output_dir / "probe_metric_distributions.png")
    plot_retention_age(aggregate, args.output_dir / "actual_retention_by_source_age.png")
    plot_prefix_trajectory(metrics, args.output_dir / "all_prefix_trajectories.png")

    print(f"Probe rows x layers: {len(metrics):,}")
    print(f"Stories: {metrics['story_key'].nunique():,}")
    print(f"Actual-retention joined rows: {len(joined):,}")
    print(f"Results: {args.output_dir}")
    if not tests.empty:
        print("\nHolm-adjusted significant tests:")
        print(tests.loc[tests["significant_holm_05"].fillna(False)].to_string(index=False))


if __name__ == "__main__":
    main()
