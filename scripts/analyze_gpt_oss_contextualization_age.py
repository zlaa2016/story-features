"""Actual-summary analysis of contextual versus isolated GPT-OSS detail vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AGE_CENTER = 8.0


def normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    parsed = series.astype(str).str.lower().map({"true": True, "false": False})
    if parsed.isna().any():
        raise ValueError("Could not parse retained_cosine as Boolean.")
    return parsed.astype(bool)


def canonical_summary_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def input_csvs(directory: Path, name: str) -> list[Path]:
    merged = directory / name
    if merged.exists():
        return [merged]
    paths = sorted(directory.glob(f"shard_*/{name}"))
    if not paths:
        raise FileNotFoundError(f"No {name} under {directory}")
    return paths


def load_actual_retention(mpnet_dir: Path) -> pd.DataFrame:
    table = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in input_csvs(mpnet_dir, "original_detail_retention.csv")],
        ignore_index=True,
    )
    required = {
        "story_key", "condition", "summary_id", "source_age", "orig_idx",
        "orig_text", "retained_cosine",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"MPNet table lacks columns: {missing}")
    table = table.loc[table["condition"].astype(str).eq("actual")].copy()
    table["summary_id_key"] = canonical_summary_id(table["summary_id"])
    table["source_age"] = pd.to_numeric(table["source_age"], errors="coerce")
    table["original_detail_index"] = pd.to_numeric(table["orig_idx"], errors="raise").astype(int)
    table["retained"] = parse_bool(table["retained_cosine"])
    table["retention_status"] = pd.Categorical(
        np.where(table["retained"], "retained", "omitted"),
        categories=["retained", "omitted"], ordered=True,
    )
    table["summary_key"] = table["story_key"].astype(str) + "::" + table["summary_id_key"]
    return table


def _load_original_vectors(
    directory: Path,
    metadata_pattern: str,
    feature_prefix: str,
    primary_array_name: str,
    control_array_name: str,
    vector_index_name: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    metadata_blocks: list[pd.DataFrame] = []
    vector_blocks: list[np.ndarray] = []
    control_vector_blocks: list[np.ndarray] = []
    audits: list[dict] = []
    metadata_paths = sorted(directory.glob(metadata_pattern))
    if not metadata_paths:
        raise FileNotFoundError(f"No {metadata_pattern} under {directory}")

    for metadata_path in metadata_paths:
        suffix = metadata_path.stem.split("_metadata_", 1)[1]
        feature_path = directory / f"{feature_prefix}{suffix}.npz"
        if not feature_path.exists():
            audits.append({"shard": suffix, "status": "missing feature file", "original_details": 0})
            continue
        metadata = pd.read_csv(metadata_path, low_memory=False)
        positions = np.flatnonzero(metadata["record_type"].astype(str).eq("original").to_numpy())
        if not len(positions):
            audits.append({"shard": suffix, "status": "no originals", "original_details": 0})
            continue
        with np.load(feature_path, allow_pickle=False) as payload:
            required_arrays = {primary_array_name, control_array_name}
            missing_arrays = sorted(required_arrays - set(payload.files))
            if missing_arrays:
                raise KeyError(f"{feature_path} lacks {missing_arrays}; arrays={payload.files}")
            layers = payload["layers"]
            layer_position = int(np.argmax(layers))
            vectors = normalized_rows(payload[primary_array_name][positions, layer_position])
            control_vectors = normalized_rows(payload[control_array_name][positions, layer_position])
        selected = metadata.iloc[positions].copy().reset_index(drop=True)
        selected[vector_index_name] = np.arange(len(vectors)) + sum(map(len, vector_blocks))
        selected["selected_layer"] = int(layers[layer_position])
        metadata_blocks.append(selected)
        vector_blocks.append(vectors)
        control_vector_blocks.append(control_vectors)
        audits.append({"shard": suffix, "status": "complete", "original_details": len(selected)})

    if not metadata_blocks:
        raise FileNotFoundError(f"No usable original-detail vectors under {directory}")
    return (
        pd.concat(metadata_blocks, ignore_index=True),
        np.vstack(vector_blocks),
        np.vstack(control_vector_blocks),
        pd.DataFrame(audits),
    )


def contextual_centrality(metadata: pd.DataFrame, vectors: np.ndarray) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for record_id, frame in metadata.groupby("record_id", sort=False):
        indices = frame["context_vector_index"].to_numpy(int)
        values = vectors[indices]
        if len(values) < 2:
            centrality = np.full(len(values), np.nan)
        else:
            similarity = values @ values.T
            centrality = (similarity.sum(axis=1) - np.diag(similarity)) / (len(values) - 1)
        block = frame[["record_id", "story_key", "detail_index", "detail_text"]].copy()
        block["original_centrality"] = centrality
        median = float(np.nanmedian(centrality)) if np.isfinite(centrality).any() else np.nan
        block["centrality_group"] = np.where(centrality >= median, "central", "peripheral")
        rows.append(block)
    result = pd.concat(rows, ignore_index=True)
    result["centrality_group"] = pd.Categorical(
        result["centrality_group"], categories=["central", "peripheral"], ordered=True
    )
    return result


def load_analysis_data(
    context_dir: Path, isolated_dir: Path, mpnet_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    retention = load_actual_retention(mpnet_dir)
    context_meta, context_vectors, context_last_vectors, context_audit = _load_original_vectors(
        context_dir,
        "contextual_detail_metadata_*.csv",
        "contextual_detail_features_",
        "mean_control_vectors",
        "last_token_vectors",
        "context_vector_index",
    )
    isolated_meta, isolated_vectors, isolated_eos_vectors, isolated_audit = _load_original_vectors(
        isolated_dir,
        "isolated_detail_metadata_*.csv",
        "isolated_detail_features_",
        "mean_vectors",
        "eos_vectors",
        "isolated_vector_index",
    )

    original_records = context_meta.loc[
        context_meta["record_type"].astype(str).eq("original"), ["story_key", "record_id"]
    ].drop_duplicates()
    if original_records["story_key"].duplicated().any():
        raise ValueError("More than one contextual original record exists for a story.")

    centrality = contextual_centrality(context_meta, context_vectors)
    context_key = context_meta[["record_id", "detail_index", "context_vector_index"]].copy()
    isolated_key = isolated_meta[["record_id", "detail_index", "isolated_vector_index"]].copy()
    details = retention.merge(
        original_records.rename(columns={"record_id": "original_record_id"}),
        on="story_key", how="inner", validate="many_to_one",
    )
    details = details.merge(
        context_key.rename(columns={"record_id": "original_record_id", "detail_index": "original_detail_index"}),
        on=["original_record_id", "original_detail_index"], how="inner", validate="many_to_one",
    )
    details = details.merge(
        isolated_key.rename(columns={"record_id": "original_record_id", "detail_index": "original_detail_index"}),
        on=["original_record_id", "original_detail_index"], how="left", validate="many_to_one",
    )
    details = details.merge(
        centrality.rename(columns={"record_id": "original_record_id", "detail_index": "original_detail_index"})[
            ["original_record_id", "original_detail_index", "original_centrality", "centrality_group"]
        ],
        on=["original_record_id", "original_detail_index"], how="left", validate="many_to_one",
    )
    details["context_isolated_cosine"] = np.nan
    has_isolated = details["isolated_vector_index"].notna().to_numpy()
    ci = details.loc[has_isolated, "context_vector_index"].to_numpy(int)
    ii = details.loc[has_isolated, "isolated_vector_index"].to_numpy(int)
    details.loc[has_isolated, "context_isolated_cosine"] = np.einsum(
        "ij,ij->i", context_vectors[ci], isolated_vectors[ii]
    )
    details.loc[has_isolated, "context_mean_last_cosine"] = np.einsum(
        "ij,ij->i", context_vectors[ci], context_last_vectors[ci]
    )
    details.loc[has_isolated, "isolated_mean_eos_cosine"] = np.einsum(
        "ij,ij->i", isolated_vectors[ii], isolated_eos_vectors[ii]
    )
    details["contextualization_shift"] = 1.0 - details["context_isolated_cosine"]
    details["source_age_centered"] = details["source_age"] - AGE_CENTER

    audit = pd.DataFrame([
        {"measure": "contextual metadata shards", "value": len(context_audit)},
        {"measure": "complete contextual shards", "value": int(context_audit.status.eq("complete").sum())},
        {"measure": "contextual selected layers", "value": ", ".join(map(str, sorted(context_meta.selected_layer.unique())))},
        {"measure": "isolated metadata shards", "value": len(isolated_audit)},
        {"measure": "complete isolated shards", "value": int(isolated_audit.status.eq("complete").sum())},
        {"measure": "isolated selected layers", "value": ", ".join(map(str, sorted(isolated_meta.selected_layer.unique())))},
        {"measure": "actual MPNet detail rows", "value": len(retention)},
        {"measure": "joined contextual detail rows", "value": len(details)},
        {"measure": "rows with matched isolated vector", "value": int(has_isolated.sum())},
        {"measure": "isolated coverage of joined rows", "value": float(has_isolated.mean())},
        {"measure": "stories with matched vectors", "value": details.loc[has_isolated, "story_key"].nunique()},
        {"measure": "actual summaries with matched vectors", "value": details.loc[has_isolated, "summary_key"].nunique()},
    ])
    if not has_isolated.any():
        raise ValueError("No contextual original details matched isolated controls.")
    return details.loc[has_isolated].copy(), audit


def fixed_effect_table(result) -> pd.DataFrame:
    names = list(result.fe_params.index) if hasattr(result, "fe_params") else list(result.params.index)
    estimates = result.fe_params if hasattr(result, "fe_params") else result.params
    confidence = result.conf_int().loc[names]
    return pd.DataFrame({
        "term": names,
        "estimate": estimates.loc[names].to_numpy(float),
        "std_error": result.bse.loc[names].to_numpy(float),
        "statistic": result.tvalues.loc[names].to_numpy(float),
        "p_value": result.pvalues.loc[names].to_numpy(float),
        "ci95_low": confidence[0].to_numpy(float),
        "ci95_high": confidence[1].to_numpy(float),
        "significant_05": result.pvalues.loc[names].to_numpy(float) < 0.05,
    })


def _linear_test(result, contrast: np.ndarray, name: str) -> dict:
    from scipy.stats import norm

    names = list(result.fe_params.index)
    values = result.fe_params.to_numpy(float)
    covariance = result.cov_params().loc[names, names].to_numpy(float)
    estimate = float(contrast @ values)
    se = float(np.sqrt(max(contrast @ covariance @ contrast, 0.0)))
    z = estimate / se if se > 0 else np.nan
    return {
        "test": name, "estimate_per_year": estimate, "std_error": se, "z": z,
        "p_value": 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan,
        "ci95_low": estimate - 1.96 * se, "ci95_high": estimate + 1.96 * se,
    }


def slope_tests(result, formula: str, group_variable: str, levels: Iterable[str]) -> pd.DataFrame:
    from patsy import dmatrix

    names = list(result.fe_params.index)
    rhs = formula.split("~", 1)[1].strip()

    def row(level: str, age: float) -> np.ndarray:
        frame = pd.DataFrame({
            "source_age_centered": [age - AGE_CENTER],
            group_variable: pd.Categorical([level], categories=list(levels), ordered=True),
        })
        matrix = dmatrix(rhs, frame, return_type="dataframe")
        return matrix.reindex(columns=names, fill_value=0.0).iloc[0].to_numpy(float)

    levels = list(levels)
    slopes = {level: row(level, AGE_CENTER + 0.5) - row(level, AGE_CENTER - 0.5) for level in levels}
    tests = [_linear_test(result, slopes[level], f"{level}: source-age slope") for level in levels]
    tests.append(_linear_test(
        result, slopes[levels[0]] - slopes[levels[1]],
        f"{levels[0]} minus {levels[1]}: slope difference",
    ))
    output = pd.DataFrame(tests)
    order = np.argsort(output["p_value"].to_numpy())
    adjusted_sorted = np.maximum.accumulate(
        (len(order) - np.arange(len(order))) * output["p_value"].to_numpy()[order]
    )
    output["p_holm"] = np.nan
    output.loc[order, "p_holm"] = np.minimum(adjusted_sorted, 1.0)
    output["significant_holm_05"] = output["p_holm"] < 0.05
    return output


def aggregate_two_rows(details: pd.DataFrame, group_column: str, keys: list[str]) -> pd.DataFrame:
    output = details.groupby(keys + [group_column], observed=True, as_index=False).agg(
        mean_contextualization_shift=("contextualization_shift", "mean"),
        details=("original_detail_index", "size"),
    )
    counts = output.groupby(keys, observed=True)[group_column].nunique()
    valid = counts[counts.eq(2)].index
    if len(keys) == 1:
        output = output.loc[output[keys[0]].isin(valid)].copy()
    else:
        valid_frame = pd.MultiIndex.from_frame(output[keys]).isin(valid)
        output = output.loc[valid_frame].copy()
    return output


def plot_group_trajectories(result, formula: str, group_variable: str, levels: list[str], data: pd.DataFrame, title: str):
    from patsy import dmatrix

    names = list(result.fe_params.index)
    values = result.fe_params.to_numpy(float)
    covariance = result.cov_params().loc[names, names].to_numpy(float)
    rhs = formula.split("~", 1)[1].strip()
    ages = np.linspace(data["source_age"].min(), data["source_age"].max(), 100)
    colors = ["#4C78A8", "#E15759"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for level, color in zip(levels, colors):
        frame = pd.DataFrame({
            "source_age_centered": ages - AGE_CENTER,
            group_variable: pd.Categorical([level] * len(ages), categories=levels, ordered=True),
        })
        design = dmatrix(rhs, frame, return_type="dataframe").reindex(columns=names, fill_value=0.0).to_numpy()
        fitted = design @ values
        se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", design, covariance, design), 0.0))
        ax.plot(ages, fitted, color=color, label=level)
        ax.fill_between(ages, fitted - 1.96 * se, fitted + 1.96 * se, color=color, alpha=0.16)
    ax.set(xlabel="Source age", ylabel="Mean contextualization shift (1 − cosine)", title=title)
    ax.legend(title=group_variable.replace("_", " "))
    plt.tight_layout()
    plt.show()
