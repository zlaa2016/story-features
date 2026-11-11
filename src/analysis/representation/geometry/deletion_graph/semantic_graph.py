"""Semantic-relatedness graph against the deletion-coupling graph of the same stories.

The deletion graph says how much deleting detail i moves detail j (causal, forward only).  The semantic graph says how alike two details
are: the cosine of their embeddings, taken from detail vectors that already exist (``isolated`` = each detail embedded on its own, or
``contextual`` = its state inside the story), so nothing is re-extracted.  It is symmetric, so it is put on the forward pairs i < j like the
deletion matrix and goes through exactly the same graph measures.

Raw cosines of LLM states are all high (one dominant direction), so by default the unit vectors are standardised first: the global mean is
removed and every dimension divided by its global standard deviation (as in the spectral-age analysis); negative similarities are set to 0
because the graph measures need non-negative strengths.  ``--standardize none`` keeps the raw cosine.

Outputs in ``--output-dir``:
* ``edge_agreement.csv``: per story, Spearman between the two graphs' pair weights (``rho_raw``, and ``rho_excess`` after dividing each
  pair by the average pair at its lag, which removes the shared fade with distance), overlap of the strongest ``keep`` share of pairs
  against what chance gives, how many of the 3 key sources / 3 key impacted details the two graphs share, and the cross-graph R²
  (``r2_deletion_gain_semantic`` = share of the deletion weights' rank variance the semantic weights explain beyond lag; ``..._lag`` = lag alone),
* ``agreement_age_tests.csv``: Spearman of each agreement statistic with the story's mid age,
* ``measures_comparison.csv`` / ``measures_overview.csv``: every graph measure of the two graphs side by side (``compare_units.compare``),
* ``semantic_story_measures.csv``, ``semantic_audit.csv`` (stories skipped and why), ``fig_agreement_by_age.png``.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.representation.geometry.deletion_graph import analyze, compare_units
from analysis.representation.geometry.deletion_graph.graph_math import excess_matrix, strengths

KEY_NODES = 3


def story_vectors(vectors: np.ndarray, mean: np.ndarray | None, scale: np.ndarray | None) -> np.ndarray:
    """Unit rows, then (optionally) globally centred and scaled."""
    unit = vectors.astype(np.float64)
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    if mean is not None and scale is not None:
        unit = (unit - mean) / scale
    return unit


def semantic_matrix(rows: np.ndarray) -> np.ndarray:
    """Forward-pair matrix of cosine similarity (negatives set to 0): entry [i, j] for i < j, zeros elsewhere."""
    norms = np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    cosine = (rows / norms) @ (rows / norms).T
    return np.triu(np.clip(cosine, 0.0, None), 1)


def _pairs(matrix: np.ndarray) -> np.ndarray:
    rows, cols = np.triu_indices(matrix.shape[0], 1)
    return matrix[rows, cols]


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y)[0])


def _top_set(values: np.ndarray, count: int) -> set[int]:
    return set(np.argsort(-values, kind="stable")[:count].tolist())


def edge_agreement(deletion: np.ndarray, semantic: np.ndarray, keep: float = 0.2) -> dict[str, float]:
    """How alike two forward-pair matrices of one story are."""
    n = deletion.shape[0]
    d_raw, s_raw = _pairs(deletion), _pairs(semantic)
    d_exc, s_exc = _pairs(excess_matrix(deletion)), _pairs(excess_matrix(semantic))
    total = d_raw.size
    count = max(1, int(round(keep * total)))
    shared = len(_top_set(d_exc, count) & _top_set(s_exc, count))
    union = 2 * count - shared
    expected_shared = count * count / total
    out_d, in_d = strengths(deletion)
    out_s, in_s = strengths(semantic)
    k = min(KEY_NODES, len(out_d), len(in_d))
    return {
        "n_details": n,
        "rho_raw": _rho(d_raw, s_raw), "rho_excess": _rho(d_exc, s_exc),
        "top_overlap": shared / union, "top_overlap_chance": expected_shared / (2 * count - expected_shared),
        "source_overlap": len(_top_set(out_d, k) & _top_set(out_s, k)) / k,
        "impacted_overlap": len(_top_set(in_d, k) & _top_set(in_s, k)) / k,
        "chance_node_overlap": k / max(len(out_d), 1),
        **cross_graph_r2(deletion, semantic),
    }


def _r2(response: np.ndarray, features: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(response)), features])
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    spread = ((response - response.mean()) ** 2).sum()
    return float(1 - ((response - design @ coefficients) ** 2).sum() / spread) if spread > 0 else float("nan")


def cross_graph_r2(deletion: np.ndarray, semantic: np.ndarray) -> dict[str, float]:
    """How much of one graph's pair weights the other explains beyond distance in the story (ranks, ordinary least squares).

    ``r2_deletion_lag``: deletion weights from lag alone; ``r2_deletion_gain_semantic``: what the semantic weights add on top of lag;
    ``r2_semantic_lag`` and ``r2_semantic_gain_deletion`` the other way round.  A small gain means the graphs carry different information.
    """
    rows, cols = np.triu_indices(deletion.shape[0], 1)
    keys = ("r2_deletion_lag", "r2_deletion_gain_semantic", "r2_semantic_lag", "r2_semantic_gain_deletion")
    if rows.size < 8:
        return {key: float("nan") for key in keys}
    lag = (cols - rows).astype(np.float64)
    lag_features = np.column_stack([lag, np.log(lag)])
    d, s = stats.rankdata(deletion[rows, cols]), stats.rankdata(semantic[rows, cols])
    d_lag, s_lag = _r2(d, lag_features), _r2(s, lag_features)
    return dict(zip(keys, (
        d_lag, _r2(d, np.column_stack([lag_features, s])) - d_lag,
        s_lag, _r2(s, np.column_stack([lag_features, d])) - s_lag,
    )))


def load_detail_vectors(folders: list[Path], view: str, layer: int, key: str | None) -> tuple[pd.DataFrame, np.ndarray]:
    run = importlib.import_module("analysis.representation.geometry.spectral_age.run")
    table = run.detail_table(run.usable(folders), view)
    return table, run.detail_vectors(table, view, layer, key)


def build_semantic(metadata: pd.DataFrame, table: pd.DataFrame, vectors: np.ndarray, standardize: bool) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Semantic matrices for the deletion-graph stories that have exactly their n details' vectors; and the audit of the others."""
    by_story = {key: group.sort_values("detail_index") for key, group in table.groupby("story_key", sort=False)}
    usable, audit = {}, []
    for _, record in metadata.iterrows():
        group = by_story.get(record["story_key"])
        if group is None:
            audit.append({"story_key": record["story_key"], "status": "no_detail_vectors"})
        elif group["detail_index"].astype(int).tolist() != list(range(int(record["n_details"]))):
            audit.append({"story_key": record["story_key"], "status": "detail_count_mismatch",
                          "vectors": len(group), "deletion_details": int(record["n_details"])})
        else:
            usable[record["key"]] = vectors[group.index.to_numpy()]
    mean = scale = None
    if standardize and usable:
        pooled = np.concatenate([story_vectors(v, None, None) for v in usable.values()])
        mean, scale = pooled.mean(axis=0), pooled.std(axis=0) + 1e-6
    return {key: semantic_matrix(story_vectors(v, mean, scale)) for key, v in usable.items()}, audit


def age_tests(agreement: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        pair = agreement[[column, "age_mid"]].dropna()
        ok = len(pair) >= 10 and pair["age_mid"].nunique() >= 3
        rho, p = stats.spearmanr(pair[column], pair["age_mid"]) if ok else (float("nan"), float("nan"))
        rows.append({"statistic": column, "n": len(pair), "median": float(agreement[column].median()), "rho_age": rho, "p": p})
    return pd.DataFrame(rows)


def plot_by_age(agreement: pd.DataFrame, membership: pd.DataFrame, columns: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_age = membership.merge(agreement, on="story_key", how="inner")
    rows_needed = -(-len(columns) // 3)
    figure, axes = plt.subplots(rows_needed, 3, figsize=(13, 3.5 * rows_needed), sharex=True, squeeze=False)
    for axis, column in zip(axes.ravel(), columns):
        part = by_age.groupby("age")[column].agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        part.columns = ["median", "q25", "q75"]
        axis.fill_between(part.index, part["q25"], part["q75"], alpha=0.25)
        axis.plot(part.index, part["median"], marker="o", markersize=3)
        axis.set_title(column, fontsize=10)
        axis.set_xlabel("age")
    for axis in axes.ravel()[len(columns):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="deletion_graph_*.npz shards (the same ones analyze.py reads)")
    parser.add_argument("--detail-dir", type=Path, nargs="+", required=True, help="folders with isolated_/contextual_detail_features_*.npz")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detail-view", choices=["isolated", "contextual"], default="isolated")
    parser.add_argument("--detail-layer", type=int, default=-1, help="layer number of the detail vectors (-1 = the last stored)")
    parser.add_argument("--detail-key", default=None)
    parser.add_argument("--standardize", choices=["global", "none"], default="global")
    parser.add_argument("--view", choices=["mean", "last"], default="mean")
    parser.add_argument("--distance", choices=["cos", "rel"], default="cos")
    parser.add_argument("--layer", type=int, default=None, help="deletion-graph layer (default: the last saved)")
    parser.add_argument("--keep", type=float, default=0.2)
    args = parser.parse_args()

    metadata, deletion, layer = analyze.load_shards(args.input_dir, args.view, args.distance, args.layer)
    table, vectors = load_detail_vectors(args.detail_dir, args.detail_view, args.detail_layer, args.detail_key)
    semantic, audit = build_semantic(metadata, table, vectors, args.standardize == "global")
    kept = metadata[metadata["key"].isin(semantic)].drop_duplicates("story_key")
    if kept.empty:
        raise SystemExit(f"none of the {len(metadata)} deletion-graph stories has detail vectors in {args.detail_dir}")

    agreement = pd.DataFrame([
        {"story_key": r["story_key"], **edge_agreement(deletion[r["key"]], semantic[r["key"]], args.keep)} for _, r in kept.iterrows()
    ])
    subset = pd.read_csv(args.batch / "subset_stories.csv", low_memory=False)
    lower, upper = ("range_lower", "range_upper") if "range_lower" in subset.columns else ("age_lo", "age_hi")
    ages = pd.DataFrame({"story_key": subset["story_key"], "age_mid": (subset[lower] + subset[upper]) / 2}).drop_duplicates("story_key")
    agreement = agreement.merge(ages, on="story_key", how="left")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agreement.to_csv(args.output_dir / "edge_agreement.csv", index=False)
    pd.DataFrame(audit).to_csv(args.output_dir / "semantic_audit.csv", index=False)

    columns = ["rho_raw", "rho_excess", "top_overlap", "source_overlap", "impacted_overlap", "n_details",
               "r2_deletion_lag", "r2_deletion_gain_semantic", "r2_semantic_lag", "r2_semantic_gain_deletion"]
    tests = age_tests(agreement, columns)
    tests.to_csv(args.output_dir / "agreement_age_tests.csv", index=False)
    plot_by_age(agreement, pd.read_csv(args.batch / "membership.csv"), columns, args.output_dir / "fig_agreement_by_age.png")

    deletion_table = analyze.story_table(kept, deletion, args.keep)
    semantic_table = analyze.story_table(kept, semantic, args.keep)
    semantic_table.to_csv(args.output_dir / "semantic_story_measures.csv", index=False)
    comparison, overview = compare_units.compare(deletion_table, semantic_table, ages, names=("deletion", "semantic"))
    comparison.to_csv(args.output_dir / "measures_comparison.csv", index=False)
    overview.to_csv(args.output_dir / "measures_overview.csv", index=False)

    summary = {
        "deletion_stories": int(len(metadata)), "with_detail_vectors": int(len(kept)), "skipped": len(audit),
        "layer_deletion": layer, "detail_view": args.detail_view, "detail_layer": args.detail_layer, "standardize": args.standardize,
        "median_rho_raw": float(agreement["rho_raw"].median()), "median_rho_excess": float(agreement["rho_excess"].median()),
        "median_top_overlap": float(agreement["top_overlap"].median()), "median_top_overlap_chance": float(agreement["top_overlap_chance"].median()),
    }
    (args.output_dir / "run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
