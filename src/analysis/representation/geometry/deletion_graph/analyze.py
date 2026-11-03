"""Directed deletion-coupling graph of every story: graph measures per story and their change with age.

Reads the shards written by ``extract_deletion_graph.py``, picks one layer and one state view (``mean`` pooling over a detail's
tokens, or ``last`` token) and one distance (``cos`` or ``rel``), computes the measures of ``graph_math.graph_measures`` per
story, and relates them to age using a batch folder (``membership.csv`` and ``subset_stories.csv`` from spectral_age ``subsets.py``).

Outputs in ``--output-dir``: ``story_measures.csv`` (one row per story), ``age_summary.csv`` (per age: median and quartiles),
``age_tests.csv`` (Spearman of each measure with age, raw and with the number of details partialled out, two age definitions),
``causal_check.csv`` (the lower-triangle noise floor against the upper-triangle signal) and ``fig_measures_by_age.png``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.representation.geometry.deletion_graph.graph_math import MEASURE_LABELS, graph_measures


def load_shards(directory: Path, view: str, distance: str, layer: int | None) -> tuple[pd.DataFrame, dict[str, np.ndarray], int]:
    """Metadata of every story and its coupling matrix at one layer (``layer`` is a layer number; None = the last one saved)."""
    metadata, matrices, chosen = [], {}, None
    for path in sorted(directory.glob("deletion_graph_metadata_*.csv")):
        shard = path.stem.rsplit("_", 1)[1]
        table = pd.read_csv(path)
        with np.load(directory / f"deletion_graph_{shard}.npz") as data:
            layers = [int(value) for value in data["layers"]]
            chosen = layers[-1] if layer is None else layer
            position = layers.index(chosen)
            for row in table["row"]:
                matrices[f"{shard}:{row}"] = data[f"s{row:04d}_{distance}_{view}"][position]
        table["key"] = [f"{shard}:{row}" for row in table["row"]]
        metadata.append(table)
    if not metadata:
        raise SystemExit(f"no deletion_graph shards in {directory}")
    return pd.concat(metadata, ignore_index=True), matrices, int(chosen)


def story_table(metadata: pd.DataFrame, matrices: dict[str, np.ndarray], keep: float) -> pd.DataFrame:
    rows = []
    for _, record in metadata.iterrows():
        weights = matrices[record["key"]]
        measures = graph_measures(weights, keep)
        lower = np.tril(weights, -1)
        upper = np.triu(weights, 1)
        rows.append({
            "story_key": record["story_key"], "video_id": record["video_id"], "token_count": record["token_count"],
            **measures,
            "causal_floor": float(np.abs(lower).max()),
            "signal_median": float(np.median(upper[np.triu_indices(len(weights), 1)])),
        })
    return pd.DataFrame(rows).drop_duplicates("story_key")


def rank_residual(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    ranks, control_ranks = stats.rankdata(values), stats.rankdata(control)
    slope, intercept = np.polyfit(control_ranks, ranks, 1)
    return ranks - (slope * control_ranks + intercept)


def association(frame: pd.DataFrame, measure: str, age: str) -> dict[str, float]:
    sub = frame[[measure, age, "n_details"]].dropna()
    if len(sub) < 10 or sub[age].nunique() < 3:
        return {"n": len(sub), "rho": float("nan"), "p": float("nan"), "partial_rho": float("nan"), "partial_p": float("nan")}
    rho, p = stats.spearmanr(sub[measure], sub[age])
    partial, partial_p = stats.pearsonr(rank_residual(sub[measure].to_numpy(), sub["n_details"].to_numpy()),
                                        rank_residual(sub[age].to_numpy(), sub["n_details"].to_numpy()))
    return {"n": len(sub), "rho": float(rho), "p": float(p), "partial_rho": float(partial), "partial_p": float(partial_p)}


def plot_by_age(summary: pd.DataFrame, measures: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 4
    rows = int(np.ceil(len(measures) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.8 * rows), squeeze=False)
    for axis, measure in zip(axes.flat, measures):
        part = summary[summary["measure"] == measure].sort_values("age")
        axis.fill_between(part["age"], part["q25"], part["q75"], alpha=0.25)
        axis.plot(part["age"], part["median"], marker="o", markersize=3)
        axis.set_title(MEASURE_LABELS.get(measure, measure), fontsize=8)
        axis.set_xlabel("age")
    for axis in list(axes.flat)[len(measures):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="folder with the deletion_graph_*.npz shards")
    parser.add_argument("--batch", type=Path, required=True, help="batch folder with membership.csv and subset_stories.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", choices=["mean", "last"], default="mean")
    parser.add_argument("--distance", choices=["cos", "rel"], default="cos")
    parser.add_argument("--layer", type=int, default=None, help="layer number (default: the last one saved)")
    parser.add_argument("--keep", type=float, default=0.2, help="share of strongest pairs kept for the path and PageRank measures")
    args = parser.parse_args()

    metadata, matrices, layer = load_shards(args.input_dir, args.view, args.distance, args.layer)
    table = story_table(metadata, matrices, args.keep)
    membership = pd.read_csv(args.batch / "membership.csv")
    stories = pd.read_csv(args.batch / "subset_stories.csv", low_memory=False)[["story_key", "range_lower", "range_upper", "view_count"]]
    table = table.merge(stories.drop_duplicates("story_key"), on="story_key", how="left")
    table["age_mid"] = (table["range_lower"] + table["range_upper"]) / 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "story_measures.csv", index=False)

    measures = [name for name in MEASURE_LABELS if name in table.columns]
    by_age = membership.merge(table, on="story_key", how="inner")
    summary = by_age.melt(id_vars=["age", "story_key"], value_vars=measures, var_name="measure").groupby(["measure", "age"])["value"].agg(
        n="count", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)).reset_index()
    summary.to_csv(args.output_dir / "age_summary.csv", index=False)

    tests = []
    for measure in measures:
        tests.append({"measure": measure, "age_definition": "membership (story counted at every batch age)", **association(by_age, measure, "age")})
        tests.append({"measure": measure, "age_definition": "midpoint of the stated range (one row per story)", **association(table, measure, "age_mid")})
    pd.DataFrame(tests).to_csv(args.output_dir / "age_tests.csv", index=False)
    table[["story_key", "n_details", "causal_floor", "signal_median"]].assign(
        floor_over_signal=lambda f: f["causal_floor"] / f["signal_median"].replace(0, np.nan)
    ).to_csv(args.output_dir / "causal_check.csv", index=False)
    plot_by_age(summary, measures, args.output_dir / "fig_measures_by_age.png")
    (args.output_dir / "run.json").write_text(json.dumps({
        "stories": int(len(table)), "layer": layer, "view": args.view, "distance": args.distance, "keep": args.keep,
        "batch": str(args.batch), "median_floor_over_signal": float(np.nanmedian(table["causal_floor"] / table["signal_median"].replace(0, np.nan))),
    }, indent=2))
    print(f"{len(table)} stories, layer {layer}, {args.view}/{args.distance}; wrote {args.output_dir}")


if __name__ == "__main__":
    main()
