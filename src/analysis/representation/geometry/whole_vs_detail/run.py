"""How much of a story's whole-story vector do its own details account for, and does that change with age?  (no PCA)

Per story (``overlap.story_measures``): cosine of the whole-story vector with the mean detail, share of it inside the span of the
details, how fast the story's own detail order approaches it against shuffled orders, and how specifically the details identify the
whole.  Vectors come from the saved extractions (whole-story vectors and isolated or contextual detail vectors of one layer), are scaled
to unit length and centred on their corpus mean, so measures describe how a story *differs* from the average story.  Because the
measures are per story they do not depend on how many stories an age has.

Outputs in ``--output-dir``: ``story_measures.csv``, ``age_summary.csv`` (median and quartiles per age, a story counted at every batch
age it serves), ``age_tests.csv`` (Spearman with age, raw and with the number of details partialled out, for the midpoint of the stated
range and for batch membership), ``prefix_by_age.csv`` / ``fig_prefix_by_age.png`` (observed-minus-shuffled prefix cosine along the
story) and ``fig_overlap_by_age.png``.

    python -m analysis.representation.geometry.whole_vs_detail.run --batch-dir data/processed/spectral_age/batch_top100_ages2-12 \
        --model-key gpt_oss_20b --detail-view isolated --output-dir results/analysis/representation/geometry/whole_vs_detail/...
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.representation.geometry.whole_vs_detail import overlap

MEASURES = ["cos_mean", "cos_excess", "cos_detail", "r2_span_excess", "auc_identification", "order_gain", "prefix_auc"]
LABELS = {
    "cos_mean": "cosine: whole vs mean detail",
    "cos_excess": "cosine above other stories' details",
    "cos_detail": "mean cosine: whole vs single detail",
    "r2_span_excess": "share of whole inside detail span (above chance)",
    "auc_identification": "identification AUC (own vs other stories' details)",
    "order_gain": "order gain (real order minus shuffled)",
    "prefix_auc": "mean prefix cosine (real order)",
}


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


def analyze(ids: np.ndarray, whole: np.ndarray, details: pd.DataFrame, detail_vectors: np.ndarray, membership: pd.DataFrame,
            subset: pd.DataFrame, min_details: int = 4, draws: int = 20, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-story table and the observed-minus-shuffled prefix curves (one row per story, columns = relative position)."""
    rng = np.random.default_rng(seed)
    details = details.reset_index(drop=True)
    position = {key: i for i, key in enumerate(ids.astype(str))}
    wanted = set(membership["story_key"].astype(str)) & set(position)
    order = details[details["story_key"].astype(str).isin(wanted)].sort_values(["story_key", "detail_index"], kind="stable")
    groups = {key: group.index.to_numpy() for key, group in order.groupby("story_key", sort=True) if len(group) >= min_details}
    keys = sorted(groups)
    wholes = overlap.unit(whole[[position[k] for k in keys]].astype(np.float64))
    wholes = wholes - wholes.mean(axis=0)
    unit_details = overlap.unit(detail_vectors.astype(np.float64))
    centre = unit_details[np.concatenate([groups[k] for k in keys])].mean(axis=0)
    rows, curves, means = [], [], []
    for k, w in zip(keys, wholes):
        d = unit_details[groups[k]] - centre
        measures, curve = overlap.story_measures(w, d, rng, draws)
        rows.append({"story_key": k, **measures})
        curves.append(curve)
        means.append(d.mean(axis=0))
    table = pd.DataFrame(rows)
    auc, others = overlap.identification(wholes, np.array(means))
    table["auc_identification"] = auc
    table["cos_excess"] = table["cos_mean"] - others
    info = subset.drop_duplicates("story_key").set_index("story_key")
    ages = membership["age"].astype(int)
    low = info["range_lower"].reindex(table["story_key"]).to_numpy(dtype=float)
    high = info["range_upper"].reindex(table["story_key"]).to_numpy(dtype=float)
    table["age_mid"] = (np.clip(low, ages.min(), ages.max()) + np.clip(high, ages.min(), ages.max())) / 2
    curve_frame = pd.DataFrame(np.array(curves), columns=[f"{(i + 1) / len(curves[0]):.1f}" for i in range(len(curves[0]))])
    curve_frame.insert(0, "story_key", keys)
    return table, curve_frame


def summarize(table: pd.DataFrame, curves: pd.DataFrame, membership: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_age = membership.merge(table, on="story_key", how="inner")
    long = by_age.melt(id_vars=["age", "story_key"], value_vars=MEASURES, var_name="measure")
    summary = long.groupby(["measure", "age"])["value"].agg(n="count", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)).reset_index()
    tests = []
    for measure in MEASURES:
        tests.append({"measure": measure, "age_definition": "membership (story counted at every batch age)", **association(by_age.rename(columns={"age": "age_value"}), measure, "age_value")})
        tests.append({"measure": measure, "age_definition": "midpoint of the stated range (one row per story)", **association(table, measure, "age_mid")})
    prefix = membership.merge(curves, on="story_key", how="inner").drop(columns=["story_key"]).groupby("age").mean().reset_index()
    return {"summary": summary, "tests": pd.DataFrame(tests), "prefix": prefix}


def plot_by_age(summary: pd.DataFrame, prefix: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 4
    rows = int(np.ceil(len(MEASURES) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.8 * rows), squeeze=False)
    for axis, measure in zip(axes.flat, MEASURES):
        part = summary[summary["measure"] == measure].sort_values("age")
        axis.fill_between(part["age"], part["q25"], part["q75"], alpha=0.25)
        axis.plot(part["age"], part["median"], marker="o", markersize=3)
        axis.set_title(LABELS[measure], fontsize=8)
        axis.set_xlabel("age")
    for axis in list(axes.flat)[len(MEASURES):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "fig_overlap_by_age.png", dpi=140)
    plt.close(figure)

    grid = prefix.set_index("age")
    figure, axis = plt.subplots(figsize=(7, 4.2))
    image = axis.imshow(grid.to_numpy().T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-np.abs(grid.to_numpy()).max(), vmax=np.abs(grid.to_numpy()).max())
    axis.set_xticks(range(len(grid)), [str(int(a)) for a in grid.index])
    axis.set_yticks(range(len(grid.columns)), [f"{float(c):.0%}" for c in grid.columns])
    axis.set_xlabel("age")
    axis.set_ylabel("share of the story's details read")
    axis.set_title("cosine to the whole-story vector: real order minus shuffled order (mean over stories)", fontsize=9)
    figure.colorbar(image, label="cosine difference")
    figure.tight_layout()
    figure.savefig(output_dir / "fig_prefix_by_age.png", dpi=140)
    plt.close(figure)


def load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    spectral = importlib.import_module("analysis.representation.geometry.spectral_age.run")
    root = args.extraction_root or Path(spectral.base.DEFAULT_ROOT)
    names = spectral.MODEL_DIRS[args.model_key]
    global_dirs = [args.global_dir] if args.global_dir else [root / n for n in names["global"]]
    if args.detail_view == "isolated":
        detail_dirs = [args.detail_dir] if args.detail_dir else [root / n for n in names["isolated"]]
    else:
        detail_dirs = [args.detail_dir] if args.detail_dir else global_dirs
    ids, whole = spectral.whole_story(global_dirs, args.vector_key, args.layer)
    details = spectral.detail_table(detail_dirs, args.detail_view)
    details = details[details["story_key"].isin(set(ids.tolist()))].reset_index(drop=True)
    details = spectral.thinned_details(details, args.max_units).reset_index(drop=True)
    return ids, whole, details, spectral.detail_vectors(details, args.detail_view, args.layer, args.detail_key)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-dir", type=Path, required=True, help="batch folder with membership.csv and subset_stories.csv")
    parser.add_argument("--model-key", default="gpt_oss_20b", choices=["gpt_oss_20b", "qwen3_8b"])
    parser.add_argument("--extraction-root", type=Path, default=None)
    parser.add_argument("--global-dir", type=Path, default=None)
    parser.add_argument("--detail-dir", type=Path, default=None)
    parser.add_argument("--detail-view", choices=["isolated", "contextual"], default="isolated")
    parser.add_argument("--vector-key", default="mean_vectors", help="whole-story array: mean_vectors or eos_vectors")
    parser.add_argument("--detail-key", default=None)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--max-units", type=int, default=300, help="details kept per story (evenly thinned)")
    parser.add_argument("--min-details", type=int, default=4)
    parser.add_argument("--draws", type=int, default=20, help="shuffled detail orders per story")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    membership = pd.read_csv(args.batch_dir / "membership.csv", dtype={"story_key": str})
    subset = pd.read_csv(args.batch_dir / "subset_stories.csv", low_memory=False, dtype={"story_key": str})
    ids, whole, details, vectors = load_inputs(args)
    print(f"{len(ids)} stories with whole-story vectors, {len(details)} {args.detail_view} detail vectors", flush=True)
    table, curves = analyze(ids, whole, details, vectors, membership, subset, args.min_details, args.draws, args.seed)
    results = summarize(table, curves, membership)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "story_measures.csv", index=False)
    results["summary"].to_csv(args.output_dir / "age_summary.csv", index=False)
    results["tests"].to_csv(args.output_dir / "age_tests.csv", index=False)
    results["prefix"].to_csv(args.output_dir / "prefix_by_age.csv", index=False)
    plot_by_age(results["summary"], results["prefix"], args.output_dir)
    (args.output_dir / "run_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n")
    print(f"{len(table)} stories analysed; wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
