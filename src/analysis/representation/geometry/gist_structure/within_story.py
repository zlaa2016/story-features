"""Within-story gist structure by age band, with confound controls.

Per story (isolated detail vectors, one story at a time, no PCA basis shared across stories):

* held-out gist curve: how much of left-out details m PCs of the other details reconstruct, minus
  a column-permutation null; summarised as strength (``gist_peak``), the number of gists where the
  curve saturates (``gist_count``) and that number per detail (``gist_count_rel``);
* event structure: variance explained by the best k contiguous segments, minus a shuffled-order null;
* held-out vector quantisation (nonlinear "unique gists");
* the same measures again after near-duplicate details (refrains) are removed (``*_dd``).

The age effect is estimated with four nested models (age only; + story size; + length and lexical
complexity covariates; + topic cluster and popularity), a bootstrap that resamples authors, and a
permutation that shuffles age within topic x size strata.
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.io import write_csv, write_json
from shared.research import benjamini_hochberg, write_run_provenance

base = importlib.import_module("analysis.representation.geometry.within_story_compression.analyze")
gm = importlib.import_module("analysis.representation.geometry.gist_structure.gist_math")

DEFAULT_ROOT = "results/analysis/representation/geometry/gist_structure"
PRIMARY = "gist_count_rel"
MODELS: dict[str, list[str]] = {
    "M0": [],
    "M1": ["log_n"],
    "M2": ["log_n", "mean_detail_tokens", "lex_ttr", "mean_word_len", "dup_share", "range_width"],
    "M3": ["log_n", "mean_detail_tokens", "lex_ttr", "mean_word_len", "dup_share", "range_width", "topic", "popularity"],
}
OUTCOMES = [
    "gist_count_rel", "gist_half_rel", "gist_count", "gist_half", "gist_peak", "gist_auc10", "gist_E1", "gist_E3",
    "event_peak", "event_E3", "event_E5", "event_k_peak_rel",
    "vq_count_rel", "vq_peak", "vq_E3",
    "dd_gist_count_rel", "dd_gist_peak", "dd_event_E5",
    "mean_pairwise_cos", "dup_share",
]


# --------------------------------------------------------------------------------------
# Per-story measures
# --------------------------------------------------------------------------------------
def unit_measures(z: np.ndarray, rng: np.random.Generator, cfg: argparse.Namespace, prefix: str = "") -> dict[str, float]:
    n = len(z)
    out: dict[str, float] = {}
    real, _, excess = gm.excess_curve(gm.oos_pca_curve, z, rng, cfg.null_draws, folds=cfg.folds)
    for m, value, gain in zip(gm.M_GRID, real, excess):
        out[f"oosR_m{m}"] = float(value)
        out[f"oosE_m{m}"] = float(gain)
    out.update(gm.saturation_scalars(excess, gm.M_GRID, n, "gist"))
    spectrum, spectrum_null = gm.excess_spectrum(z, rng, cfg.null_draws)
    for m, value, reference in zip(gm.M_GRID, spectrum, spectrum_null):
        out[f"cum_r{m}"] = float(value)
        out[f"cumexc_r{m}"] = float(value - reference)

    ks = [k for k in gm.EVENT_KS if k <= max(2, n // 3)]
    _, _, event = gm.event_excess(z @ z.T, rng, cfg.null_draws, ks)
    for k, value in zip(ks, event):
        out[f"evE_k{k}"] = float(value)
        if k in (3, 5, 8):
            out[f"event_E{k}"] = float(value)
    if np.isfinite(event).any():
        best = int(np.nanargmax(event))
        out["event_peak"] = float(event[best])
        out["event_k_peak"] = float(ks[best])
        out["event_k_peak_rel"] = float(ks[best]) / n

    if cfg.vq:
        _, _, quantised = gm.excess_curve(gm.vq_curve, z, rng, cfg.null_draws, folds=cfg.vq_folds)
        for m, value in zip(gm.VQ_MS, quantised):
            out[f"vqE_m{m}"] = float(value)
        out.update(gm.saturation_scalars(quantised, gm.VQ_MS, n, "vq"))
    return {f"{prefix}{key}": value for key, value in out.items()}


def gist_profile(z: np.ndarray, normalized: np.ndarray, threshold: float, rng: np.random.Generator, cfg: argparse.Namespace) -> dict[str, float]:
    """All gist measures for one story, plus the duplicate-removed (dd) version."""
    n = len(z)
    row: dict[str, float] = {"n_units": float(n)}
    similarity = normalized @ normalized.T
    row["mean_pairwise_cos"] = float((similarity.sum() - np.trace(similarity)) / (n * (n - 1)))
    keep = gm.dedupe_keep(normalized, threshold)
    row["dup_share"] = float(1.0 - keep.mean())
    row.update(unit_measures(z, rng, cfg))
    if cfg.dedupe:
        if keep.all():
            row.update({f"dd_{key}": value for key, value in unit_measures(z, rng, cfg).items()})
            row["n_units_dd"] = float(n)
        elif keep.sum() >= cfg.min_units:
            row.update(unit_measures(z[keep], rng, cfg, prefix="dd_"))
            row["n_units_dd"] = float(keep.sum())
    return row


_STATE: dict[str, Any] = {}


def _profile_task(story_key: str) -> dict[str, float]:
    normalized, z = _STATE["matrices"][story_key]
    rng = base.stable_rng(_STATE["seed"], story_key)
    row = gist_profile(z, normalized, _STATE["threshold"], rng, _STATE["cfg"])
    row["story_key"] = story_key
    return row


def profile_matrices(matrices: dict[str, tuple[np.ndarray, np.ndarray]], threshold: float, cfg: argparse.Namespace, workers: int) -> pd.DataFrame:
    """Profile every story; ``matrices[story] = (row-normalised vectors, scaled vectors)``."""
    _STATE.update(matrices=matrices, threshold=threshold, cfg=cfg, seed=cfg.seed)
    keys = list(matrices)
    if workers > 1:
        context = multiprocessing.get_context("fork")
        with context.Pool(workers) as pool:
            rows = []
            for number, row in enumerate(pool.imap(_profile_task, keys, chunksize=4), start=1):
                rows.append(row)
                if number % 100 == 0:
                    print(f"profiled {number}/{len(keys)} stories", flush=True)
    else:
        rows = []
        for number, key in enumerate(keys, start=1):
            rows.append(_profile_task(key))
            if number % 100 == 0:
                print(f"profiled {number}/{len(keys)} stories", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Age effects with controls
# --------------------------------------------------------------------------------------
def _clean(frame: pd.DataFrame, outcome: str, controls: list[str], age_col: str) -> pd.DataFrame | None:
    needed = {"n_units", age_col, outcome, "author_group"}
    if "topic" in controls:
        needed.add("topic_cluster")
    needed |= {name for name in controls if name not in ("log_n", "topic")}
    if any(name not in frame.columns for name in needed):
        return None
    valid = frame[sorted(needed)].replace([np.inf, -np.inf], np.nan).dropna()
    return valid if len(valid) >= 30 and valid[outcome].nunique() > 2 else None


def _design(valid: pd.DataFrame, controls: list[str], age_col: str) -> np.ndarray:
    columns = [np.ones(len(valid)), valid[age_col].to_numpy(dtype=float)]
    for name in controls:
        if name == "log_n":
            columns.append(np.log(valid["n_units"].to_numpy(dtype=float)))
        elif name == "topic":
            dummies = pd.get_dummies(valid["topic_cluster"].astype(int), drop_first=True).to_numpy(dtype=float)
            columns.extend(dummies.T)
        else:
            values = valid[name].to_numpy(dtype=float)
            columns.append((values - values.mean()) / (values.std() + 1e-12))
    return np.column_stack(columns)


def _strata(valid: pd.DataFrame) -> list[np.ndarray]:
    size_bins = pd.qcut(valid["n_units"].rank(method="first"), q=min(3, max(1, len(valid) // 40)), labels=False)
    topic = valid["topic_cluster"].astype(int) if "topic_cluster" in valid.columns else pd.Series(0, index=valid.index)
    labels = (topic.astype(str) + "|" + size_bins.astype(str)).to_numpy()
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    return [members for members in groups if len(members) > 1]


def fit_age_effect(
    frame: pd.DataFrame, outcome: str, controls: list[str], rng: np.random.Generator,
    age_col: str = "band_mid", draws: int = 300, permutations: int = 300,
) -> dict[str, float] | None:
    valid = _clean(frame, outcome, controls, age_col)
    if valid is None:
        return None
    design = _design(valid, controls, age_col)
    response = valid[outcome].to_numpy(dtype=float)
    beta = float(np.linalg.lstsq(design, response, rcond=None)[0][1])
    scale = float(response.std()) or 1.0

    clusters = valid["author_group"].astype(str).to_numpy()
    members = [np.flatnonzero(clusters == name) for name in np.unique(clusters)]
    boot = []
    for _ in range(draws):
        pick = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        boot.append(np.linalg.lstsq(design[pick], response[pick], rcond=None)[0][1])
    groups = _strata(valid)
    age = valid[age_col].to_numpy(dtype=float)
    exceed = 0
    for _ in range(permutations):
        shuffled = age.copy()
        for group in groups:
            shuffled[group] = age[rng.permutation(group)]
        local = design.copy()
        local[:, 1] = shuffled
        if abs(np.linalg.lstsq(local, response, rcond=None)[0][1]) >= abs(beta) - 1e-12:
            exceed += 1
    return {
        "n": len(valid), "n_clusters": len(members), "beta_per_year": beta, "beta_std_per_year": beta / scale,
        "ci_low_std": float(np.quantile(boot, 0.025)) / scale, "ci_high_std": float(np.quantile(boot, 0.975)) / scale,
        "p_perm": (1 + exceed) / (1 + permutations),
    }


def age_effects(frame: pd.DataFrame, outcomes: list[str], rng: np.random.Generator, cfg: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        for model, controls in MODELS.items():
            result = fit_age_effect(frame, outcome, controls, rng, draws=cfg.bootstrap, permutations=cfg.permutations)
            if result is not None:
                rows.append({"outcome": outcome, "model": model, **result})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["q_bh"] = np.nan
        for model in table["model"].unique():
            selected = table["model"] == model
            table.loc[selected, "q_bh"] = benjamini_hochberg(table.loc[selected, "p_perm"])
    return table


def band_summary(frame: pd.DataFrame, outcomes: list[str], rng: np.random.Generator, draws: int) -> pd.DataFrame:
    rows = []
    for band, group in frame.groupby("band", sort=False):
        clusters = group["author_group"].astype(str).to_numpy()
        members = [np.flatnonzero(clusters == name) for name in np.unique(clusters)]
        for outcome in outcomes:
            if outcome not in group.columns:
                continue
            values = group[outcome].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if finite.sum() < 3:
                continue
            means = []
            for _ in range(draws):
                pick = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
                chosen = values[pick]
                means.append(np.nanmean(chosen) if np.isfinite(chosen).any() else np.nan)
            rows.append({
                "band": band, "band_mid": float(group["band_mid"].iloc[0]), "outcome": outcome, "n": int(finite.sum()),
                "mean": float(np.nanmean(values)), "ci_low": float(np.nanquantile(means, 0.025)), "ci_high": float(np.nanquantile(means, 0.975)),
            })
    return pd.DataFrame(rows)


def hypothesis_check(effects: pd.DataFrame) -> dict[str, Any]:
    """Primary prediction: gists per detail fall with age (negative standardised effect)."""
    out: dict[str, Any] = {"primary_outcome": PRIMARY, "prediction": "older stories need fewer gists per detail (negative effect)"}
    for outcome in (PRIMARY, "dd_" + PRIMARY, "event_k_peak_rel", "vq_count_rel"):
        for model in MODELS:
            row = effects[(effects["outcome"] == outcome) & (effects["model"] == model)]
            if row.empty:
                continue
            row = row.iloc[0]
            if row["ci_high_std"] < 0 and row["p_perm"] < 0.05:
                verdict = "supports"
            elif row["ci_low_std"] > 0 and row["p_perm"] < 0.05:
                verdict = "contradicts"
            else:
                verdict = "inconclusive"
            out[f"{outcome}|{model}"] = {
                "beta_std_per_year": float(row["beta_std_per_year"]), "ci": [float(row["ci_low_std"]), float(row["ci_high_std"])],
                "p_perm": float(row["p_perm"]), "verdict": verdict,
            }
    return out


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def age_values(frame: pd.DataFrame) -> pd.Series:
    """Age axis for the spectrum figures: the band midpoint, or integer ages when the data are continuous."""
    ages = frame["band_mid"].astype(float)
    return ages.round() if ages.nunique() > 12 else ages


def spectrum_by_age(frame: pd.DataFrame, prefix: str, ranks: tuple[int, ...] = gm.M_GRID) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``<prefix><m>`` over the stories at each age: (sorted ages, matrix of shape ranks x ages)."""
    ages = age_values(frame)
    ordered = np.sort(ages.unique())
    matrix = np.array([[frame.loc[ages == age, f"{prefix}{m}"].mean() for age in ordered] for m in ranks])
    return ordered, matrix


def draw_age_by_spectrum(panels: list[dict[str, Any]], ages: np.ndarray, ranks: tuple[int, ...], path: Path, title: str, note: str) -> None:
    """Heatmaps (age on x, component rank on y) over line plots of the same curves (one line per age)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(panels)
    figure, axes = plt.subplots(2, count, figsize=(5.4 * count, 8.2), facecolor="#fcfcfb")
    shades = plt.cm.Blues(np.linspace(0.3, 0.95, max(len(ages), 2)))
    for column, panel in enumerate(panels):
        matrix = panel["matrix"]
        heat = axes[0][column]
        if panel.get("diverging"):
            limit = max(float(np.nanmax(np.abs(matrix))), 1e-6)
            image = heat.imshow(matrix, aspect="auto", origin="lower", cmap=panel["cmap"], vmin=-limit, vmax=limit)
        else:
            image = heat.imshow(matrix, aspect="auto", origin="lower", cmap="Blues")
        heat.set_xticks(range(len(ages)))
        heat.set_xticklabels([f"{age:g}" for age in ages], fontsize=8)
        heat.set_yticks(range(len(ranks)))
        heat.set_yticklabels([str(m) for m in ranks], fontsize=8)
        heat.set_xlabel("age")
        heat.set_ylabel("number of components m")
        heat.set_title(panel["title"], fontsize=10, loc="left")
        figure.colorbar(image, ax=heat, shrink=0.85)
        line = axes[1][column]
        curves = panel["lines"] if panel.get("lines") is not None else matrix
        for index, age in enumerate(ages):
            line.plot(ranks, curves[:, index], color=shades[index], lw=2, label=f"age {age:g}")
        line.set_xscale("log")
        line.set_xlabel("number of components m")
        line.set_ylabel(panel["ylabel"], fontsize=9)
        line.set_title(panel["line_title"], fontsize=10, loc="left")
        if panel.get("zero"):
            line.axhline(0, color="#6b6b66", lw=0.8)
        if column == 0:
            line.legend(fontsize=7, ncol=2)
    figure.suptitle(title, fontsize=12, x=0.01, ha="left")
    figure.text(0.01, 0.005, note, fontsize=8.5, color="#6b6b66")
    figure.tight_layout(rect=(0, 0.02, 1, 0.95))
    figure.savefig(path, dpi=140, facecolor="#fcfcfb")
    plt.close(figure)


def make_age_spectrum_figure(frame: pd.DataFrame, output_dir: Path, title: str = "Variance spectrum by age") -> None:
    """The headline result: how a story's own spectrum and held-out gist curve change with age."""
    ages, cumulative = spectrum_by_age(frame, "cum_r")
    _, held_out = spectrum_by_age(frame, "oosE_m")
    change = held_out - held_out[:, [0]]
    panels = [
        {"matrix": cumulative, "title": "variance explained by the top m components (story's own spectrum)",
         "ylabel": "cumulative variance explained", "line_title": "same, one line per age"},
        {"matrix": held_out, "title": "held-out gist curve: reconstruction of left-out details, beyond the null",
         "ylabel": "excess held-out reconstruction", "line_title": "same, one line per age"},
        {"matrix": change, "title": "held-out gist curve minus the youngest age (what changes with age)", "diverging": True,
         "cmap": "RdBu_r", "ylabel": "difference from youngest age", "line_title": "same, one line per age", "zero": True},
    ]
    note = ("Each column of a heatmap is one age (mean over its stories, equal numbers per band when the matched selection is used). "
            "Left: the story's own spectrum (in-sample). Middle: how much of left-out details the top m components generalise to, minus a column-shuffle null. "
            "Right: the middle panel relative to the youngest age; red = older stories reconstruct better with m components.")
    draw_age_by_spectrum(panels, ages, gm.M_GRID, output_dir / "fig_age_by_spectrum.png", title, note)


def make_plots(frame: pd.DataFrame, effects: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = list(frame.sort_values("band_mid")["band"].drop_duplicates())
    colors = dict(zip(bands, plt.cm.viridis(np.linspace(0, 1, max(len(bands), 2)))))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    for band in bands:
        members = frame[frame["band"] == band]
        curve = np.array([members[f"oosE_m{m}"].mean() for m in gm.M_GRID])
        axes[0].plot(gm.M_GRID, curve, color=colors[band], label=f"{band} (n={len(members)})")
        ks = [k for k in gm.EVENT_KS if f"evE_k{k}" in members.columns]
        axes[1].plot(ks, [members[f"evE_k{k}"].mean() for k in ks], color=colors[band])
        if "vqE_m2" in members.columns:
            axes[2].plot(gm.VQ_MS, [members[f"vqE_m{m}"].mean() for m in gm.VQ_MS], color=colors[band])
    axes[0].set_xscale("log")
    axes[0].set_title("held-out PCA gist curve, excess over null")
    axes[0].set_xlabel("components m")
    axes[1].set_title("event structure, excess over shuffled order")
    axes[1].set_xlabel("contiguous segments k")
    axes[2].set_title("held-out vector quantisation, excess over null")
    axes[2].set_xlabel("centroids m")
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "fig_gist_curves_by_band.png", dpi=150)
    plt.close(figure)

    outcomes = [name for name in (PRIMARY, "gist_peak", "event_k_peak_rel", "event_E5", "vq_count_rel", "mean_pairwise_cos") if name in set(effects["outcome"])]
    if outcomes:
        figure, axes = plt.subplots(1, len(outcomes), figsize=(3.2 * len(outcomes), 3.6), squeeze=False)
        for axis, outcome in zip(axes[0], outcomes):
            for position, model in enumerate(MODELS):
                row = effects[(effects["outcome"] == outcome) & (effects["model"] == model)]
                if row.empty:
                    continue
                row = row.iloc[0]
                axis.errorbar(position, row["beta_std_per_year"], yerr=[[row["beta_std_per_year"] - row["ci_low_std"]], [row["ci_high_std"] - row["beta_std_per_year"]]], fmt="o")
            axis.axhline(0, color="black", lw=0.6)
            axis.set_xticks(range(len(MODELS)))
            axis.set_xticklabels(list(MODELS), fontsize=8)
            axis.set_title(outcome, fontsize=8)
        axes[0][0].set_ylabel("standardised age effect per year")
        figure.suptitle("M0 age; M1 + size; M2 + length/lexical/duplicates; M3 + topic and popularity")
        figure.tight_layout()
        figure.savefig(output_dir / "fig_age_effects_by_model.png", dpi=150)
        plt.close(figure)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def prepare_matrices(args: argparse.Namespace, fixed_units: int) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float, pd.DataFrame]:
    details = base.load_metadata(args.isolated_dir, "isolated")
    selected = pd.read_csv(args.selected_csv, low_memory=False)
    details = details[details["story_key"].isin(selected["story_key"])]
    details = details.sort_values(["story_key", "detail_index"]).reset_index(drop=True)
    vectors = base.normalize_rows(base.load_vectors(args.isolated_dir, "isolated", details, args.layer, args.vector_key))
    keys = details["story_key"].to_numpy()
    edges = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    slices = {keys[a]: slice(a, b) for a, b in zip(edges[:-1], edges[1:])}

    rng = np.random.default_rng(args.seed)
    pool_index = rng.choice(len(vectors), size=min(args.pool_size, len(vectors)), replace=False)
    pool = vectors[pool_index]
    scale = pool.std(axis=0) + 1e-6
    threshold = gm.duplicate_threshold(pool, keys[pool_index], rng)

    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for story_key, window in slices.items():
        rows = vectors[window]
        if fixed_units > 0:
            if len(rows) < fixed_units:
                continue
            rows = rows[np.sort(base.stable_rng(args.seed, f"fixed|{story_key}").choice(len(rows), fixed_units, replace=False))]
        rows, _ = base.thin_units(rows, args.max_units)
        if len(rows) >= args.min_units:
            matrices[story_key] = (rows, rows / scale)
    covariates = base.story_covariates(details)
    return matrices, threshold, covariates


def inclusion_report(selected: pd.DataFrame, kept: pd.DataFrame) -> pd.DataFrame:
    """Stories kept after length filters vs selected, per band.  Uneven loss across bands is itself a confound."""
    rows = []
    for band, group in selected.groupby("band", sort=False):
        remaining = int(kept["story_key"].isin(group["story_key"]).sum())
        rows.append({"band": band, "band_mid": float(group["band_mid"].iloc[0]), "selected": len(group), "kept": remaining,
                     "share_kept": remaining / len(group)})
    return pd.DataFrame(rows).sort_values("band_mid")


def real_data_controls(
    matrices: dict[str, tuple[np.ndarray, np.ndarray]], frame: pd.DataFrame, cfg: argparse.Namespace,
    rng: np.random.Generator, per_band: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate the measures on the real embeddings.

    * merge control: two real stories of the same band are concatenated; if the gist count means
      anything it should be about the sum of the two stories' counts, and the event structure of the
      concatenation should vanish when its order is shuffled;
    * split-half reliability: a story's details are split at random into two halves and each half is
      scored; a measure whose halves do not agree cannot carry an age effect.
    """
    from scipy.stats import spearmanr

    scored = frame.set_index("story_key")
    merge_rows = []
    for band, group in frame.groupby("band", sort=False):
        members = [key for key in group["story_key"] if key in matrices]
        if len(members) < 4:
            continue
        for _ in range(per_band):
            first, second = rng.choice(members, 2, replace=False)
            merged = np.vstack([matrices[first][1], matrices[second][1]])
            joined = unit_measures(merged, rng, cfg)
            shuffled = unit_measures(merged[rng.permutation(len(merged))], rng, cfg)
            merge_rows.append({
                "band": band, "story_a": first, "story_b": second,
                "count_a": scored.loc[first, "gist_count"], "count_b": scored.loc[second, "gist_count"],
                "count_merged": joined.get("gist_count", np.nan),
                "peak_a": scored.loc[first, "gist_peak"], "peak_b": scored.loc[second, "gist_peak"], "peak_merged": joined.get("gist_peak", np.nan),
                "event_peak_ordered": joined.get("event_peak", np.nan), "event_peak_shuffled": shuffled.get("event_peak", np.nan),
            })
    merges = pd.DataFrame(merge_rows)

    measures = ["gist_peak", "gist_count", "gist_half", "event_peak"] + (["vq_peak"] if cfg.vq else [])
    half_rows = []
    for key, (_, z) in matrices.items():
        order = rng.permutation(len(z))
        halves = [unit_measures(z[order[: len(z) // 2]], rng, cfg), unit_measures(z[order[len(z) // 2:]], rng, cfg)]
        half_rows.append({"story_key": key, **{f"{name}_h{number + 1}": halves[number].get(name, np.nan) for name in measures for number in (0, 1)}})
    halves_frame = pd.DataFrame(half_rows)
    reliability = []
    for name in measures:
        pair = halves_frame[[f"{name}_h1", f"{name}_h2"]].dropna()
        correlation = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic if len(pair) >= 10 else float("nan")
        reliability.append({"measure": name, "n_stories": len(pair), "split_half_rho": float(correlation),
                            "spearman_brown": float(2 * correlation / (1 + correlation)) if np.isfinite(correlation) else float("nan")})
    summary: dict[str, Any] = {}
    valid = merges.dropna(subset=["count_a", "count_b", "count_merged"]) if not merges.empty else merges
    if len(valid) >= 10:
        total = valid["count_a"] + valid["count_b"]
        summary["merge_count_over_sum_median"] = float((valid["count_merged"] / total).median())
        summary["merge_count_vs_sum_spearman"] = float(spearmanr(valid["count_merged"], total).statistic)
        summary["merge_expectation"] = "count of a merged pair is about the sum of the two stories' counts (ratio near 1)"
    if len(merges):
        summary["event_peak_ordered_minus_shuffled_median"] = float((merges["event_peak_ordered"] - merges["event_peak_shuffled"]).median())
    return merges, pd.DataFrame(reliability), summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isolated-dir", type=Path, default=Path(f"{base.DEFAULT_ROOT}/gpt_oss_20b_detail_controls"))
    parser.add_argument("--selected-csv", type=Path, default=Path(f"{DEFAULT_ROOT}/selection/selected_stories.csv"))
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--vector-key", default=None)
    parser.add_argument("--fixed-units", default="0", help="comma list; 0 = use every unit, N = subsample each story to N units")
    parser.add_argument("--min-units", type=int, default=20)
    parser.add_argument("--max-units", type=int, default=300)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--null-draws", type=int, default=3)
    parser.add_argument("--no-vq", dest="vq", action="store_false")
    parser.add_argument("--vq-folds", type=int, default=5)
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    parser.add_argument("--no-real-controls", dest="real_controls", action="store_false", help="skip the merge control and split-half reliability")
    parser.add_argument("--control-pairs", type=int, default=30, help="merged story pairs per band")
    parser.add_argument("--pool-size", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--permutations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(f"{DEFAULT_ROOT}/within_story/gpt_oss_20b"))
    return parser.parse_args(argv)


def run_one(args: argparse.Namespace, fixed_units: int) -> None:
    label = "all_units" if fixed_units == 0 else f"fixed_{fixed_units}"
    output_dir = args.output_dir / label
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices, threshold, covariates = prepare_matrices(args, fixed_units)
    print(f"[{label}] {len(matrices)} stories, duplicate cosine threshold {threshold:.3f}", flush=True)
    profiles = profile_matrices(matrices, threshold, args, args.workers)
    selected = pd.read_csv(args.selected_csv, low_memory=False)
    selected = selected.drop(columns=[c for c in covariates.columns if c != "story_key" and c in selected.columns])
    frame = selected.merge(profiles, on="story_key", how="inner").merge(covariates, on="story_key", how="left")
    if "author_group" not in frame.columns:
        frame["author_group"] = frame["story_key"]
    write_csv(output_dir / "story_gist_profiles.csv", frame)
    inclusion = inclusion_report(selected, frame)
    write_csv(output_dir / "inclusion_by_band.csv", inclusion)
    print(f"[{label}] stories kept / selected per band:\n{inclusion.to_string(index=False)}", flush=True)

    rng = np.random.default_rng(args.seed + 7)
    if args.real_controls:
        merges, reliability, validation = real_data_controls(matrices, frame, args, rng, per_band=args.control_pairs)
        write_csv(output_dir / "real_data_merge_control.csv", merges)
        write_csv(output_dir / "split_half_reliability.csv", reliability)
        write_json(output_dir / "real_data_validation.json", validation)
        print(f"[{label}] split-half reliability:\n{reliability.to_string(index=False)}\n[{label}] merge control: {validation}", flush=True)
    outcomes = [name for name in OUTCOMES if name in frame.columns]
    effects = age_effects(frame, outcomes, rng, args)
    write_csv(output_dir / "age_effects.csv", effects)
    write_csv(output_dir / "band_summary.csv", band_summary(frame, outcomes, rng, min(args.bootstrap, 200)))
    write_json(output_dir / "hypothesis_check.json", hypothesis_check(effects))
    make_plots(frame, effects, output_dir)
    make_age_spectrum_figure(frame, output_dir, f"{label}: variance spectrum and gist curve by age")
    write_run_provenance(output_dir, args, extra={"label": label, "stories": int(len(frame)), "duplicate_threshold": threshold})


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for value in [int(item) for item in str(args.fixed_units).split(",") if item.strip() != ""]:
        run_one(args, value)


if __name__ == "__main__":
    main()
