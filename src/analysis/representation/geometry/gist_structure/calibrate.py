"""Calibration on synthetic stories with known gist structure and embedding artifacts.

Before any age effect on real stories is believed, the measures must show they can tell apart the
things that could produce one.  Each scenario plants one thing that changes with the age band and
keeps the rest fixed; the table shows which statistics respond:

* ``null``            nothing changes with age (story size varies at random);
* ``compressibility`` older stories need fewer gists per detail (the hypothesis);
* ``reverse``         older stories need more gists per detail;
* ``complexity_only`` older stories are longer and have proportionally more gists (same ratio);
* ``size_only``       older stories are longer with the same absolute number of gists;
* ``noise_only``      older stories' details are more idiosyncratic (same gists, same size).

Synthetic embeddings carry the artifacts of real LLM embeddings: a power-law spread of per-dimension
variance, two "rogue" dimensions with huge variance, and a shared offset.  The legacy statistics
(components needed for a variance share, decay exponent) are included to show where they mislead.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from shared.io import write_csv

base = importlib.import_module("analysis.representation.geometry.within_story_compression.analyze")
ws = importlib.import_module("analysis.representation.geometry.gist_structure.within_story")

BANDS = [("a", 3.0), ("b", 5.0), ("c", 7.0), ("d", 9.0), ("e", 11.0)]
OUTCOMES = ["gist_count_rel", "gist_count", "gist_peak", "event_E5",
            "legacy_kn90", "legacy_kn20", "legacy_alpha", "legacy_head3", "legacy_k20"]


def simulate_story(rng: np.random.Generator, n: int, gists: int, noise: float, d: int, scale: np.ndarray, rogue: float) -> np.ndarray:
    gists = max(1, min(gists, n))
    centers = rng.normal(size=(gists, d))
    cuts = np.sort(rng.choice(np.arange(1, n), gists - 1, replace=False)) if gists > 1 else np.array([], dtype=int)
    assignment = np.zeros(n, dtype=int)
    for cut in cuts:
        assignment[cut:] += 1
    details = centers[assignment] + noise * rng.normal(size=(n, d))
    details = details * scale[None, :] + rng.normal(size=d)[None, :] * scale[None, :]
    details[:, :2] += rogue * rng.normal(size=(n, 2))
    return details


def scenario_parameters(name: str, position: float, rng: np.random.Generator) -> tuple[int, int, float]:
    """(n, gists, noise) for a story at ``position`` in [0, 1] along the age bands."""
    jitter = float(np.exp(rng.normal(0, 0.15)))
    if name == "null":
        n = int(round(70 * np.exp(rng.normal(0, 0.3))))
        return n, max(2, int(round(0.12 * n * jitter))), 1.0
    if name in ("compressibility", "reverse"):
        n = int(round((45 + 55 * position) * jitter))
        ratio = 0.30 - 0.24 * position if name == "compressibility" else 0.06 + 0.24 * position
        return n, max(2, int(round(ratio * n))), 1.0
    if name == "complexity_only":
        n = int(round((45 + 55 * position) * jitter))
        return n, max(2, int(round(0.12 * n))), 1.0
    if name == "size_only":
        n = int(round((45 + 55 * position) * jitter))
        return n, max(2, int(round(8 * float(np.exp(rng.normal(0, 0.1)))))), 1.0
    if name == "noise_only":
        n = int(round(70 * jitter))
        return n, 8, 0.6 + 1.2 * position
    raise ValueError(name)


def legacy_stats(z: np.ndarray) -> dict[str, float]:
    n = len(z)
    centered = z - z.mean(axis=0)
    eigenvalues = base.eigenvalues(base.center_gram(centered @ centered.T) / (n - 1))
    stats = base.spectrum_stats(eigenvalues, n)
    return {"legacy_kn90": stats["kn90"], "legacy_kn20": stats["kn20"], "legacy_alpha": stats["alpha"],
            "legacy_head3": stats["head3"], "legacy_k20": stats["k20"]}


def build_frame(
    name: str, per_band: int, d: int, fixed_units: int, cfg: argparse.Namespace, rng: np.random.Generator,
) -> pd.DataFrame:
    scale = (1.0 + np.arange(d)) ** -0.25
    scale = scale / scale.mean()
    rows = []
    for band_number, (band, mid) in enumerate(BANDS):
        for story in range(per_band):
            n, gists, noise = scenario_parameters(name, band_number / (len(BANDS) - 1), rng)
            details = simulate_story(rng, max(n, 25), gists, noise, d, scale, rogue=15.0)
            if fixed_units:
                if len(details) < fixed_units:
                    continue
                details = details[np.sort(rng.choice(len(details), fixed_units, replace=False))]
            normalized = base.normalize_rows(details)
            z = normalized / (normalized.std(axis=0) + 1e-6)
            story_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            row = ws.unit_measures(z, story_rng, cfg)
            row.update(legacy_stats(z))
            row.update({"band": band, "band_mid": mid, "n_units": float(len(details)), "story_key": f"{name}-{band}-{story}",
                        "author_group": f"{name}-{band}-{story}", "true_gists": float(gists), "true_ratio": gists / len(details)})
            rows.append(row)
    return pd.DataFrame(rows)


def effects_table(frame: pd.DataFrame, scenario: str, setting: str, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        for model in ("M0", "M1"):
            result = ws.fit_age_effect(frame, outcome, ws.MODELS[model], rng, draws=100, permutations=100)
            if result is not None:
                rows.append({"scenario": scenario, "setting": setting, "outcome": outcome, "model": model, **result})
    return pd.DataFrame(rows)


def run_calibration(
    scenarios: list[str], per_band: int, d: int, fixed_units: int, seed: int,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    cfg = argparse.Namespace(folds=10, null_draws=3, vq=False, vq_folds=5)
    tables = []
    for scenario in scenarios:
        for setting, fixed in (("all_units", 0), (f"fixed_{fixed_units}", fixed_units)) if fixed_units else (("all_units", 0),):
            rng = np.random.default_rng(seed + hash((scenario, setting)) % 10_000)
            frame = build_frame(scenario, per_band, d, fixed, cfg, rng)
            tables.append(effects_table(frame, scenario, setting, rng))
            log(f"calibrated {scenario} / {setting}: {len(frame)} stories")
    return pd.concat(tables, ignore_index=True)


def summarise(effects: pd.DataFrame) -> pd.DataFrame:
    """Standardised size-adjusted age effect (M1) per scenario and outcome, starred when the permutation p < 0.05."""
    table = effects[effects["model"] == "M1"].copy()
    table["cell"] = table.apply(lambda row: f"{row['beta_std_per_year']:+.2f}{'*' if row['p_perm'] < 0.05 else ' '}", axis=1)
    return table.pivot_table(index=["scenario", "setting"], columns="outcome", values="cell", aggfunc="first")[
        [name for name in OUTCOMES if name in set(table["outcome"])]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", default="null,compressibility,reverse,complexity_only,size_only,noise_only")
    parser.add_argument("--per-band", type=int, default=30)
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--fixed-units", type=int, default=45)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/representation/geometry/gist_structure/calibration"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effects = run_calibration(args.scenarios.split(","), args.per_band, args.dimensions, args.fixed_units, args.seed)
    write_csv(args.output_dir / "calibration_effects.csv", effects)
    table = summarise(effects)
    write_csv(args.output_dir / "calibration_summary.csv", table.reset_index())
    print("\nstandardised age effect per year, adjusted for story size (M1); * = permutation p < 0.05\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
