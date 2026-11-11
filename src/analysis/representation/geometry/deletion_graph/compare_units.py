"""Event graphs against detail graphs of the same stories.

Takes the ``story_measures.csv`` of an event-unit run and of a detail-unit run (``analyze.py`` output of each), keeps the stories
present in both whose texts match (token counts within ``--token-ratio``; an event file that belongs to another story of the same
video does not), and writes ``units_comparison.csv``, one row per measure:

* ``rho_units``: Spearman across stories between the event value and the detail value (do both graphs rank stories alike?),
* medians of both and a Wilcoxon signed-rank p for their difference (graphs of different size differ in level: read with ``n_*``),
* ``rho_age_event`` / ``rho_age_detail``: Spearman of each with the story's middle age (``age_lo``..``age_hi`` of the batch).

``units_overview.csv`` has the story counts and the number of units per story in each version.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

NON_MEASURES = {"story_key", "video_id", "token_count", "n_details", "n_pairs", "causal_floor", "signal_median"}


def measure_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in NON_MEASURES and pd.api.types.is_numeric_dtype(frame[c])]


def spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 10 or pair.iloc[:, 0].nunique() < 3 or pair.iloc[:, 1].nunique() < 3:
        return float("nan")
    return float(stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])[0])


def compare(events: pd.DataFrame, details: pd.DataFrame, ages: pd.DataFrame, token_ratio: float = 1.25,
            names: tuple[str, str] = ("event", "detail")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``events``/``details`` are the two tables to compare; ``names`` label them in the output columns."""
    first, second = names
    both = events.merge(details, on="story_key", suffixes=(f"_{first}", f"_{second}"))
    ratio = both[f"token_count_{first}"] / both[f"token_count_{second}"]
    kept = both[(ratio <= token_ratio) & (ratio >= 1 / token_ratio)].merge(ages, on="story_key", how="left")
    rows = []
    for measure in [m for m in measure_columns(events) if m in details.columns]:
        e, d = kept[f"{measure}_{first}"], kept[f"{measure}_{second}"]
        pair = pd.concat([e, d], axis=1).dropna()
        p = float("nan")
        if len(pair) >= 10 and (pair.iloc[:, 0] != pair.iloc[:, 1]).any():
            p = float(stats.wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1]).pvalue)
        rows.append({
            "measure": measure, "n": len(pair), "rho_units": spearman(e, d),
            f"median_{first}": float(e.median()), f"median_{second}": float(d.median()), "wilcoxon_p": p,
            f"rho_age_{first}": spearman(e, kept["age_mid"]), f"rho_age_{second}": spearman(d, kept["age_mid"]),
        })
    overview = pd.DataFrame([{
        f"{first}_stories": len(events), f"{second}_stories": len(details), "in_both": len(both), "text_matched": len(kept),
        "dropped_text_mismatch": len(both) - len(kept),
        f"{first}_units_median": float(kept[f"n_details_{first}"].median()), f"{second}_units_median": float(kept[f"n_details_{second}"].median()),
    }])
    return pd.DataFrame(rows), overview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, required=True, help="analyze.py output folder of the event-unit run")
    parser.add_argument("--details", type=Path, required=True, help="analyze.py output folder of the detail-unit run")
    parser.add_argument("--batch", type=Path, required=True, help="batch folder with subset_stories.csv (age_lo, age_hi)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-ratio", type=float, default=1.25)
    args = parser.parse_args()
    subset = pd.read_csv(args.batch / "subset_stories.csv")
    ages = pd.DataFrame({"story_key": subset["story_key"], "age_mid": (subset["age_lo"] + subset["age_hi"]) / 2})
    table, overview = compare(
        pd.read_csv(args.events / "story_measures.csv"), pd.read_csv(args.details / "story_measures.csv"), ages, args.token_ratio
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "units_comparison.csv", index=False)
    overview.to_csv(args.output_dir / "units_overview.csv", index=False)
    print(overview.T.to_string(header=False))
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
