"""Does per-detail compression change with source age, and with the age a summary was written for?

Input is the span-search output (details.csv and summaries.csv) at one length penalty. Each story has a
younger, an actual and an older summary, so the data are nested: summary details within summaries within
a story. The analyses respect that:

  mixed models   random intercept per story, summaries as the repeated measurements
                 outcome ~ source age * summary type [+ story length]
                 (summary level for three outcomes; detail level with summary nested in story for span length)
  likelihood-ratio tests   age effect, age x summary-type interaction, story-length control (fitted by ML)
  rank-based tests         Spearman correlation of age with compression (story as unit, bootstrap interval and
                           permutation p-value), Friedman test of younger/actual/older with Holm-corrected
                           Wilcoxon follow-ups
  chance check             matched span score against the same search on another story's source

Compression outcomes are on a log2 scale: a slope of s per year multiplies the span by 2**s per year.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CONDITIONS = ["younger", "actual", "older"]
REF = "C(condition, Treatment('actual'))"
OUTCOMES = {
    "mean_log2_span": "log2 span length (source details per summary detail)",
    "mean_log2_word_ratio": "log2 word ratio (span words / summary-detail words)",
    "log2_compression": "log2 overall compression (source details / summary details)",
}


def prepare(details: pd.DataFrame, summaries: pd.DataFrame, penalty: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = details[np.isclose(details.length_penalty, penalty)].copy()
    s = summaries[np.isclose(summaries.length_penalty, penalty)].copy()
    d["log2_span"] = np.log2(d["span_len"])
    s = s.merge(d.groupby("record_id")["log2_span"].mean().rename("mean_log2_span"), left_on="record_id", right_index=True)
    s["log2_compression"] = np.log2(s["n_source_details"] / s["n_summary_details"])
    center = s.drop_duplicates("story_key")
    s["age_c"] = s["source_age"] - center["source_age"].mean()
    s["log2_n_c"] = np.log2(s["n_source_details"]) - np.log2(center["n_source_details"]).mean()
    d = d.merge(s[["record_id", "age_c", "log2_n_c"]], on="record_id")
    s["condition"] = pd.Categorical(s["condition"], CONDITIONS)
    d["condition"] = pd.Categorical(d["condition"], CONDITIONS)
    return d, s


def _fit(formula: str, data: pd.DataFrame, reml: bool, groups: str = "story_key", vc_formula: dict | None = None):
    import statsmodels.formula.api as smf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, data, groups=data[groups], vc_formula=vc_formula)
        return model.fit(reml=reml, method="lbfgs", maxiter=500)


def _slopes(result, label: str, outcome: str, with_interaction: bool = True) -> list[dict]:
    names = list(result.fe_params.index)
    cov = result.cov_params().loc[names, names]
    rows = []
    for condition in CONDITIONS:
        w = pd.Series(0.0, index=names)
        w["age_c"] = 1.0
        if with_interaction and condition != "actual":
            w[f"age_c:{REF}[T.{condition}]"] = 1.0
        estimate = float(w @ result.fe_params)
        se = float(np.sqrt(w.values @ cov.values @ w.values))
        p = 2 * (1 - stats.norm.cdf(abs(estimate / se)))
        rows.append({"model": label, "outcome": outcome, "summary": condition, "slope_per_year": estimate,
                     "ci_lo": estimate - 1.96 * se, "ci_hi": estimate + 1.96 * se, "p": p,
                     "pct_change_per_year": 100 * (2 ** estimate - 1)})
    return rows


def _lr(small, big, name: str, outcome: str) -> dict:
    chi2 = max(2 * (big.llf - small.llf), 0.0)
    df = len(big.fe_params) - len(small.fe_params)
    return {"outcome": outcome, "test": name, "chi2": chi2, "df": df, "p": float(stats.chi2.sf(chi2, df))}


def mixed_models(summary_table: pd.DataFrame, detail_table: pd.DataFrame | None = None):
    slopes, lr_rows, fixed_rows, icc_rows = [], [], [], []
    for outcome in OUTCOMES:
        none = f"{outcome} ~ {REF}"
        base = f"{outcome} ~ age_c + {REF}"
        full = f"{outcome} ~ age_c * {REF}"
        with_len = full + " + log2_n_c"
        ml = {k: _fit(f, summary_table, reml=False) for k, f in [("none", none), ("base", base), ("full", full), ("len", with_len)]}
        lr_rows += [_lr(ml["none"], ml["base"], "age effect (same slope for all summary types)", outcome),
                    _lr(ml["base"], ml["full"], "age x summary type interaction", outcome),
                    _lr(ml["full"], ml["len"], "story length (log2 details) adds to the model", outcome)]
        reml_full, reml_len = _fit(full, summary_table, reml=True), _fit(with_len, summary_table, reml=True)
        slopes += _slopes(reml_full, "summary level", outcome) + _slopes(reml_len, "summary level + story length", outcome)
        for label, res in [("summary level", reml_full), ("summary level + story length", reml_len)]:
            ci = res.conf_int().loc[res.fe_params.index]
            fixed_rows += [{"model": label, "outcome": outcome, "term": t, "estimate": res.fe_params[t], "ci_lo": ci.loc[t, 0],
                            "ci_hi": ci.loc[t, 1], "p": res.pvalues[t]} for t in res.fe_params.index]
        story_var = float(reml_full.cov_re.iloc[0, 0])
        icc_rows.append({"outcome": outcome, "story_variance": story_var, "residual_variance": float(reml_full.scale),
                         "share_of_variance_between_stories": story_var / (story_var + float(reml_full.scale))})
    if detail_table is not None:
        full = f"log2_span ~ age_c * {REF}"
        try:
            res = _fit(full, detail_table, reml=True, vc_formula={"summary": "0 + C(record_id)"})
            slopes += _slopes(res, "detail level, summary nested in story", "mean_log2_span")
        except Exception as error:   # fall back to summaries as the only grouping
            print(f"nested detail model failed ({type(error).__name__}); using summary as the grouping")
            res = _fit(full, detail_table, reml=True, groups="record_id")
            slopes += _slopes(res, "detail level, summary as group", "mean_log2_span")
    return pd.DataFrame(slopes), pd.DataFrame(lr_rows), pd.DataFrame(fixed_rows), pd.DataFrame(icc_rows)


def rank_tests(summary_table: pd.DataFrame, n_boot: int = 2000, n_perm: int = 10000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    wide = summary_table.pivot(index="story_key", columns="condition", values="mean_log2_span").dropna()
    age = summary_table.drop_duplicates("story_key").set_index("story_key").loc[wide.index, "source_age"].to_numpy()

    def spearman(x, y):
        return stats.spearmanr(x, y).statistic

    for label, y in [("all summary types, story mean", wide.mean(axis=1).to_numpy())] + [(c, wide[c].to_numpy()) for c in CONDITIONS]:
        rho = spearman(age, y)
        boots = [spearman(age[i], y[i]) for i in (rng.integers(0, len(age), len(age)) for _ in range(n_boot))]
        perm = np.array([spearman(rng.permutation(age), y) for _ in range(n_perm)])
        rows.append({"test": "Spearman, source age vs log2 span", "group": label, "n_stories": len(age), "statistic": rho,
                     "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
                     "p": (1 + (np.abs(perm) >= abs(rho)).sum()) / (n_perm + 1)})
    friedman = stats.friedmanchisquare(*[wide[c] for c in CONDITIONS])
    rows.append({"test": "Friedman, younger vs actual vs older", "group": "story is the block", "n_stories": len(wide),
                 "statistic": friedman.statistic, "p": friedman.pvalue})
    pairs = [("younger", "actual"), ("actual", "older"), ("younger", "older")]
    raw_p = [stats.wilcoxon(wide[a], wide[b]).pvalue for a, b in pairs]
    order = np.argsort(raw_p)
    holm, running = np.empty(len(pairs)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pairs) - rank) * raw_p[idx])
        holm[idx] = min(running, 1.0)
    for (a, b), p, ph in zip(pairs, raw_p, holm):
        rows.append({"test": "Wilcoxon signed-rank (Holm)", "group": f"{a} vs {b}", "n_stories": len(wide),
                     "statistic": float((wide[a] - wide[b]).median()), "p": ph, "note": f"raw p {p:.3g}; statistic = median paired difference in log2 span"})
    diff = (summary_table["mean_score"] - summary_table["mean_null_score"]).dropna()
    w = stats.wilcoxon(diff)
    rows.append({"test": "Wilcoxon, match score vs null score", "group": "all summaries", "n_stories": len(diff),
                 "statistic": float(diff.median()), "p": w.pvalue, "note": f"{(diff > 0).mean():.1%} of summaries score above their null; statistic = median difference"})
    return pd.DataFrame(rows)


def forest(slopes: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"younger": "#2a78d6", "actual": "#eb6834", "older": "#1baf7a"}
    data = slopes[slopes.outcome.eq("mean_log2_span")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(data) + 1.8))
    for i, row in data.iterrows():
        ax.plot([row.ci_lo, row.ci_hi], [i, i], color=colors[row.summary], lw=2.2)
        ax.plot(row.slope_per_year, i, "o", color=colors[row.summary], mfc="#fcfcfb", mew=1.8, ms=7)
    ax.axvline(0, color="#898781", lw=1, ls="--")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([f"{r.summary}  |  {r.model}" for r in data.itertuples()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Change in log2 span length per year of source age (95% CI)")
    ax.grid(axis="x", color="#e6e5e1")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("results/analysis/natural_language/detail_compression/span_search"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/natural_language/detail_compression/stats"))
    parser.add_argument("--penalty", type=float, default=0.0, help="Length penalty of the search to analyse (0 = none).")
    parser.add_argument("--no-detail-model", action="store_true", help="Skip the detail-level nested model (slowest).")
    args = parser.parse_args()
    details = pd.read_csv(args.input_dir / "details.csv")
    summaries = pd.read_csv(args.input_dir / "summaries.csv")
    d, s = prepare(details, summaries, args.penalty)
    print(f"{s.story_key.nunique()} stories, {len(s)} summaries, {len(d)} summary details, length penalty {args.penalty:g}")
    slopes, lr, fixed, icc = mixed_models(s, None if args.no_detail_model else d)
    rank = rank_tests(s)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in [("age_slopes", slopes), ("likelihood_ratio_tests", lr), ("mixed_model_fixed_effects", fixed),
                        ("variance_by_level", icc), ("rank_tests", rank)]:
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    forest(slopes, args.output_dir / "age_slopes_forest.png")
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\nAge slope per summary type (log2 span per year):\n", slopes[slopes.outcome.eq("mean_log2_span")].round(4).to_string(index=False))
    print("\nLikelihood-ratio tests:\n", lr.round(4).to_string(index=False))
    print("\nRank-based tests:\n", rank.round(4).to_string(index=False))
    print("\nVariance by level:\n", icc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
