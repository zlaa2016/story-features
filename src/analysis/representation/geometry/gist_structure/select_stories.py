"""Select a popular, representative, topic-matched set of stories for each age band.

Why this step exists.  Age is confounded with what stories are about, how long they are, which
channel read them and which author wrote them.  Selecting stories so that bands are the same size,
cover each band's topic space, favour stories children actually meet (popularity), and share the
same topic mix removes the most obvious ways for an age effect to be a selection artefact.

For each band:

1. every story is assigned to the band that contains most of its stated age range (stories whose
   range is mostly outside every band, e.g. "2-12", are left out and handled as a sensitivity set);
2. stories are clustered by the mean of their detail vectors (topic clusters) and each band gets the
   same topic mix (``--topic-quota match``);
3. inside each topic quota, stories are chosen by greedy weighted facility location on the story
   mean vectors (cover the band's topic space) with weights from popularity (``popular_coverage``),
   subject to one story per author group and a channel cap.

``--objective random`` gives the control selection to show a result does not depend on the choice.
``--mode all`` is the sensitivity set that keeps stories whose stated range spans several bands.
"""
from __future__ import annotations

import argparse
import importlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.io import write_csv, write_json
from shared.research import write_run_provenance

base = importlib.import_module("analysis.representation.geometry.within_story_compression.analyze")

DEFAULT_ROOT = "results/analysis/representation/geometry/gist_structure"
POPULARITY_COLUMNS = ("yt_view_count", "Num Reviews", "yt_like_count")


def parse_bands(spec: str) -> list[tuple[str, int, int]]:
    bands = []
    for part in spec.split(","):
        low, high = part.strip().split("-")
        bands.append((part.strip(), int(low), int(high)))
    return bands


def overlap_fractions(age_lo: np.ndarray, age_hi: np.ndarray, bands: list[tuple[str, int, int]]) -> np.ndarray:
    """Share of each story's stated age range (inclusive years) that falls inside each band."""
    width = (age_hi - age_lo + 1).astype(float)
    columns = []
    for _, low, high in bands:
        overlap = np.maximum(0, np.minimum(age_hi, high) + 1 - np.maximum(age_lo, low))
        columns.append(overlap / width)
    return np.column_stack(columns)


def assign_primary_band(age_lo: np.ndarray, age_hi: np.ndarray, bands: list[tuple[str, int, int]], min_overlap: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    fractions = overlap_fractions(age_lo, age_hi, bands)
    jittered = fractions + rng.random(fractions.shape) * 1e-6
    chosen = jittered.argmax(axis=1)
    best = fractions[np.arange(len(chosen)), chosen]
    chosen = np.where(best >= min_overlap, chosen, -1)
    return chosen, best


def popularity_scores(table: pd.DataFrame) -> pd.Series:
    """Mean percentile rank of log views, log reviews and log likes (0.5 when none is known)."""
    ranks = []
    for column in POPULARITY_COLUMNS:
        if column in table.columns:
            values = np.log1p(pd.to_numeric(table[column], errors="coerce").clip(lower=0))
            if values.notna().sum() > 5:
                ranks.append(values.rank(pct=True))
    if not ranks:
        return pd.Series(0.5, index=table.index)
    return pd.concat(ranks, axis=1).mean(axis=1).fillna(0.5)


def normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(value).lower()).strip()


def author_groups(table: pd.DataFrame) -> pd.Series:
    """Author when known, otherwise the first two words of the title (series share them), else the story key."""
    groups = []
    for _, row in table.iterrows():
        author = normalise_name(row.get("Author", ""))
        title = normalise_name(row.get("Title", row.get("video_title", "")))
        if author and author not in ("nan", "none", "unknown"):
            groups.append(author)
        elif title:
            groups.append(" ".join(title.split()[:2]))
        else:
            groups.append(str(row["story_key"]))
    return pd.Series(groups, index=table.index)


def topic_clusters(story_means: np.ndarray, clusters: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    centred = story_means - story_means.mean(axis=0)
    components = min(50, len(centred) - 1, centred.shape[1])
    reduced = PCA(n_components=components, random_state=seed).fit_transform(centred)
    clusters = max(2, min(clusters, len(reduced) // 10))
    return KMeans(n_clusters=clusters, n_init=5, random_state=seed).fit_predict(reduced)


def allocate_quotas(available: np.ndarray, target: int, proportions: np.ndarray) -> np.ndarray:
    """Topic quotas for one band: as close to ``target * proportions`` as the available counts allow."""
    available = np.asarray(available, dtype=int)
    quota = np.minimum(np.floor(target * proportions).astype(int), available)
    while quota.sum() < target:
        spare = available - quota
        if spare.sum() <= 0:
            break
        deficit = target * proportions - quota
        deficit[spare <= 0] = -np.inf
        quota[int(np.argmax(deficit))] += 1
    return quota


def facility_location(
    similarity: np.ndarray, weights: np.ndarray, count: int, blocked: np.ndarray, author_ids: np.ndarray,
    author_cap: int, channel_ids: np.ndarray, channel_cap: int, already_authors: dict, already_channels: dict,
) -> list[int]:
    """Greedy weighted facility location on a [0, 1] similarity; respects author and channel caps."""
    coverage = np.zeros(len(weights))
    chosen: list[int] = []
    for _ in range(count):
        gain = (weights[None, :] * np.maximum(0.0, similarity - coverage[None, :])).sum(axis=1)
        gain[blocked] = -np.inf
        for index in np.flatnonzero(np.isfinite(gain)):
            if already_authors.get(author_ids[index], 0) >= author_cap or already_channels.get(channel_ids[index], 0) >= channel_cap:
                gain[index] = -np.inf
        if not np.isfinite(gain).any():
            break
        best = int(np.argmax(gain))
        chosen.append(best)
        blocked[best] = True
        coverage = np.maximum(coverage, similarity[best])
        already_authors[author_ids[best]] = already_authors.get(author_ids[best], 0) + 1
        already_channels[channel_ids[best]] = already_channels.get(channel_ids[best], 0) + 1
    return chosen


def pick(frame: pd.DataFrame, similarity: np.ndarray, count: int, objective: str, rng: np.random.Generator, author_cap: int, channel_cap: int, authors: dict, channels: dict) -> list[int]:
    """Row positions (within ``frame``) chosen for one quota."""
    if count <= 0 or len(frame) == 0:
        return []
    author_ids = frame["author_group"].astype(str).to_numpy()
    channel_ids = frame["channel"].astype(str).to_numpy()
    blocked = np.zeros(len(frame), dtype=bool)
    if objective in ("popular_coverage", "coverage"):
        weights = np.ones(len(frame)) if objective == "coverage" else 0.25 + 0.75 * frame["popularity"].to_numpy(dtype=float)
        return facility_location(similarity, weights, count, blocked, author_ids, author_cap, channel_ids, channel_cap, authors, channels)
    order = np.argsort(-frame["popularity"].to_numpy(dtype=float), kind="stable") if objective == "popular" else rng.permutation(len(frame))
    chosen: list[int] = []
    for index in order:
        if len(chosen) >= count:
            break
        if authors.get(author_ids[index], 0) >= author_cap or channels.get(channel_ids[index], 0) >= channel_cap:
            continue
        chosen.append(int(index))
        authors[author_ids[index]] = authors.get(author_ids[index], 0) + 1
        channels[channel_ids[index]] = channels.get(channel_ids[index], 0) + 1
    return chosen


def select_stories(
    candidates: pd.DataFrame, story_means: np.ndarray, bands: list[tuple[str, int, int]], per_band: int, objective: str,
    topic_quota: str, author_cap: int, channel_share: float, equalize: bool, rng: np.random.Generator,
) -> pd.DataFrame:
    """Return ``candidates`` with a boolean ``selected`` column (band and topic_cluster already present)."""
    candidates = candidates.copy()
    candidates["selected"] = False
    counts = candidates[candidates["band_index"] >= 0].groupby("band_index").size()
    sizes = [int(counts.get(index, 0)) for index in range(len(bands))]
    target = per_band if not equalize else min(per_band, min(size for size in sizes if size > 0))
    clusters = sorted(candidates["topic_cluster"].unique())
    pooled = candidates[candidates["band_index"] >= 0]["topic_cluster"].value_counts(normalize=True)
    proportions = np.array([float(pooled.get(cluster, 0.0)) for cluster in clusters])
    proportions = proportions / proportions.sum()
    for band_index in range(len(bands)):
        in_band = candidates[candidates["band_index"] == band_index]
        if in_band.empty:
            continue
        goal = min(target, len(in_band))
        channel_cap = max(1, int(np.ceil(channel_share * goal)))
        authors: dict[str, int] = {}
        channels: dict[str, int] = {}
        chosen_index: list[Any] = []
        if topic_quota == "match":
            available = np.array([(in_band["topic_cluster"] == cluster).sum() for cluster in clusters])
            quotas = allocate_quotas(available, goal, proportions)
            for cluster, quota in zip(clusters, quotas):
                members = in_band[in_band["topic_cluster"] == cluster]
                positions = np.array([candidates.index.get_loc(index) for index in members.index])
                similarity = (story_means[positions] @ story_means[positions].T + 1.0) / 2.0
                chosen = pick(members, similarity, int(quota), objective, rng, author_cap, channel_cap, authors, channels)
                chosen_index.extend(members.index[chosen])
        remaining = goal - len(chosen_index)
        if remaining > 0:
            leftover = in_band.loc[~in_band.index.isin(chosen_index)]
            if not leftover.empty:
                positions = np.array([candidates.index.get_loc(index) for index in leftover.index])
                similarity = (story_means[positions] @ story_means[positions].T + 1.0) / 2.0
                chosen = pick(leftover, similarity, remaining, objective, rng, author_cap, channel_cap, authors, channels)
                chosen_index.extend(leftover.index[chosen])
        candidates.loc[chosen_index, "selected"] = True
    return candidates


def keep_all(stories: pd.DataFrame, bands: list[tuple[str, int, int]], author_cap: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sensitivity set: every story with an age range, placed in the band nearest its range midpoint.

    ``band_mid`` is the story's own range midpoint (so wide-range stories keep their ambiguity as a
    covariate, ``range_width``), and authors are capped so a prolific author cannot dominate.
    """
    result = stories.copy()
    middle = (result["age_lo"].to_numpy(dtype=float) + result["age_hi"].to_numpy(dtype=float)) / 2.0
    centres = np.array([(low + high) / 2.0 for _, low, high in bands])
    nearest = np.abs(middle[:, None] - centres[None, :]).argmin(axis=1)
    result["band_index"] = nearest
    result["band"] = [bands[i][0] for i in nearest]
    result["band_mid"] = middle
    keep = np.zeros(len(result), dtype=bool)
    order = rng.permutation(len(result))
    seen: dict[str, int] = {}
    for position in order:
        group = str(result["author_group"].iloc[position])
        if seen.get(group, 0) < author_cap:
            keep[position] = True
            seen[group] = seen.get(group, 0) + 1
    result["selected"] = keep
    return result


def balance_report(selected: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    from scipy.stats import chi2_contingency, f_oneway

    rows = []
    ordered = selected.sort_values("band_mid")
    youngest, oldest = ordered["band"].iloc[0], ordered["band"].iloc[-1]
    for column in columns:
        if column not in selected.columns:
            continue
        values = selected[["band", column]].dropna()
        groups = [group[column].to_numpy(dtype=float) for _, group in values.groupby("band") if len(group) > 1]
        if len(groups) < 2:
            continue
        young = values[values["band"] == youngest][column].to_numpy(dtype=float)
        old = values[values["band"] == oldest][column].to_numpy(dtype=float)
        pooled = np.sqrt((young.var(ddof=1) + old.var(ddof=1)) / 2.0) if len(young) > 1 and len(old) > 1 else np.nan
        smd = float((old.mean() - young.mean()) / pooled) if pooled and np.isfinite(pooled) and pooled > 0 else np.nan
        rows.append({"variable": column, **{f"mean_{name}": float(group[column].mean()) for name, group in values.groupby("band")},
                     "smd_oldest_vs_youngest": smd, "anova_p": float(f_oneway(*groups).pvalue), "imbalanced": bool(abs(smd) > 0.25) if np.isfinite(smd) else False})
    table = pd.crosstab(selected["band"], selected["topic_cluster"])
    if table.shape[0] > 1 and table.shape[1] > 1:
        rows.append({"variable": "topic_cluster_mix", "anova_p": float(chi2_contingency(table)[1]), "imbalanced": bool(chi2_contingency(table)[1] < 0.05)})
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isolated-dir", type=Path, default=Path(f"{base.DEFAULT_ROOT}/gpt_oss_20b_detail_controls"))
    parser.add_argument("--master-csv", type=Path, default=Path("data/processed/master_stories.csv.gz"))
    parser.add_argument("--subset-csv", type=Path, default=None, help="video_id, range_lower, range_upper (e.g. flat_videos_D_2-12.csv)")
    parser.add_argument("--bands", default="2-3,4-5,6-7,8-12", help="age bands; two-year bands leave 8-9 nearly empty in the master table")
    parser.add_argument("--per-band", type=int, default=100)
    parser.add_argument("--no-equalize", dest="equalize", action="store_false")
    parser.add_argument("--objective", choices=["popular_coverage", "coverage", "popular", "random"], default="popular_coverage")
    parser.add_argument("--mode", choices=["matched", "all"], default="matched",
                        help="matched: balanced, topic-matched bands (stories mostly outside every band are left out); "
                             "all: keep every story with an age range, banded by its range midpoint, authors capped (sensitivity)")
    parser.add_argument("--all-author-cap", type=int, default=3, help="mode all: maximum stories kept per author group")
    parser.add_argument("--topic-quota", choices=["match", "none"], default="match")
    parser.add_argument("--topic-clusters", type=int, default=8)
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--min-units", type=int, default=30)
    parser.add_argument("--author-cap", type=int, default=1)
    parser.add_argument("--channel-share", type=float, default=0.2)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--vector-key", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(f"{DEFAULT_ROOT}/selection"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    bands = parse_bands(args.bands)

    details = base.load_metadata(args.isolated_dir, "isolated")
    stories = base.build_story_table(details, args.subset_csv, min(b[1] for b in bands), max(b[2] for b in bands), args.min_units)
    details = details[details["story_key"].isin(stories["story_key"])].sort_values(["story_key", "detail_index"]).reset_index(drop=True)
    vectors = base.normalize_rows(base.load_vectors(args.isolated_dir, "isolated", details, args.layer, args.vector_key))
    keys = details["story_key"].to_numpy()
    edges = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    means = {keys[a]: vectors[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])}
    stories = stories[stories["story_key"].isin(means)].reset_index(drop=True)
    story_means = base.normalize_rows(np.stack([means[key] for key in stories["story_key"]]))

    if args.master_csv.exists():
        master = pd.read_csv(args.master_csv, low_memory=False)
        keep = ["video_id", *[c for c in (*POPULARITY_COLUMNS, "Author", "Title", "yt_channel", "Categories") if c in master.columns]]
        master = master[keep].drop_duplicates("video_id")
        master["video_id"] = master["video_id"].astype(str)
        stories["video_id"] = stories["video_id"].astype(str)
        stories = stories.merge(master, on="video_id", how="left")
    stories["popularity"] = popularity_scores(stories)
    stories["author_group"] = author_groups(stories)
    stories["channel"] = stories["yt_channel"].fillna("unknown") if "yt_channel" in stories.columns else "unknown"

    chosen, overlap = assign_primary_band(stories["age_lo"].to_numpy(), stories["age_hi"].to_numpy(), bands, args.min_overlap, rng)
    stories["band_index"] = chosen
    stories["overlap_fraction"] = overlap
    stories["band"] = [bands[i][0] if i >= 0 else "unassigned" for i in chosen]
    stories["band_mid"] = [(bands[i][1] + bands[i][2]) / 2.0 if i >= 0 else np.nan for i in chosen]
    stories["topic_cluster"] = topic_clusters(story_means, args.topic_clusters, args.seed)
    stories = stories.merge(base.story_covariates(details), on="story_key", how="left")  # left merge keeps row order = story_means order
    print(f"candidates {len(stories)}, assigned {(stories['band_index'] >= 0).sum()}: "
          f"{stories[stories['band_index'] >= 0].groupby('band').size().to_dict()}", flush=True)

    if args.mode == "all":
        result = keep_all(stories, bands, args.all_author_cap, rng)
    else:
        result = select_stories(stories, story_means, bands, args.per_band, args.objective, args.topic_quota,
                                args.author_cap, args.channel_share, args.equalize, rng)
    selected = result[result["selected"]].copy()
    print(f"selected {len(selected)}: {selected.groupby('band').size().to_dict()}", flush=True)
    write_csv(args.output_dir / "candidate_stories.csv", result.drop(columns=[c for c in ("yt_description",) if c in result.columns]))
    write_csv(args.output_dir / "selected_stories.csv", selected)
    report = balance_report(selected, ["n_details", "mean_detail_tokens", "lex_ttr", "mean_word_len", "popularity", "overlap_fraction"])
    write_csv(args.output_dir / "balance_report.csv", report)
    write_json(args.output_dir / "selection_summary.json", {
        "objective": args.objective, "topic_quota": args.topic_quota, "bands": [b[0] for b in bands],
        "candidates": int(len(stories)), "assigned": int((stories["band_index"] >= 0).sum()),
        "selected_per_band": {k: int(v) for k, v in selected.groupby("band").size().items()},
        "unassigned_wide_range": int((stories["band_index"] < 0).sum()),
        "imbalanced_variables": report[report["imbalanced"]]["variable"].tolist() if "imbalanced" in report else [],
    })
    write_run_provenance(args.output_dir, args)


if __name__ == "__main__":
    main()
