#!/usr/bin/env python3
# LEGACY (event-segmented pipeline). This script works on the event-segmented review samples
# (review_events_json / has_events in the master table), not on the detail-segmented stories used by
# the current experiments (data_pipeline -> features/summaries -> analysis). See docs/data_lineage.md.
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize


WORD_RE = re.compile(r"[A-Za-z']+")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Compute corpus-level story AoA statistics from review_events_json, "
            "then derive age-group cutoffs with clustering and separability search."
        )
    )
    ap.add_argument(
        "--master",
        type=Path,
        default=None,
        help="Optional master dataset path. Defaults to the first available file under data/processed/.",
    )
    ap.add_argument(
        "--aoa-lexicon",
        type=Path,
        default=None,
        help="Optional AoA lexicon path. Defaults to the first available file under data/aoa/.",
    )
    ap.add_argument(
        "--amazon-youtube-meta",
        type=Path,
        default=None,
        help="Optional metadata CSV with cleaned/story_id/age fields, e.g. amazon_youtube_batch1_wid.csv.",
    )
    ap.add_argument(
        "--cleaned-text-dir",
        type=Path,
        default=None,
        help="Directory containing cleaned transcript/story .txt files named with zero-padded story_id prefixes.",
    )
    ap.add_argument(
        "--cleaned-col",
        type=str,
        default="cleaned",
        help="Column in --amazon-youtube-meta indicating which rows to keep.",
    )
    ap.add_argument(
        "--word-clusters",
        type=Path,
        default=Path("results/features/eda/concept_age_unified/word_clusters.csv"),
        help="Lexicon concept clusters used for separability search.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/features/eda/corpus_aoa_cutoffs"),
        help="Output directory for CSV summaries.",
    )
    ap.add_argument(
        "--stats",
        nargs="+",
        default=["aoa_mean_story", "aoa_median_story", "aoa_p90_story", "aoa_std_story", "aoa_max_story"],
        help="Story-level AoA statistics to evaluate.",
    )
    ap.add_argument(
        "--min-frac",
        type=float,
        default=0.15,
        help="Minimum fraction required in each group during cutoff search.",
    )
    ap.add_argument(
        "--k-min",
        type=int,
        default=2,
        help="Minimum number of clusters to evaluate for 1D KMeans/GMM.",
    )
    ap.add_argument(
        "--k-max",
        type=int,
        default=5,
        help="Maximum number of clusters to evaluate for 1D KMeans/GMM.",
    )
    return ap.parse_args()


def find_existing(candidates: list[Path]) -> Path:
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No file found among: {candidates}")
    return path


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".gz", ".csv"}:
        return pd.read_csv(path, low_memory=False)
    if path.suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def pick_aoa_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    word_candidates = ["lemma", "lemmas", "word", "words"]
    aoa_candidates = [
        "aoa",
        "aoa_mean",
        "mean_aoa",
        "aoa_kup_lem",
        "aoa_kup",
        "aoa_kup_lemmas",
        "aoa_kup_lemma",
    ]
    word_col = next((cols[c] for c in word_candidates if c in cols), None)
    aoa_col = next((cols[c] for c in aoa_candidates if c in cols), None)
    if word_col is None:
        raise ValueError(f"Could not infer AoA word column from: {list(df.columns)}")
    if aoa_col is None:
        raise ValueError(f"Could not infer AoA column from: {list(df.columns)}")
    return word_col, aoa_col


def safe_json_loads(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, (dict, list)):
        return x
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def iter_event_texts(review_events_json):
    events = safe_json_loads(review_events_json)
    if not isinstance(events, list):
        return
    for event in events:
        if not isinstance(event, dict):
            continue
        sentences = event.get("sentences")
        if not isinstance(sentences, list):
            continue
        parts: list[str] = []
        for sent in sentences:
            if isinstance(sent, dict) and "text" in sent:
                parts.append(str(sent["text"]))
            elif isinstance(sent, str):
                parts.append(sent)
        text = " ".join(p.strip() for p in parts if p and str(p).strip()).strip()
        if text:
            yield text


def lemma_heuristic(word: str) -> str:
    word = word.lower().strip("'")
    for suffix in ["ing", "ed", "ies", "s"]:
        if len(word) > 4 and word.endswith(suffix):
            if suffix == "ies":
                return word[:-3] + "y"
            if suffix == "s":
                return word[:-1]
            return word[: -len(suffix)]
    return word


def lemmatize(text: str) -> list[str]:
    toks = [t.lower() for t in WORD_RE.findall(text)]
    toks = [t for t in toks if len(t) >= 2]
    return [lemma_heuristic(t) for t in toks]


def build_story_aoa(
    stories: pd.DataFrame,
    aoa_map: dict[str, float],
    lemma_to_cluster: dict[str, int] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    story_rows: list[dict[str, float | int | str]] = []
    concept_rows: list[dict[str, float | int | str]] = []

    for row in stories.itertuples(index=False):
        story_id = getattr(row, "story_id", None)
        age_mid = float(getattr(row, "age_mid"))
        matched: list[float] = []
        total_tokens = 0
        concept_counts = Counter()

        for event_text in iter_event_texts(getattr(row, "review_events_json")):
            toks = lemmatize(event_text)
            total_tokens += len(toks)
            for tok in toks:
                aoa_val = aoa_map.get(tok)
                if aoa_val is not None and not np.isnan(aoa_val):
                    matched.append(float(aoa_val))
                if lemma_to_cluster is not None:
                    cluster = lemma_to_cluster.get(tok)
                    if cluster is not None:
                        concept_counts[int(cluster)] += 1

        if not matched:
            continue

        arr = np.asarray(matched, dtype=float)
        story_rows.append(
            {
                "story_id": story_id,
                "age_mid": age_mid,
                "aoa_mean_story": float(arr.mean()),
                "aoa_median_story": float(np.median(arr)),
                "aoa_std_story": float(arr.std(ddof=0)),
                "aoa_p90_story": float(np.quantile(arr, 0.90)),
                "aoa_max_story": float(arr.max()),
                "aoa_n_matched": int(len(arr)),
                "aoa_total_tokens": int(total_tokens),
                "aoa_coverage_story": float(len(arr) / max(1, total_tokens)),
            }
        )

        if concept_counts:
            for cluster, n_tokens in concept_counts.items():
                concept_rows.append(
                    {
                        "story_id": story_id,
                        "cluster": int(cluster),
                        "n_tokens": int(n_tokens),
                    }
                )

    return pd.DataFrame(story_rows), pd.DataFrame(concept_rows)


def build_story_aoa_from_texts(
    meta: pd.DataFrame,
    text_dir: Path,
    aoa_map: dict[str, float],
    lemma_to_cluster: dict[str, int] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    text_files = {}
    for path in text_dir.glob("*.txt"):
        prefix = path.name.split("_", 1)[0]
        if prefix.isdigit():
            text_files[int(prefix)] = path

    story_rows: list[dict[str, float | int | str]] = []
    concept_rows: list[dict[str, float | int | str]] = []
    missing = 0

    for row in meta.itertuples(index=False):
        story_id = getattr(row, "story_id_num")
        if pd.isna(story_id):
            continue
        story_id = int(story_id)
        text_path = text_files.get(story_id)
        if text_path is None:
            missing += 1
            continue
        text = text_path.read_text(errors="ignore")
        toks = lemmatize(text)
        if not toks:
            continue

        matched = [float(aoa_map[t]) for t in toks if t in aoa_map and not np.isnan(aoa_map[t])]
        if not matched:
            continue

        concept_counts = Counter()
        if lemma_to_cluster is not None:
            for tok in toks:
                cluster = lemma_to_cluster.get(tok)
                if cluster is not None:
                    concept_counts[int(cluster)] += 1

        arr = np.asarray(matched, dtype=float)
        story_rows.append(
            {
                "story_id": story_id,
                "video_id": getattr(row, "video_id", None),
                "age_lo": pd.to_numeric(getattr(row, "age_lo", np.nan), errors="coerce"),
                "age_hi": pd.to_numeric(getattr(row, "age_hi", np.nan), errors="coerce"),
                "age_mid": pd.to_numeric(getattr(row, "age_mid", np.nan), errors="coerce"),
                "aoa_mean_story": float(arr.mean()),
                "aoa_median_story": float(np.median(arr)),
                "aoa_std_story": float(arr.std(ddof=0)),
                "aoa_p90_story": float(np.quantile(arr, 0.90)),
                "aoa_max_story": float(arr.max()),
                "aoa_n_matched": int(len(arr)),
                "aoa_total_tokens": int(len(toks)),
                "aoa_coverage_story": float(len(arr) / max(1, len(toks))),
                "text_path": str(text_path),
            }
        )

        if concept_counts:
            for cluster, n_tokens in concept_counts.items():
                concept_rows.append(
                    {
                        "story_id": story_id,
                        "cluster": int(cluster),
                        "n_tokens": int(n_tokens),
                    }
                )

    if missing:
        print(f"Warning: {missing} cleaned rows had no matching text file in {text_dir}")
    return pd.DataFrame(story_rows), pd.DataFrame(concept_rows)


def centroid_sep(X: np.ndarray, mask_a: np.ndarray) -> float:
    A = X[mask_a]
    B = X[~mask_a]
    if len(A) == 0 or len(B) == 0:
        return -1.0
    return float(np.linalg.norm(A.mean(axis=0) - B.mean(axis=0)))


def best_2bin_cutoff(X: np.ndarray, y: np.ndarray, min_frac: float, n_grid: int = 140) -> tuple[float, float | None]:
    lo, hi = np.quantile(y, [min_frac, 1 - min_frac])
    thresholds = np.linspace(lo, hi, n_grid)
    best = (-1.0, None)
    for threshold in thresholds:
        mask = y <= threshold
        if min(mask.mean(), 1 - mask.mean()) < min_frac:
            continue
        score = centroid_sep(X, mask)
        if score > best[0]:
            best = (score, float(threshold))
    return best


def between_scatter_objective(X: np.ndarray, y: np.ndarray, t1: float, t2: float) -> float:
    groups = [X[y <= t1], X[(y > t1) & (y <= t2)], X[y > t2]]
    if any(len(g) == 0 for g in groups):
        return -1.0
    mus = [g.mean(axis=0) for g in groups]
    ns = [len(g) for g in groups]
    grand = sum(n * mu for n, mu in zip(ns, mus)) / float(sum(ns))
    return float(sum(n * np.dot(mu - grand, mu - grand) for n, mu in zip(ns, mus)))


def best_3bin_cutoffs(
    X: np.ndarray,
    y: np.ndarray,
    min_frac: float,
    grid_n: int = 55,
) -> tuple[float, float | None, float | None]:
    quantiles = np.linspace(min_frac, 1 - min_frac, grid_n)
    candidates = np.quantile(y, quantiles)
    best = (-1.0, None, None)
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            t1 = float(candidates[i])
            t2 = float(candidates[j])
            f1 = np.mean(y <= t1)
            f2 = np.mean((y > t1) & (y <= t2))
            f3 = np.mean(y > t2)
            if min(f1, f2, f3) < min_frac:
                continue
            score = between_scatter_objective(X, y, t1, t2)
            if score > best[0]:
                best = (score, t1, t2)
    return best


def evaluate_1d_clusters(values: np.ndarray, k_min: int, k_max: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    kmeans_rows: list[dict[str, float | int | str]] = []
    gmm_rows: list[dict[str, float | int | str]] = []
    X = values.reshape(-1, 1)

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=0, n_init=20)
        km_labels = km.fit_predict(X)
        km_centers = np.sort(km.cluster_centers_.ravel())
        km_cuts = (km_centers[:-1] + km_centers[1:]) / 2.0
        kmeans_rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(X, km_labels)),
                "centers": "|".join(f"{x:.6f}" for x in km_centers),
                "cutoffs": "|".join(f"{x:.6f}" for x in km_cuts),
            }
        )

        gmm = GaussianMixture(n_components=k, random_state=0, n_init=10)
        gmm.fit(X)
        gmm_labels = gmm.predict(X)
        gmm_means = np.sort(gmm.means_.ravel())
        gmm_cuts = (gmm_means[:-1] + gmm_means[1:]) / 2.0
        gmm_rows.append(
            {
                "k": k,
                "bic": float(gmm.bic(X)),
                "aic": float(gmm.aic(X)),
                "silhouette": float(silhouette_score(X, gmm_labels)),
                "means": "|".join(f"{x:.6f}" for x in gmm_means),
                "cutoffs": "|".join(f"{x:.6f}" for x in gmm_cuts),
            }
        )

    return pd.DataFrame(kmeans_rows), pd.DataFrame(gmm_rows)


def main() -> None:
    args = parse_args()

    master_path = None
    stories = None
    meta_path = args.amazon_youtube_meta
    text_dir = args.cleaned_text_dir
    if meta_path is None and text_dir is None:
        master_path = args.master or find_existing(
            [
                Path("data/processed/master_batch1.csv.gz"),
                Path("data/processed/master_batch1.parquet"),
                Path("data/processed/master_stories.csv.gz"),
                Path("data/processed/master_stories.parquet"),
            ]
        )

    if meta_path is not None or text_dir is not None:
        if meta_path is None or text_dir is None:
            raise ValueError("Use --amazon-youtube-meta and --cleaned-text-dir together.")

    aoa_path = args.aoa_lexicon or find_existing(
        [
            Path("data/aoa/AoA_51715_words.xlsx"),
            Path("data/aoa/AoA_ratings_Kuperman_et_al_BRM_with_PoS.xlsx"),
        ]
    )

    if master_path is not None:
        master = read_table(master_path)
        master["age_mid"] = pd.to_numeric(master.get("age_mid"), errors="coerce")
        stories = master[(master.get("review_events_json").notna()) & (master["age_mid"].notna())].copy()
    else:
        meta = pd.read_csv(meta_path, low_memory=False)
        meta["story_id_num"] = pd.to_numeric(meta.get("story_id"), errors="coerce").astype("Int64")
        meta["age_lo"] = pd.to_numeric(meta.get("age_lo"), errors="coerce")
        meta["age_hi"] = pd.to_numeric(meta.get("age_hi"), errors="coerce")
        if "age_mid" not in meta.columns:
            meta["age_mid"] = (meta["age_lo"] + meta["age_hi"]) / 2.0
        else:
            meta["age_mid"] = pd.to_numeric(meta.get("age_mid"), errors="coerce")

        cleaned_vals = pd.to_numeric(meta.get(args.cleaned_col), errors="coerce").fillna(0).astype(int)
        meta = meta[cleaned_vals == 1].copy()
        meta = meta[meta["story_id_num"].notna()].copy()
        stories = meta

    aoa_df = read_table(aoa_path)
    word_col, aoa_col = pick_aoa_columns(aoa_df)
    aoa_map = {
        str(word).strip().lower(): float(aoa)
        for word, aoa in zip(aoa_df[word_col], aoa_df[aoa_col])
        if pd.notna(word) and pd.notna(aoa)
    }

    lemma_to_cluster = None
    if args.word_clusters.exists():
        word_clusters = pd.read_csv(args.word_clusters)
        lemma_to_cluster = dict(
            zip(
                word_clusters["lemma"].astype(str).str.lower(),
                word_clusters["cluster"].astype(int),
            )
        )

    if master_path is not None:
        story_aoa, story_concepts = build_story_aoa(stories, aoa_map, lemma_to_cluster)
    else:
        story_aoa, story_concepts = build_story_aoa_from_texts(stories, text_dir, aoa_map, lemma_to_cluster)
    if story_aoa.empty:
        raise RuntimeError("No stories with AoA coverage were found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    story_aoa.to_csv(args.out_dir / "story_aoa_stats.csv", index=False)
    if not story_concepts.empty:
        story_concepts.to_csv(args.out_dir / "story_concept_token_counts.csv", index=False)

    summary_rows: list[dict[str, float | int | str]] = []
    kmeans_tables: list[pd.DataFrame] = []
    gmm_tables: list[pd.DataFrame] = []
    separability_rows: list[dict[str, float | str | None]] = []

    concept_wide = None
    if not story_concepts.empty:
        concept_wide = story_concepts.pivot_table(
            index="story_id",
            columns="cluster",
            values="n_tokens",
            fill_value=0,
        )

    for stat in args.stats:
        if stat not in story_aoa.columns:
            continue
        d = story_aoa[["story_id", "age_mid", stat]].dropna().copy()
        values = d[stat].to_numpy(dtype=float)
        summary_rows.append(
            {
                "stat": stat,
                "n_stories": int(len(d)),
                "min": float(np.min(values)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(np.max(values)),
                "corr_with_age_mid": float(np.corrcoef(d[stat], d["age_mid"])[0, 1]),
            }
        )

        km_df, gmm_df = evaluate_1d_clusters(values, args.k_min, args.k_max)
        km_df.insert(0, "stat", stat)
        gmm_df.insert(0, "stat", stat)
        kmeans_tables.append(km_df)
        gmm_tables.append(gmm_df)

        if concept_wide is not None:
            aligned = concept_wide.reindex(d["story_id"]).fillna(0.0)
            X = normalize(aligned.to_numpy(dtype=float), norm="l1")
            score2, cutoff2 = best_2bin_cutoff(X, values, min_frac=args.min_frac)
            score3, cutoff31, cutoff32 = best_3bin_cutoffs(X, values, min_frac=args.min_frac)
            separability_rows.append(
                {
                    "stat": stat,
                    "two_bin_score": score2,
                    "two_bin_cutoff": cutoff2,
                    "three_bin_score": score3,
                    "three_bin_cutoff_1": cutoff31,
                    "three_bin_cutoff_2": cutoff32,
                }
            )

    pd.DataFrame(summary_rows).to_csv(args.out_dir / "story_aoa_summary.csv", index=False)
    if kmeans_tables:
        pd.concat(kmeans_tables, ignore_index=True).to_csv(args.out_dir / "kmeans_cutoffs.csv", index=False)
    if gmm_tables:
        pd.concat(gmm_tables, ignore_index=True).to_csv(args.out_dir / "gmm_cutoffs.csv", index=False)
    if separability_rows:
        pd.DataFrame(separability_rows).to_csv(args.out_dir / "separability_cutoffs.csv", index=False)

    if master_path is not None:
        print(f"Master: {master_path}")
        print(f"Stories with event text + age_mid: {len(stories)}")
    else:
        print(f"Metadata: {meta_path}")
        print(f"Cleaned text dir: {text_dir}")
        print(f"Rows with {args.cleaned_col} = 1: {len(stories)}")
    print(f"AoA lexicon: {aoa_path}")
    print(f"Stories with AoA coverage: {len(story_aoa)}")
    print(f"Outputs written to: {args.out_dir}")

    if summary_rows:
        summary = pd.DataFrame(summary_rows).sort_values("corr_with_age_mid", ascending=False)
        print("\nTop story-level AoA stats by correlation with age_mid:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
