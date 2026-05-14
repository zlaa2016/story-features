#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _normalize_story_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float)):
        try:
            return str(int(value)).zfill(6)
        except Exception:
            return None
    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"\d{1,6}", s):
        return s.zfill(6)
    return None


def load_cleaned_filter(meta_path: Path, cleaned_col: str) -> tuple[set[str], set[str]]:
    meta = pd.read_csv(meta_path, low_memory=False)
    keep = pd.to_numeric(meta.get(cleaned_col), errors="coerce").fillna(0).astype(int) == 1
    meta = meta[keep].copy()
    story_ids = set()
    video_ids = set()
    if "story_id" in meta.columns:
        story_ids = set(meta["story_id"].map(_normalize_story_id).dropna().astype(str))
    if "video_id" in meta.columns:
        video_ids = set(meta["video_id"].astype(str).str.strip())
    return story_ids, video_ids


def filter_cleaned_df(df: pd.DataFrame, story_ids: set[str], video_ids: set[str]) -> pd.DataFrame:
    keep = pd.Series(False, index=df.index)
    if "story_id" in df.columns and story_ids:
        keep = keep | df["story_id"].map(_normalize_story_id).isin(story_ids)
    if "video_id" in df.columns and video_ids:
        keep = keep | df["video_id"].astype(str).str.strip().isin(video_ids)
    return df[keep].copy()


def parse_cols(spec: str) -> List[str]:
    return [c.strip() for c in spec.split(",") if c.strip()]


def parse_paths(spec: str) -> List[Path]:
    return [Path(s.strip()) for s in spec.split(",") if s.strip()]


def load_artifacts(vectors_path: Path, feature_paths: Sequence[Path]) -> tuple[np.ndarray, pd.DataFrame]:
    vec = np.load(vectors_path, allow_pickle=True)
    vectors = vec["vectors"]
    dfv = pd.DataFrame(
        {
            "vec_row": np.arange(vectors.shape[0], dtype=int),
            "item_id": vec["item_id"].astype(str),
            "video_id": vec["item_id"].astype(str),
            "story_id": [None if x is None else str(x) for x in vec["story_id"].tolist()],
            "age_mid_vec": vec["age_mid"].astype(float),
        }
    )

    merged = dfv.copy()
    for i, features_path in enumerate(feature_paths):
        feat = pd.read_csv(features_path, low_memory=False)
        feat["item_id"] = feat["item_id"].astype(str)
        if "story_id" in feat.columns:
            feat["story_id"] = feat["story_id"].map(_normalize_story_id)
        if "video_id" in feat.columns:
            feat["video_id"] = feat["video_id"].astype(str).str.strip()
        if "age_mid" in feat.columns:
            feat["age_mid"] = pd.to_numeric(feat["age_mid"], errors="coerce")
        if "aoa_mean" in feat.columns:
            feat["aoa_mean"] = pd.to_numeric(feat["aoa_mean"], errors="coerce")
        merged = merged.merge(feat, on="item_id", how="inner", suffixes=("", f"_feat{i}"))
        for base_col in ["story_id", "video_id", "age_mid"]:
            extra_col = f"{base_col}_feat{i}"
            if extra_col in merged.columns:
                merged[base_col] = merged[base_col].fillna(merged[extra_col])
                merged = merged.drop(columns=[extra_col])
    if "age_mid" not in merged.columns:
        merged["age_mid"] = merged["age_mid_vec"]
    merged["story_id"] = merged["story_id"].map(_normalize_story_id)
    return vectors, merged


def build_age_groups(df: pd.DataFrame, age_cutoff: float) -> pd.DataFrame:
    out = df.copy()
    out["age_mid"] = pd.to_numeric(out["age_mid"], errors="coerce")
    out = out[out["age_mid"].notna()].copy()
    out["age_label"] = np.where(out["age_mid"] <= age_cutoff, 0, 1)
    return out


def greedy_match(df: pd.DataFrame, match_cols: Sequence[str], seed: int) -> pd.DataFrame:
    cols = [c for c in match_cols if c in df.columns]
    if not cols:
        raise ValueError("No matching columns found.")
    work = df[["item_id", "age_label"] + list(cols)].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna().copy()
    if work["age_label"].nunique() != 2:
        raise ValueError("Need exactly two age groups for matching.")

    g0 = work[work["age_label"] == 0].copy()
    g1 = work[work["age_label"] == 1].copy()
    if len(g0) == 0 or len(g1) == 0:
        raise ValueError("Both age groups must be non-empty.")

    small, large = (g0, g1) if len(g0) <= len(g1) else (g1, g0)
    scaler = StandardScaler()
    X_small = scaler.fit_transform(small[cols])
    X_large = scaler.transform(large[cols])

    nn = NearestNeighbors(n_neighbors=min(20, len(large)), metric="euclidean")
    nn.fit(X_large)
    distances, indices = nn.kneighbors(X_small)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(small))
    used_large: set[int] = set()
    matched_small_idx: List[int] = []
    matched_large_idx: List[int] = []
    for row in order:
        chosen = None
        for cand in indices[row]:
            cand_i = int(cand)
            if cand_i not in used_large:
                chosen = cand_i
                break
        if chosen is None:
            continue
        used_large.add(chosen)
        matched_small_idx.append(row)
        matched_large_idx.append(chosen)

    small_keep = small.iloc[matched_small_idx][["item_id"]]
    large_keep = large.iloc[matched_large_idx][["item_id"]]
    keep_ids = set(pd.concat([small_keep, large_keep], ignore_index=True)["item_id"].tolist())
    return df[df["item_id"].isin(keep_ids)].copy()


def evaluate_layers(vectors: np.ndarray, df: pd.DataFrame, seeds: Sequence[int]) -> pd.DataFrame:
    keep_rows = df["vec_row"].astype(int).to_numpy()
    X = vectors[keep_rows]
    y = df["age_label"].to_numpy(dtype=np.int64)

    results: List[Dict[str, Any]] = []
    for layer in range(X.shape[1]):
        layer_acc: List[float] = []
        layer_f1: List[float] = []
        for seed in seeds:
            Xl = X[:, layer, :]
            X_train, X_test, y_train, y_test = train_test_split(
                Xl, y, test_size=0.25, random_state=int(seed), stratify=y
            )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            clf = LogisticRegression(max_iter=2000, n_jobs=1)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            layer_acc.append(float(accuracy_score(y_test, pred)))
            layer_f1.append(float(f1_score(y_test, pred, average="macro")))
        results.append(
            {
                "layer": layer,
                "n_items": int(len(df)),
                "mean_accuracy": float(np.mean(layer_acc)),
                "std_accuracy": float(np.std(layer_acc)),
                "mean_f1_macro": float(np.mean(layer_f1)),
                "std_f1_macro": float(np.std(layer_f1)),
            }
        )
    return pd.DataFrame(results)


def summarize_balance(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for c in cols:
        if c not in df.columns:
            continue
        d = df[["age_label", c]].copy()
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna()
        if d.empty:
            continue
        young = d.loc[d["age_label"] == 0, c].to_numpy(dtype=float)
        old = d.loc[d["age_label"] == 1, c].to_numpy(dtype=float)
        pooled = np.sqrt((young.var() + old.var()) / 2.0) if len(young) and len(old) else np.nan
        smd = float((old.mean() - young.mean()) / pooled) if pooled and pooled > 0 else float("nan")
        rows.append(
            {
                "feature": c,
                "young_mean": float(young.mean()),
                "old_mean": float(old.mean()),
                "standardized_mean_diff": smd,
            }
        )
    return pd.DataFrame(rows)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an <= 1e-12 or bn <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (an * bn))


def normalized_direction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = a - b
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def direction_comparison(vectors: np.ndarray, df: pd.DataFrame, aoa_col: str, low_q: float, high_q: float) -> pd.DataFrame:
    work = df.copy()
    work[aoa_col] = pd.to_numeric(work[aoa_col], errors="coerce")
    work = work[work[aoa_col].notna()].copy()
    lo = float(work[aoa_col].quantile(low_q))
    hi = float(work[aoa_col].quantile(high_q))
    work["aoa_group"] = np.where(work[aoa_col] <= lo, 0, np.where(work[aoa_col] >= hi, 1, np.nan))
    work = work[work["aoa_group"].notna()].copy()
    work["aoa_group"] = work["aoa_group"].astype(int)

    age_idx = df["vec_row"].astype(int).to_numpy()
    aoa_idx = work["vec_row"].astype(int).to_numpy()

    age_y = df["age_label"].to_numpy(dtype=int)
    aoa_y = work["aoa_group"].to_numpy(dtype=int)
    X_age = vectors[age_idx]
    X_aoa = vectors[aoa_idx]

    rows: List[Dict[str, Any]] = []
    for layer in range(vectors.shape[1]):
        age_old = X_age[age_y == 1, layer, :].mean(axis=0)
        age_young = X_age[age_y == 0, layer, :].mean(axis=0)
        v_age = normalized_direction(age_old, age_young)

        aoa_high = X_aoa[aoa_y == 1, layer, :].mean(axis=0)
        aoa_low = X_aoa[aoa_y == 0, layer, :].mean(axis=0)
        v_aoa = normalized_direction(aoa_high, aoa_low)

        age_proj_on_age = X_age[:, layer, :] @ v_age
        age_proj_on_aoa = X_age[:, layer, :] @ v_aoa
        aoa_proj_on_age = X_aoa[:, layer, :] @ v_age
        aoa_proj_on_aoa = X_aoa[:, layer, :] @ v_aoa

        rows.append(
            {
                "layer": layer,
                "cos_age_vs_aoa": cosine(v_age, v_aoa),
                "age_auc_on_v_age": float(roc_auc_score(age_y, age_proj_on_age)),
                "age_auc_on_v_aoa": float(roc_auc_score(age_y, age_proj_on_aoa)),
                "aoa_auc_on_v_age": float(roc_auc_score(aoa_y, aoa_proj_on_age)),
                "aoa_auc_on_v_aoa": float(roc_auc_score(aoa_y, aoa_proj_on_aoa)),
                "age_proj_corr_aoa_mean_on_v_age": float(np.corrcoef(age_proj_on_age, df[aoa_col].to_numpy(dtype=float))[0, 1]),
                "age_proj_corr_age_mid_on_v_aoa": float(np.corrcoef(age_proj_on_aoa, df["age_mid"].to_numpy(dtype=float))[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Age/AoA direction experiments on existing vector + feature artifacts.")
    ap.add_argument("--vectors", type=Path, default=Path("artifacts/distilbert_vectors/story_vectors.npz"))
    ap.add_argument(
        "--features",
        type=str,
        default="artifacts/text_features_batch1.csv,artifacts/text_features_batch1_review_samples.csv",
        help="Comma-separated feature CSVs to merge on item_id.",
    )
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/age_aoa_direction_experiments"))
    ap.add_argument("--cleaned-meta", type=Path, default=Path("story_wAmazonData_042626/amazon_youtube_batch1_wid.csv"))
    ap.add_argument("--cleaned-col", type=str, default="cleaned")
    ap.add_argument("--age-cutoff", type=float, default=5.0)
    ap.add_argument("--aoa-col", type=str, default="aoa_mean")
    ap.add_argument("--match-aoa-cols", type=str, default="aoa_mean,aoa_std,aoa_coverage")
    ap.add_argument(
        "--match-narrative-cols",
        type=str,
        default="num_sentences,num_words,mean_sentence_length_words,ttr,flesch_kincaid_grade_est",
    )
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--aoa-low-quantile", type=float, default=0.33)
    ap.add_argument("--aoa-high-quantile", type=float, default=0.67)
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    vectors, merged = load_artifacts(args.vectors, parse_paths(args.features))

    if args.cleaned_meta is not None and args.cleaned_meta.exists():
        story_ids, video_ids = load_cleaned_filter(args.cleaned_meta, args.cleaned_col)
        merged = filter_cleaned_df(merged, story_ids, video_ids)

    merged = build_age_groups(merged, args.age_cutoff)
    merged = merged.drop_duplicates(subset=["item_id"]).reset_index(drop=True)

    args.outdir.mkdir(parents=True, exist_ok=True)

    full_probe = evaluate_layers(vectors, merged, seeds)
    full_probe["dataset"] = "full_cleaned"
    full_probe.to_csv(args.outdir / "age_probe_full_cleaned.csv", index=False)

    aoa_cols = parse_cols(args.match_aoa_cols)
    narr_cols = parse_cols(args.match_narrative_cols)

    aoa_matched = greedy_match(merged, aoa_cols, seed=seeds[0])
    aoa_probe = evaluate_layers(vectors, aoa_matched, seeds)
    aoa_probe["dataset"] = "aoa_matched"
    aoa_probe.to_csv(args.outdir / "age_probe_aoa_matched.csv", index=False)
    summarize_balance(aoa_matched, aoa_cols + narr_cols).to_csv(args.outdir / "balance_aoa_matched.csv", index=False)

    narr_matched = greedy_match(merged, narr_cols, seed=seeds[0])
    narr_probe = evaluate_layers(vectors, narr_matched, seeds)
    narr_probe["dataset"] = "narrative_matched"
    narr_probe.to_csv(args.outdir / "age_probe_narrative_matched.csv", index=False)
    summarize_balance(narr_matched, aoa_cols + narr_cols).to_csv(args.outdir / "balance_narrative_matched.csv", index=False)

    direction_df = direction_comparison(
        vectors=vectors,
        df=merged,
        aoa_col=args.aoa_col,
        low_q=args.aoa_low_quantile,
        high_q=args.aoa_high_quantile,
    )
    direction_df.to_csv(args.outdir / "direction_comparison.csv", index=False)

    summary = {
        "n_full_cleaned": int(len(merged)),
        "n_aoa_matched": int(len(aoa_matched)),
        "n_narrative_matched": int(len(narr_matched)),
        "age_cutoff": float(args.age_cutoff),
        "aoa_col": args.aoa_col,
        "aoa_low_quantile": float(args.aoa_low_quantile),
        "aoa_high_quantile": float(args.aoa_high_quantile),
        "best_full_layer": int(full_probe.sort_values("mean_accuracy", ascending=False).iloc[0]["layer"]),
        "best_aoa_matched_layer": int(aoa_probe.sort_values("mean_accuracy", ascending=False).iloc[0]["layer"]),
        "best_narrative_matched_layer": int(narr_probe.sort_values("mean_accuracy", ascending=False).iloc[0]["layer"]),
        "best_direction_layer_by_cosine": int(direction_df["cos_age_vs_aoa"].abs().idxmax()),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.outdir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
