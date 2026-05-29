#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def normalized_direction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = a - b
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def parse_paths(spec: str) -> List[Path]:
    return [Path(s.strip()) for s in spec.split(",") if s.strip()]


def parse_floats(spec: str) -> List[float]:
    return [float(s.strip()) for s in spec.split(",") if s.strip()]


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


def merge_story_level_aoa_stats(base: pd.DataFrame, aoa_stats_path: Optional[Path]) -> pd.DataFrame:
    if aoa_stats_path is None or not aoa_stats_path.exists():
        return base
    stats = pd.read_csv(aoa_stats_path, low_memory=False)
    if "story_id" in stats.columns:
        stats["story_id"] = stats["story_id"].map(_normalize_story_id)
    if "video_id" in stats.columns:
        stats["video_id"] = stats["video_id"].astype(str).str.strip()
    keep_cols = [
        c
        for c in [
            "story_id",
            "video_id",
            "age_lo",
            "age_hi",
            "age_mid",
            "aoa_mean_story",
            "aoa_median_story",
            "aoa_std_story",
            "aoa_p90_story",
            "aoa_max_story",
            "aoa_n_matched",
            "aoa_total_tokens",
            "aoa_coverage_story",
        ]
        if c in stats.columns
    ]
    stats = stats[keep_cols].copy()
    merged = base.copy()
    if "story_id" in merged.columns and "story_id" in stats.columns:
        merged = merged.merge(stats.drop_duplicates(subset=["story_id"]), on="story_id", how="left", suffixes=("", "_aoastat"))
    if "video_id" in merged.columns and "video_id" in stats.columns:
        merged = merged.merge(stats.drop_duplicates(subset=["video_id"]), on="video_id", how="left", suffixes=("", "_aoavid"))
    for col in ["age_lo","age_hi","age_mid","aoa_mean_story","aoa_median_story","aoa_std_story","aoa_p90_story","aoa_max_story","aoa_n_matched","aoa_total_tokens","aoa_coverage_story"]:
        for suffix in ["_aoastat", "_aoavid"]:
            c = f"{col}{suffix}"
            if c in merged.columns:
                merged[col] = pd.to_numeric(merged.get(col), errors="coerce").fillna(pd.to_numeric(merged[c], errors="coerce"))
    drop_cols = [c for c in merged.columns if c.endswith("_aoastat") or c.endswith("_aoavid")]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    return merged


def load_story_artifacts(vectors_path: Path, feature_paths: Sequence[Path], aoa_stats_path: Optional[Path]) -> tuple[np.ndarray, pd.DataFrame]:
    vec = np.load(vectors_path, allow_pickle=True)
    vectors = vec["vectors"]
    dfv = pd.DataFrame(
        {
            "vec_row": np.arange(vectors.shape[0], dtype=int),
            "item_id": vec["item_id"].astype(str),
            "video_id": vec["item_id"].astype(str),
            "story_id": [None if x is None or str(x).strip()=="" else str(x) for x in vec["story_id"].tolist()],
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
        merged = merged.merge(feat, on="item_id", how="left", suffixes=("", f"_feat{i}"))
        for base_col in ["story_id", "video_id", "age_mid"]:
            extra_col = f"{base_col}_feat{i}"
            if extra_col in merged.columns:
                merged[base_col] = merged[base_col].fillna(merged[extra_col])
                merged = merged.drop(columns=[extra_col])
    if "age_mid" not in merged.columns:
        merged["age_mid"] = merged["age_mid_vec"]
    merged["story_id"] = merged["story_id"].map(_normalize_story_id)
    merged = merge_story_level_aoa_stats(merged, aoa_stats_path)
    return vectors, merged


def choose_alpha(X_val: np.ndarray, weight_vec: np.ndarray, bias: float, steer_dir: np.ndarray, alphas: Sequence[float], target_frac: float) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha in alphas:
        Xs = X_val + float(alpha) * steer_dir[None, :]
        probs = sigmoid(Xs @ weight_vec + bias)
        rows.append({"alpha": float(alpha), "mean_prob": float(probs.mean())})
    curve = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    max_score = float(curve["mean_prob"].max())
    threshold = float(target_frac * max_score)
    chosen = curve[curve["mean_prob"] >= threshold].iloc[0]
    return float(chosen["alpha"]), curve


def summarize_shift(X: np.ndarray, steer_dir: np.ndarray, alpha: float, age_w: np.ndarray, age_b: float, aoa_w: np.ndarray, aoa_b: float) -> Dict[str, float]:
    Xs = X + float(alpha) * steer_dir[None, :]
    age_before = sigmoid(X @ age_w + age_b)
    age_after = sigmoid(Xs @ age_w + age_b)
    aoa_before = sigmoid(X @ aoa_w + aoa_b)
    aoa_after = sigmoid(Xs @ aoa_w + aoa_b)
    cos = np.sum(X * Xs, axis=1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Xs, axis=1) + 1e-12)
    return {
        "n_items": int(X.shape[0]),
        "alpha": float(alpha),
        "mean_age_prob_before": float(age_before.mean()),
        "mean_age_prob_after": float(age_after.mean()),
        "delta_age_prob": float(age_after.mean() - age_before.mean()),
        "mean_aoa_prob_before": float(aoa_before.mean()),
        "mean_aoa_prob_after": float(aoa_after.mean()),
        "delta_aoa_prob": float(aoa_after.mean() - aoa_before.mean()),
        "mean_cosine_to_original": float(cos.mean()),
    }


def per_item_shift(X: np.ndarray, steer_dir: np.ndarray, alpha: float, age_w: np.ndarray, age_b: float, aoa_w: np.ndarray, aoa_b: float) -> Dict[str, np.ndarray]:
    Xs = X + float(alpha) * steer_dir[None, :]
    age_before = sigmoid(X @ age_w + age_b)
    age_after = sigmoid(Xs @ age_w + age_b)
    aoa_before = sigmoid(X @ aoa_w + aoa_b)
    aoa_after = sigmoid(Xs @ aoa_w + aoa_b)
    cos = np.sum(X * Xs, axis=1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Xs, axis=1) + 1e-12)
    return {
        "age_prob_before": age_before,
        "age_prob_after": age_after,
        "delta_age_prob": age_after - age_before,
        "aoa_prob_before": aoa_before,
        "aoa_prob_after": aoa_after,
        "delta_aoa_prob": aoa_after - aoa_before,
        "cosine_to_original": cos,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Steer Amazon story embeddings from precomputed AoA word vectors.")
    ap.add_argument("--story-vectors", type=Path, required=True)
    ap.add_argument("--word-vectors", type=Path, required=True)
    ap.add_argument("--features", type=str, default="artifacts/text_features_batch1.csv")
    ap.add_argument("--story-aoa-stats", type=Path, default=Path("artifacts/corpus_aoa_cutoffs_cleaned_idchecked/story_aoa_stats.csv"))
    ap.add_argument("--cleaned-meta", type=Path, default=Path("story_wAmazonData_042626/amazon_youtube_batch1_wid.csv"))
    ap.add_argument("--cleaned-col", type=str, default="cleaned")
    ap.add_argument("--age-cutoff", type=float, default=5.0)
    ap.add_argument("--story-aoa-col", type=str, default="aoa_mean_story")
    ap.add_argument("--story-aoa-low-quantile", type=float, default=0.33)
    ap.add_argument("--story-aoa-high-quantile", type=float, default=0.67)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alphas", type=str, default="0,0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--alpha-target-frac", type=float, default=0.95)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    story_vectors, merged = load_story_artifacts(args.story_vectors, parse_paths(args.features), args.story_aoa_stats)
    story_ids, video_ids = load_cleaned_filter(args.cleaned_meta, args.cleaned_col)
    merged = filter_cleaned_df(merged, story_ids, video_ids)
    merged["age_mid"] = pd.to_numeric(merged["age_mid"], errors="coerce")
    merged[args.story_aoa_col] = pd.to_numeric(merged[args.story_aoa_col], errors="coerce")
    merged = merged[merged["age_mid"].notna() & merged[args.story_aoa_col].notna()].copy()
    merged["age_label"] = (merged["age_mid"] > float(args.age_cutoff)).astype(int)
    merged = merged.drop_duplicates(subset=["item_id"]).reset_index(drop=True)

    word_npz = np.load(args.word_vectors, allow_pickle=True)
    word_vectors = word_npz["vectors"]
    word_aoa_label = word_npz["aoa_label"].astype(int)
    word_meta = json.loads(str(word_npz["meta"]))

    idx_all = np.arange(len(merged))
    idx_train, idx_tmp = train_test_split(idx_all, test_size=float(args.val_size + args.test_size), random_state=int(args.seed), stratify=merged["age_label"].to_numpy(dtype=int))
    rel_test = float(args.test_size / (args.val_size + args.test_size))
    idx_val, idx_test = train_test_split(idx_tmp, test_size=rel_test, random_state=int(args.seed), stratify=merged.iloc[idx_tmp]["age_label"].to_numpy(dtype=int))
    split_col = np.array([""] * len(merged), dtype=object)
    split_col[idx_train] = "train"
    split_col[idx_val] = "val"
    split_col[idx_test] = "test"
    merged["split"] = split_col
    train_df = merged.iloc[idx_train].copy()
    val_df = merged.iloc[idx_val].copy()
    test_df = merged.iloc[idx_test].copy()

    lo = float(train_df[args.story_aoa_col].quantile(args.story_aoa_low_quantile))
    hi = float(train_df[args.story_aoa_col].quantile(args.story_aoa_high_quantile))
    for df_part in (train_df, val_df, test_df):
        df_part["story_aoa_label"] = np.where(df_part[args.story_aoa_col] <= lo, 0, np.where(df_part[args.story_aoa_col] >= hi, 1, np.nan))

    args.outdir.mkdir(parents=True, exist_ok=True)
    alphas = parse_floats(args.alphas)
    results = []
    curves = []
    per_item_rows = []
    age_bin_rows = []
    for layer in range(story_vectors.shape[1]):
        X_train = story_vectors[train_df["vec_row"].astype(int).to_numpy(), layer, :]
        X_val = story_vectors[val_df["vec_row"].astype(int).to_numpy(), layer, :]
        X_test = story_vectors[test_df["vec_row"].astype(int).to_numpy(), layer, :]
        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)
        X_val_z = scaler.transform(X_val)
        X_test_z = scaler.transform(X_test)

        age_probe = LogisticRegression(max_iter=2000, n_jobs=1)
        age_probe.fit(X_train_z, train_df["age_label"].to_numpy(dtype=int))
        age_val_auc = float(roc_auc_score(val_df["age_label"].to_numpy(dtype=int), age_probe.predict_proba(X_val_z)[:, 1]))
        age_test_auc = float(roc_auc_score(test_df["age_label"].to_numpy(dtype=int), age_probe.predict_proba(X_test_z)[:, 1]))

        train_ext = train_df["story_aoa_label"].notna().to_numpy()
        val_ext = val_df["story_aoa_label"].notna().to_numpy()
        test_ext = test_df["story_aoa_label"].notna().to_numpy()
        aoa_probe = LogisticRegression(max_iter=2000, n_jobs=1)
        aoa_probe.fit(X_train_z[train_ext], train_df.loc[train_ext, "story_aoa_label"].to_numpy(dtype=int))
        aoa_val_auc = float(roc_auc_score(val_df.loc[val_ext, "story_aoa_label"].to_numpy(dtype=int), aoa_probe.predict_proba(X_val_z[val_ext])[:, 1]))
        aoa_test_auc = float(roc_auc_score(test_df.loc[test_ext, "story_aoa_label"].to_numpy(dtype=int), aoa_probe.predict_proba(X_test_z[test_ext])[:, 1]))

        X_word = word_vectors[:, layer, :]
        v_word_raw = normalized_direction(X_word[word_aoa_label == 1].mean(axis=0), X_word[word_aoa_label == 0].mean(axis=0))
        v_story = normalized_direction(v_word_raw / (scaler.scale_ + 1e-12), np.zeros_like(v_word_raw))

        age_w = age_probe.coef_[0]
        age_b = float(age_probe.intercept_[0])
        aoa_w = aoa_probe.coef_[0]
        aoa_b = float(aoa_probe.intercept_[0])

        X_val_young = X_val_z[val_df["age_label"].to_numpy(dtype=int) == 0]
        X_test_young = X_test_z[test_df["age_label"].to_numpy(dtype=int) == 0]
        best_alpha, curve = choose_alpha(X_val_young, age_w, age_b, v_story, alphas, float(args.alpha_target_frac))
        shift = summarize_shift(X_test_young, v_story, best_alpha, age_w, age_b, aoa_w, aoa_b)
        young_mask_test = test_df["age_label"].to_numpy(dtype=int) == 0
        young_test_df = test_df.loc[young_mask_test].copy().reset_index(drop=True)
        item_shift = per_item_shift(X_test_young, v_story, best_alpha, age_w, age_b, aoa_w, aoa_b)
        results.append(
            {
                "layer": layer,
                "age_val_auc": age_val_auc,
                "age_test_auc": age_test_auc,
                "story_aoa_val_auc": aoa_val_auc,
                "story_aoa_test_auc": aoa_test_auc,
                "best_alpha": best_alpha,
                **shift,
            }
        )
        for _, row in curve.iterrows():
            curves.append({"layer": layer, "alpha": float(row["alpha"]), "val_mean_age_prob": float(row["mean_prob"])})
        item_df = young_test_df[["item_id", "story_id", "video_id", "age_mid", args.story_aoa_col]].copy()
        item_df["layer"] = layer
        item_df["best_alpha"] = float(best_alpha)
        for key, vals in item_shift.items():
            item_df[key] = vals
        per_item_rows.append(item_df)

        bin_df = item_df.copy()
        bin_df["age_mid_bin"] = bin_df["age_mid"].round(1)
        grouped = (
            bin_df.groupby("age_mid_bin", dropna=True)
            .agg(
                n_items=("item_id", "count"),
                mean_age_mid=("age_mid", "mean"),
                mean_delta_age_prob=("delta_age_prob", "mean"),
                mean_delta_aoa_prob=("delta_aoa_prob", "mean"),
                mean_age_prob_before=("age_prob_before", "mean"),
                mean_age_prob_after=("age_prob_after", "mean"),
                mean_cosine_to_original=("cosine_to_original", "mean"),
            )
            .reset_index()
        )
        grouped["layer"] = layer
        grouped["best_alpha"] = float(best_alpha)
        age_bin_rows.append(grouped)

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.outdir / "steering_results.csv", index=False)
    pd.DataFrame(curves).to_csv(args.outdir / "val_curve.csv", index=False)
    if per_item_rows:
        pd.concat(per_item_rows, ignore_index=True).to_csv(args.outdir / "per_item_steering.csv", index=False)
    if age_bin_rows:
        pd.concat(age_bin_rows, ignore_index=True).to_csv(args.outdir / "age_binned_steering.csv", index=False)
    merged[["item_id","story_id","video_id","age_mid",args.story_aoa_col,"age_label","split"]].to_csv(args.outdir / "split_assignments.csv", index=False)
    best = results_df.sort_values("delta_age_prob", ascending=False).iloc[0]
    summary = {
        "n_items": int(len(merged)),
        "split_counts": merged["split"].value_counts().to_dict(),
        "age_cutoff": float(args.age_cutoff),
        "story_aoa_col": args.story_aoa_col,
        "story_aoa_train_low_cutoff": lo,
        "story_aoa_train_high_cutoff": hi,
        "word_vector_meta": word_meta,
        "best_layer": int(best["layer"]),
        "best_alpha": float(best["best_alpha"]),
        "best_delta_age_prob": float(best["delta_age_prob"]),
        "best_delta_aoa_prob": float(best["delta_aoa_prob"]),
        "best_mean_cosine_to_original": float(best["mean_cosine_to_original"]),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote: {args.outdir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
