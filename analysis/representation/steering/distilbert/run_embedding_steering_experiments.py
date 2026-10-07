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


def parse_paths(spec: str) -> List[Path]:
    return [Path(s.strip()) for s in spec.split(",") if s.strip()]


def parse_floats(spec: str) -> List[float]:
    return [float(s.strip()) for s in spec.split(",") if s.strip()]


def normalized_direction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = a - b
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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
        for c in ["age_mid", "aoa_mean", "aoa_std", "aoa_coverage"]:
            if c in feat.columns:
                feat[c] = pd.to_numeric(feat[c], errors="coerce")
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


def choose_alpha(
    X_val: np.ndarray,
    weight_vec: np.ndarray,
    bias: float,
    steer_dir: np.ndarray,
    alphas: Sequence[float],
    target_frac: float,
) -> tuple[float, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    for alpha in alphas:
        Xs = X_val + float(alpha) * steer_dir[None, :]
        probs = sigmoid(Xs @ weight_vec + bias)
        score = float(probs.mean())
        rows.append({"alpha": float(alpha), "mean_prob": score})
    curve = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    max_score = float(curve["mean_prob"].max())
    threshold = float(target_frac * max_score)
    chosen = curve[curve["mean_prob"] >= threshold].iloc[0]
    return float(chosen["alpha"]), curve


def summarize_shift(
    X: np.ndarray,
    steer_dir: np.ndarray,
    alpha: float,
    age_w: np.ndarray,
    age_b: float,
    aoa_w: np.ndarray,
    aoa_b: float,
) -> Dict[str, float]:
    Xs = X + float(alpha) * steer_dir[None, :]
    age_before = sigmoid(X @ age_w + age_b)
    age_after = sigmoid(Xs @ age_w + age_b)
    aoa_before = sigmoid(X @ aoa_w + aoa_b)
    aoa_after = sigmoid(Xs @ aoa_w + aoa_b)
    return {
        "n_items": int(X.shape[0]),
        "alpha": float(alpha),
        "mean_age_prob_before": float(age_before.mean()),
        "mean_age_prob_after": float(age_after.mean()),
        "delta_age_prob": float(age_after.mean() - age_before.mean()),
        "older_pred_frac_before": float((age_before >= 0.5).mean()),
        "older_pred_frac_after": float((age_after >= 0.5).mean()),
        "delta_older_pred_frac": float((age_after >= 0.5).mean() - (age_before >= 0.5).mean()),
        "mean_aoa_prob_before": float(aoa_before.mean()),
        "mean_aoa_prob_after": float(aoa_after.mean()),
        "delta_aoa_prob": float(aoa_after.mean() - aoa_before.mean()),
        "high_aoa_pred_frac_before": float((aoa_before >= 0.5).mean()),
        "high_aoa_pred_frac_after": float((aoa_after >= 0.5).mean()),
        "delta_high_aoa_pred_frac": float((aoa_after >= 0.5).mean() - (aoa_before >= 0.5).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out embedding steering experiments for age and AoA directions.")
    ap.add_argument("--vectors", type=Path, default=Path("results/features/representation_extraction/encoder/distilbert_vectors/story_vectors.npz"))
    ap.add_argument(
        "--features",
        type=str,
        default="results/features/eda/text_features_batch1_review_samples.csv",
        help="Comma-separated feature CSVs to merge on item_id.",
    )
    ap.add_argument("--outdir", type=Path, default=Path("results/analysis/representation/steering/distilbert/embedding_steering_experiments"))
    ap.add_argument("--cleaned-meta", type=Path, default=Path("story_wAmazonData_042626/amazon_youtube_batch1_wid.csv"))
    ap.add_argument("--cleaned-col", type=str, default="cleaned")
    ap.add_argument("--age-cutoff", type=float, default=5.0)
    ap.add_argument("--aoa-col", type=str, default="aoa_mean")
    ap.add_argument("--aoa-low-quantile", type=float, default=0.33)
    ap.add_argument("--aoa-high-quantile", type=float, default=0.67)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alphas", type=str, default="0,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--alpha-target-frac", type=float, default=0.95)
    args = ap.parse_args()

    vectors, merged = load_artifacts(args.vectors, parse_paths(args.features))
    if args.cleaned_meta is not None and args.cleaned_meta.exists():
        story_ids, video_ids = load_cleaned_filter(args.cleaned_meta, args.cleaned_col)
        merged = filter_cleaned_df(merged, story_ids, video_ids)

    merged["age_mid"] = pd.to_numeric(merged["age_mid"], errors="coerce")
    merged[args.aoa_col] = pd.to_numeric(merged[args.aoa_col], errors="coerce")
    merged = merged[merged["age_mid"].notna() & merged[args.aoa_col].notna()].copy()
    merged["age_label"] = (merged["age_mid"] > float(args.age_cutoff)).astype(int)
    merged = merged.drop_duplicates(subset=["item_id"]).reset_index(drop=True)

    args.outdir.mkdir(parents=True, exist_ok=True)

    item_ids = merged["item_id"].astype(str).to_numpy()
    age_y = merged["age_label"].to_numpy(dtype=int)
    idx_train, idx_tmp = train_test_split(
        np.arange(len(merged)),
        test_size=float(args.val_size + args.test_size),
        random_state=int(args.seed),
        stratify=age_y,
    )
    rel_test = float(args.test_size / (args.val_size + args.test_size))
    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=rel_test,
        random_state=int(args.seed),
        stratify=age_y[idx_tmp],
    )

    split_col = np.array([""] * len(merged), dtype=object)
    split_col[idx_train] = "train"
    split_col[idx_val] = "val"
    split_col[idx_test] = "test"
    merged["split"] = split_col

    train_df = merged.iloc[idx_train].copy()
    val_df = merged.iloc[idx_val].copy()
    test_df = merged.iloc[idx_test].copy()

    lo = float(train_df[args.aoa_col].quantile(args.aoa_low_quantile))
    hi = float(train_df[args.aoa_col].quantile(args.aoa_high_quantile))
    for df_part in (train_df, val_df, test_df):
        df_part["aoa_label"] = np.where(df_part[args.aoa_col] <= lo, 0, np.where(df_part[args.aoa_col] >= hi, 1, np.nan))

    alphas = parse_floats(args.alphas)
    layer_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []

    for layer in range(vectors.shape[1]):
        X_train = vectors[train_df["vec_row"].astype(int).to_numpy(), layer, :]
        X_val = vectors[val_df["vec_row"].astype(int).to_numpy(), layer, :]
        X_test = vectors[test_df["vec_row"].astype(int).to_numpy(), layer, :]

        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)
        X_val_z = scaler.transform(X_val)
        X_test_z = scaler.transform(X_test)

        age_probe = LogisticRegression(max_iter=2000, n_jobs=1)
        age_probe.fit(X_train_z, train_df["age_label"].to_numpy(dtype=int))
        age_val_auc = float(roc_auc_score(val_df["age_label"].to_numpy(dtype=int), age_probe.predict_proba(X_val_z)[:, 1]))
        age_test_auc = float(roc_auc_score(test_df["age_label"].to_numpy(dtype=int), age_probe.predict_proba(X_test_z)[:, 1]))

        train_aoa_extreme = train_df["aoa_label"].notna().to_numpy()
        val_aoa_extreme = val_df["aoa_label"].notna().to_numpy()
        test_aoa_extreme = test_df["aoa_label"].notna().to_numpy()
        X_train_aoa = X_train_z[train_aoa_extreme]
        X_val_aoa = X_val_z[val_aoa_extreme]
        X_test_aoa = X_test_z[test_aoa_extreme]
        y_train_aoa = train_df.loc[train_aoa_extreme, "aoa_label"].to_numpy(dtype=int)
        y_val_aoa = val_df.loc[val_aoa_extreme, "aoa_label"].to_numpy(dtype=int)
        y_test_aoa = test_df.loc[test_aoa_extreme, "aoa_label"].to_numpy(dtype=int)

        aoa_probe = LogisticRegression(max_iter=2000, n_jobs=1)
        aoa_probe.fit(X_train_aoa, y_train_aoa)
        aoa_val_auc = float(roc_auc_score(y_val_aoa, aoa_probe.predict_proba(X_val_aoa)[:, 1]))
        aoa_test_auc = float(roc_auc_score(y_test_aoa, aoa_probe.predict_proba(X_test_aoa)[:, 1]))

        v_age = normalized_direction(
            X_train_z[train_df["age_label"].to_numpy(dtype=int) == 1].mean(axis=0),
            X_train_z[train_df["age_label"].to_numpy(dtype=int) == 0].mean(axis=0),
        )
        v_aoa = normalized_direction(
            X_train_aoa[y_train_aoa == 1].mean(axis=0),
            X_train_aoa[y_train_aoa == 0].mean(axis=0),
        )

        age_w = age_probe.coef_[0]
        age_b = float(age_probe.intercept_[0])
        aoa_w = aoa_probe.coef_[0]
        aoa_b = float(aoa_probe.intercept_[0])

        X_val_young = X_val_z[val_df["age_label"].to_numpy(dtype=int) == 0]
        X_test_young = X_test_z[test_df["age_label"].to_numpy(dtype=int) == 0]
        best_alpha_age, age_val_curve = choose_alpha(
            X_val_young, age_w, age_b, v_age, alphas, target_frac=float(args.alpha_target_frac)
        )
        age_test_shift = summarize_shift(X_test_young, v_age, best_alpha_age, age_w, age_b, aoa_w, aoa_b)

        X_val_low = X_val_aoa[y_val_aoa == 0]
        X_test_low = X_test_aoa[y_test_aoa == 0]
        best_alpha_aoa, aoa_val_curve = choose_alpha(
            X_val_low, aoa_w, aoa_b, v_aoa, alphas, target_frac=float(args.alpha_target_frac)
        )
        aoa_test_shift = summarize_shift(X_test_low, v_aoa, best_alpha_aoa, age_w, age_b, aoa_w, aoa_b)

        layer_rows.append(
            {
                "layer": layer,
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "n_val_young": int(X_val_young.shape[0]),
                "n_test_young": int(X_test_young.shape[0]),
                "n_val_low_aoa": int(X_val_low.shape[0]),
                "n_test_low_aoa": int(X_test_low.shape[0]),
                "age_val_auc": age_val_auc,
                "age_test_auc": age_test_auc,
                "aoa_val_auc": aoa_val_auc,
                "aoa_test_auc": aoa_test_auc,
                "cos_age_vs_aoa": float(np.dot(v_age, v_aoa)),
                "best_alpha_age": best_alpha_age,
                "best_alpha_aoa": best_alpha_aoa,
                **{f"agesteer_{k}": v for k, v in age_test_shift.items()},
                **{f"aoasteer_{k}": v for k, v in aoa_test_shift.items()},
            }
        )

        for _, row in age_val_curve.iterrows():
            val_rows.append(
                {
                    "layer": layer,
                    "experiment": "age_to_older",
                    "alpha": float(row["alpha"]),
                    "val_target_mean_prob": float(row["mean_prob"]),
                }
            )
        for _, row in aoa_val_curve.iterrows():
            val_rows.append(
                {
                    "layer": layer,
                    "experiment": "aoa_to_late",
                    "alpha": float(row["alpha"]),
                    "val_target_mean_prob": float(row["mean_prob"]),
                }
            )

    results_df = pd.DataFrame(layer_rows)
    val_df_out = pd.DataFrame(val_rows)
    results_df.to_csv(args.outdir / "steering_results.csv", index=False)
    val_df_out.to_csv(args.outdir / "steering_val_curves.csv", index=False)
    merged[["item_id", "story_id", "video_id", "age_mid", args.aoa_col, "age_label", "split"]].to_csv(
        args.outdir / "split_assignments.csv", index=False
    )

    best_age = results_df.sort_values("agesteer_delta_age_prob", ascending=False).iloc[0].to_dict()
    best_aoa = results_df.sort_values("aoasteer_delta_aoa_prob", ascending=False).iloc[0].to_dict()
    summary = {
        "n_items": int(len(merged)),
        "age_cutoff": float(args.age_cutoff),
        "aoa_col": args.aoa_col,
        "aoa_low_quantile": float(args.aoa_low_quantile),
        "aoa_high_quantile": float(args.aoa_high_quantile),
        "aoa_train_low_cutoff": lo,
        "aoa_train_high_cutoff": hi,
        "split_counts": merged["split"].value_counts().to_dict(),
        "age_label_counts": merged["age_label"].value_counts().sort_index().to_dict(),
        "best_age_steer_layer": int(best_age["layer"]),
        "best_age_steer_alpha": float(best_age["best_alpha_age"]),
        "best_age_steer_delta_age_prob": float(best_age["agesteer_delta_age_prob"]),
        "best_age_steer_delta_aoa_prob": float(best_age["agesteer_delta_aoa_prob"]),
        "best_aoa_steer_layer": int(best_aoa["layer"]),
        "best_aoa_steer_alpha": float(best_aoa["best_alpha_aoa"]),
        "best_aoa_steer_delta_aoa_prob": float(best_aoa["aoasteer_delta_aoa_prob"]),
        "best_aoa_steer_delta_age_prob": float(best_aoa["aoasteer_delta_age_prob"]),
        "alpha_target_frac": float(args.alpha_target_frac),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.outdir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
