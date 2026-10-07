#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    stats = stats[keep_cols].drop_duplicates(subset=[c for c in ["story_id", "video_id"] if c in keep_cols]).copy()
    merged = base.copy()
    if "story_id" in merged.columns and "story_id" in stats.columns:
        stats_by_story = stats.drop_duplicates(subset=["story_id"]).copy()
        merged = merged.merge(stats_by_story, on="story_id", how="left", suffixes=("", "_aoastat"))
    if "video_id" in merged.columns and "video_id" in stats.columns:
        stats_by_video = stats.drop_duplicates(subset=["video_id"]).copy()
        merged = merged.merge(stats_by_video, on="video_id", how="left", suffixes=("", "_aoavid"))
    for col in [
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
    ]:
        c_story = f"{col}_aoastat"
        c_video = f"{col}_aoavid"
        if c_story in merged.columns:
            merged[col] = pd.to_numeric(merged.get(col), errors="coerce").fillna(pd.to_numeric(merged[c_story], errors="coerce"))
        if c_video in merged.columns:
            merged[col] = pd.to_numeric(merged.get(col), errors="coerce").fillna(pd.to_numeric(merged[c_video], errors="coerce"))
    drop_cols = [c for c in merged.columns if c.endswith("_aoastat") or c.endswith("_aoavid")]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    return merged


def load_story_artifacts(
    vectors_path: Path,
    feature_paths: Sequence[Path],
    aoa_stats_path: Optional[Path] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
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
    merged = merge_story_level_aoa_stats(merged, aoa_stats_path)
    return vectors, merged


def embed_aoa_words(
    lexicon_path: Path,
    model_name: str,
    aoa_low: float,
    aoa_high: float,
    batch_size: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    import torch  # type: ignore
    from transformers import DistilBertModel, DistilBertTokenizerFast  # type: ignore

    lex = pd.read_excel(lexicon_path)
    lex["Rating.Mean"] = pd.to_numeric(lex["Rating.Mean"], errors="coerce")
    lex["Word"] = lex["Word"].astype(str).str.strip()
    lex = lex[lex["Rating.Mean"].notna() & lex["Word"].ne("")].copy()
    lex = lex.drop_duplicates(subset=["Word"]).reset_index(drop=True)
    lex["aoa_label"] = np.where(lex["Rating.Mean"] < aoa_low, 0, np.where(lex["Rating.Mean"] > aoa_high, 1, np.nan))
    lex = lex[lex["aoa_label"].notna()].copy()
    lex["aoa_label"] = lex["aoa_label"].astype(int)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    vectors: List[np.ndarray] = []
    words = lex["Word"].tolist()
    with torch.no_grad():
        for start in range(0, len(words), batch_size):
            batch = words[start : start + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states
            attn = enc["attention_mask"].bool()
            attn[:, 0] = False
            lengths = enc["attention_mask"].long().sum(dim=1).clamp(min=1)
            sep_idx = (lengths - 1).clamp(min=0)
            attn[torch.arange(attn.shape[0], device=device), sep_idx] = False
            denom = attn.sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled = []
            for hs in hidden:
                pooled.append((hs * attn.unsqueeze(-1)).sum(dim=1) / denom)
            batch_vec = torch.stack(pooled, dim=1).cpu().numpy()
            vectors.append(batch_vec)
    vec_arr = np.concatenate(vectors, axis=0)
    return vec_arr, lex.reset_index(drop=True)


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
        rows.append({"alpha": float(alpha), "mean_prob": float(probs.mean())})
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
        "mean_aoa_prob_before": float(aoa_before.mean()),
        "mean_aoa_prob_after": float(aoa_after.mean()),
        "delta_aoa_prob": float(aoa_after.mean() - aoa_before.mean()),
        "high_aoa_pred_frac_before": float((aoa_before >= 0.5).mean()),
        "high_aoa_pred_frac_after": float((aoa_after >= 0.5).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Steer story embeddings with fixed AoA word directions (<4 vs >9).")
    ap.add_argument("--vectors", type=Path, default=Path("results/features/representation_extraction/encoder/distilbert_vectors/story_vectors.npz"))
    ap.add_argument("--features", type=str, default="results/features/eda/text_features_batch1_review_samples.csv")
    ap.add_argument("--cleaned-meta", type=Path, default=Path("story_wAmazonData_042626/amazon_youtube_batch1_wid.csv"))
    ap.add_argument("--cleaned-col", type=str, default="cleaned")
    ap.add_argument(
        "--story-aoa-stats",
        type=Path,
        default=Path("results/features/eda/corpus_aoa_cutoffs_cleaned_idchecked/story_aoa_stats.csv"),
    )
    ap.add_argument("--aoa-lexicon", type=Path, default=Path("data/aoa/AoA_ratings_Kuperman_et_al_BRM_with_PoS.xlsx"))
    ap.add_argument("--model", type=str, default="distilbert-base-uncased")
    ap.add_argument("--age-cutoff", type=float, default=5.0)
    ap.add_argument("--story-aoa-col", type=str, default="aoa_mean_story")
    ap.add_argument("--story-aoa-low-quantile", type=float, default=0.33)
    ap.add_argument("--story-aoa-high-quantile", type=float, default=0.67)
    ap.add_argument("--word-aoa-low", type=float, default=4.0)
    ap.add_argument("--word-aoa-high", type=float, default=9.0)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alphas", type=str, default="0,0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--alpha-target-frac", type=float, default=0.95)
    ap.add_argument("--word-batch-size", type=int, default=64)
    ap.add_argument("--outdir", type=Path, default=Path("results/analysis/representation/steering/distilbert/steer/fixed_aoa_words"))
    args = ap.parse_args()

    story_vectors, merged = load_story_artifacts(
        args.vectors,
        parse_paths(args.features),
        aoa_stats_path=args.story_aoa_stats,
    )
    if args.cleaned_meta is not None and args.cleaned_meta.exists():
        story_ids, video_ids = load_cleaned_filter(args.cleaned_meta, args.cleaned_col)
        merged = filter_cleaned_df(merged, story_ids, video_ids)
    merged["age_mid"] = pd.to_numeric(merged["age_mid"], errors="coerce")
    merged[args.story_aoa_col] = pd.to_numeric(merged[args.story_aoa_col], errors="coerce")
    merged = merged[merged["age_mid"].notna() & merged[args.story_aoa_col].notna()].copy()
    merged["age_label"] = (merged["age_mid"] > float(args.age_cutoff)).astype(int)
    merged = merged.drop_duplicates(subset=["item_id"]).reset_index(drop=True)

    item_index = np.arange(len(merged))
    idx_train, idx_tmp = train_test_split(
        item_index,
        test_size=float(args.val_size + args.test_size),
        random_state=int(args.seed),
        stratify=merged["age_label"].to_numpy(dtype=int),
    )
    rel_test = float(args.test_size / (args.val_size + args.test_size))
    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=rel_test,
        random_state=int(args.seed),
        stratify=merged.iloc[idx_tmp]["age_label"].to_numpy(dtype=int),
    )
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
        df_part["story_aoa_label"] = np.where(
            df_part[args.story_aoa_col] <= lo, 0, np.where(df_part[args.story_aoa_col] >= hi, 1, np.nan)
        )

    word_vectors, lex = embed_aoa_words(
        lexicon_path=args.aoa_lexicon,
        model_name=args.model,
        aoa_low=float(args.word_aoa_low),
        aoa_high=float(args.word_aoa_high),
        batch_size=int(args.word_batch_size),
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    lex.to_csv(args.outdir / "aoa_word_split.csv", index=False)

    alphas = parse_floats(args.alphas)
    rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    word_rows: List[Dict[str, Any]] = []

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

        train_story_extreme = train_df["story_aoa_label"].notna().to_numpy()
        val_story_extreme = val_df["story_aoa_label"].notna().to_numpy()
        test_story_extreme = test_df["story_aoa_label"].notna().to_numpy()
        X_train_aoa = X_train_z[train_story_extreme]
        X_val_aoa = X_val_z[val_story_extreme]
        X_test_aoa = X_test_z[test_story_extreme]
        y_train_aoa = train_df.loc[train_story_extreme, "story_aoa_label"].to_numpy(dtype=int)
        y_val_aoa = val_df.loc[val_story_extreme, "story_aoa_label"].to_numpy(dtype=int)
        y_test_aoa = test_df.loc[test_story_extreme, "story_aoa_label"].to_numpy(dtype=int)
        aoa_probe = LogisticRegression(max_iter=2000, n_jobs=1)
        aoa_probe.fit(X_train_aoa, y_train_aoa)
        aoa_val_auc = float(roc_auc_score(y_val_aoa, aoa_probe.predict_proba(X_val_aoa)[:, 1]))
        aoa_test_auc = float(roc_auc_score(y_test_aoa, aoa_probe.predict_proba(X_test_aoa)[:, 1]))

        X_word = word_vectors[:, layer, :]
        word_labels = lex["aoa_label"].to_numpy(dtype=int)
        v_aoa_word_raw = normalized_direction(
            X_word[word_labels == 1].mean(axis=0),
            X_word[word_labels == 0].mean(axis=0),
        )
        v_aoa_word_story = normalized_direction(v_aoa_word_raw / (scaler.scale_ + 1e-12), np.zeros_like(v_aoa_word_raw))

        age_w = age_probe.coef_[0]
        age_b = float(age_probe.intercept_[0])
        aoa_w = aoa_probe.coef_[0]
        aoa_b = float(aoa_probe.intercept_[0])

        X_val_young = X_val_z[val_df["age_label"].to_numpy(dtype=int) == 0]
        X_test_young = X_test_z[test_df["age_label"].to_numpy(dtype=int) == 0]
        best_alpha, val_curve = choose_alpha(
            X_val_young, age_w, age_b, v_aoa_word_story, alphas, float(args.alpha_target_frac)
        )
        shift = summarize_shift(X_test_young, v_aoa_word_story, best_alpha, age_w, age_b, aoa_w, aoa_b)
        word_rows.append(
            {
                "layer": layer,
                "n_word_low": int((lex["aoa_label"] == 0).sum()),
                "n_word_high": int((lex["aoa_label"] == 1).sum()),
                "word_low_mean_norm": float(np.linalg.norm(X_word[word_labels == 0].mean(axis=0))),
                "word_high_mean_norm": float(np.linalg.norm(X_word[word_labels == 1].mean(axis=0))),
            }
        )

        rows.append(
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
        for _, row in val_curve.iterrows():
            val_rows.append(
                {
                    "layer": layer,
                    "alpha": float(row["alpha"]),
                    "val_mean_age_prob": float(row["mean_prob"]),
                }
            )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.outdir / "fixed_aoa_word_steering_results.csv", index=False)
    pd.DataFrame(val_rows).to_csv(args.outdir / "fixed_aoa_word_steering_val_curve.csv", index=False)
    pd.DataFrame(word_rows).to_csv(args.outdir / "fixed_aoa_word_direction_stats.csv", index=False)
    merged[["item_id", "story_id", "video_id", "age_mid", args.story_aoa_col, "age_label", "split"]].to_csv(
        args.outdir / "split_assignments.csv", index=False
    )

    best = results_df.sort_values("delta_age_prob", ascending=False).iloc[0]
    summary = {
        "n_items": int(len(merged)),
        "split_counts": merged["split"].value_counts().to_dict(),
        "age_cutoff": float(args.age_cutoff),
        "story_aoa_train_low_cutoff": lo,
        "story_aoa_train_high_cutoff": hi,
        "word_aoa_low": float(args.word_aoa_low),
        "word_aoa_high": float(args.word_aoa_high),
        "n_word_low": int((lex["aoa_label"] == 0).sum()),
        "n_word_high": int((lex["aoa_label"] == 1).sum()),
        "best_layer": int(best["layer"]),
        "best_alpha": float(best["best_alpha"]),
        "best_delta_age_prob": float(best["delta_age_prob"]),
        "best_delta_aoa_prob": float(best["delta_aoa_prob"]),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote: {args.outdir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
