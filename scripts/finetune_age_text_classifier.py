#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split


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


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists() and path.suffix.lower() == ".parquet":
        alt = path.with_suffix(".csv.gz")
        if alt.exists():
            path = alt
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError:
            alt = path.with_suffix(".csv.gz")
            if alt.exists():
                return pd.read_csv(alt, low_memory=False)
            raise
    if path.suffix.lower() in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        df = pd.read_csv(path, low_memory=False)
        if "story_id" in df.columns:
            df["story_id"] = df["story_id"].map(_normalize_story_id)
        if "video_id" in df.columns:
            df["video_id"] = df["video_id"].astype(str)
        return df
    raise ValueError(f"Unsupported input: {path}")


def filter_cleaned_rows(df: pd.DataFrame, meta_path: Path, cleaned_col: str) -> pd.DataFrame:
    meta = pd.read_csv(meta_path, low_memory=False)
    keep = pd.to_numeric(meta.get(cleaned_col), errors="coerce").fillna(0).astype(int) == 1
    meta = meta[keep].copy()
    matched = pd.Series(False, index=df.index)
    out = df.copy()
    if "story_id" in out.columns and "story_id" in meta.columns:
        meta["story_id"] = meta["story_id"].map(_normalize_story_id)
        matched = matched | out["story_id"].isin(set(meta["story_id"].dropna().astype(str)))
    if "video_id" in out.columns and "video_id" in meta.columns:
        meta["video_id"] = meta["video_id"].astype(str).str.strip()
        out["video_id"] = out["video_id"].astype(str).str.strip()
        matched = matched | out["video_id"].isin(set(meta["video_id"].dropna().astype(str)))
    return out[matched].copy()


def compute_binary_metrics_from_scores(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    pred = (scores >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(labels, pred)),
        "f1_macro": float(f1_score(labels, pred, average="macro")),
    }
    if len(np.unique(labels)) == 2:
        out["roc_auc"] = float(roc_auc_score(labels, scores))
    return out


def compute_classifier_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_pred
    probs = np.asarray(logits)
    if probs.ndim == 2 and probs.shape[1] == 2:
        pos = probs[:, 1]
    else:
        pos = probs.reshape(-1)
    labels = np.asarray(labels).astype(int)
    return compute_binary_metrics_from_scores(pos, labels)


def compute_regression_metrics(eval_pred: tuple[np.ndarray, np.ndarray], age_cutoff: float) -> Dict[str, float]:
    preds, labels = eval_pred
    y_pred = np.asarray(preds).reshape(-1)
    y_true = np.asarray(labels).reshape(-1)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    out: Dict[str, float] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }
    cls_true = (y_true > float(age_cutoff)).astype(int)
    cls_score = 1.0 / (1.0 + np.exp(-(y_pred - float(age_cutoff))))
    cls_metrics = compute_binary_metrics_from_scores(cls_score, cls_true)
    for k, v in cls_metrics.items():
        out[f"cutoff5_{k}" if age_cutoff == 5.0 else f"cutoff_{age_cutoff:g}_{k}"] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a text classifier for younger vs older story age.")
    ap.add_argument("--master", type=Path, default=Path("data/processed/master_stories.parquet"))
    ap.add_argument("--cleaned-meta", type=Path, default=Path("story_wAmazonData_042626/amazon_youtube_batch1_wid.csv"))
    ap.add_argument("--cleaned-col", type=str, default="cleaned")
    ap.add_argument("--text-col", type=str, default="transcript_best")
    ap.add_argument("--model", type=str, default="roberta-base")
    ap.add_argument("--task", type=str, choices=["classification", "regression"], default="regression")
    ap.add_argument("--age-cutoff", type=float, default=5.0)
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/finetuned_age_classifier"))
    ap.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="Optional CSV with item_id and split columns to force train/val/test assignment.",
    )
    ap.add_argument(
        "--split-col",
        type=str,
        default="split",
        help="Which split column to read from --split-csv. Use evaluator_split for decoder rewrite evaluation.",
    )
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--train-batch-size", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    global torch
    import torch  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )

    df = read_table(args.master)
    if args.cleaned_meta is not None and args.cleaned_meta.exists():
        df = filter_cleaned_rows(df, args.cleaned_meta, args.cleaned_col)
    cols = [c for c in ["video_id", "story_id", "age_mid", args.text_col] if c in df.columns]
    work = df[cols].copy().rename(columns={"video_id": "item_id", args.text_col: "text"})
    work["item_id"] = work["item_id"].astype(str)
    work["age_mid"] = pd.to_numeric(work["age_mid"], errors="coerce")
    work = work[work["text"].notna() & work["age_mid"].notna()].copy()
    work["text"] = work["text"].astype(str).str.strip()
    work = work[work["text"].ne("")].copy()
    work["label"] = (work["age_mid"] > float(args.age_cutoff)).astype(int)
    work["target"] = work["age_mid"].astype(float) if args.task == "regression" else work["label"].astype(int)
    work = work.drop_duplicates(subset=["item_id"]).reset_index(drop=True)

    if args.split_csv is not None and args.split_csv.exists():
        split_df = pd.read_csv(args.split_csv, low_memory=False)
        if "item_id" not in split_df.columns or args.split_col not in split_df.columns:
            raise ValueError(f"--split-csv must contain item_id and {args.split_col} columns")
        split_df["item_id"] = split_df["item_id"].astype(str)
        work = work.merge(split_df[["item_id", args.split_col]].rename(columns={args.split_col: "split"}), on="item_id", how="inner")
        if work["split"].nunique() < 3:
            raise ValueError("Merged split CSV does not contain train/val/test examples")
        idx_train = work.index[work["split"] == "train"].to_numpy()
        idx_val = work.index[work["split"] == "val"].to_numpy()
        idx_test = work.index[work["split"] == "test"].to_numpy()
    else:
        idx_all = np.arange(len(work))
        idx_train, idx_tmp = train_test_split(
            idx_all,
            test_size=float(args.val_size + args.test_size),
            random_state=int(args.seed),
            stratify=work["label"].to_numpy(dtype=int),
        )
        rel_test = float(args.test_size / (args.val_size + args.test_size))
        idx_val, idx_test = train_test_split(
            idx_tmp,
            test_size=rel_test,
            random_state=int(args.seed),
            stratify=work.iloc[idx_tmp]["label"].to_numpy(dtype=int),
        )
        split = np.array([""] * len(work), dtype=object)
        split[idx_train] = "train"
        split[idx_val] = "val"
        split[idx_test] = "test"
        work["split"] = split

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, frame: pd.DataFrame) -> None:
            self.texts = frame["text"].tolist()
            self.targets = frame["target"].tolist()
            self.item_ids = frame["item_id"].astype(str).tolist()

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            enc = tokenizer(self.texts[idx], truncation=True, max_length=int(args.max_length))
            val = self.targets[idx]
            enc["labels"] = float(val) if args.task == "regression" else int(val)
            return enc

    train_ds = TextDataset(work.iloc[idx_train].copy())
    val_ds = TextDataset(work.iloc[idx_val].copy())
    test_ds = TextDataset(work.iloc[idx_test].copy())

    args.outdir.mkdir(parents=True, exist_ok=True)
    num_labels = 1 if args.task == "regression" else 2
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    if args.task == "regression":
        model.config.problem_type = "regression"
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    train_kwargs: Dict[str, Any] = {
        "output_dir": str(args.outdir / "hf_trainer"),
        "per_device_train_batch_size": int(args.train_batch_size),
        "per_device_eval_batch_size": int(args.eval_batch_size),
        "num_train_epochs": float(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
    }
    optional_kwargs: Dict[str, Any] = {
        "overwrite_output_dir": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_rmse" if args.task == "regression" else "eval_accuracy",
        "greater_is_better": False if args.task == "regression" else True,
        "save_total_limit": 1,
        "report_to": [],
    }
    for key, value in optional_kwargs.items():
        if key in ta_params:
            train_kwargs[key] = value
    train_args = TrainingArguments(**train_kwargs)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=(lambda p: compute_regression_metrics(p, float(args.age_cutoff))) if args.task == "regression" else compute_classifier_metrics,
    )
    trainer.train()
    val_out = trainer.predict(val_ds)
    test_out = trainer.predict(test_ds)
    if args.task == "regression":
        val_metrics = compute_regression_metrics((val_out.predictions, val_out.label_ids), float(args.age_cutoff))
        test_metrics = compute_regression_metrics((test_out.predictions, test_out.label_ids), float(args.age_cutoff))
    else:
        val_metrics = compute_classifier_metrics((val_out.predictions, val_out.label_ids))
        test_metrics = compute_classifier_metrics((test_out.predictions, test_out.label_ids))
    trainer.save_model(str(args.outdir / "model"))
    tokenizer.save_pretrained(str(args.outdir / "model"))
    work[["item_id", "story_id", "age_mid", "label", "target", "split"]].to_csv(args.outdir / "split_assignments.csv", index=False)
    summary = {
        "n_items": int(len(work)),
        "split_counts": work["split"].value_counts().to_dict(),
        "age_cutoff": float(args.age_cutoff),
        "model": args.model,
        "task": args.task,
        "target_source": "amazon_age_mid",
        "text_source": args.text_col,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
