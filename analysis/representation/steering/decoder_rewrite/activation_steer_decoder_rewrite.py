#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
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


def merge_story_level_aoa_stats(base: pd.DataFrame, aoa_stats_path: Optional[Path]) -> pd.DataFrame:
    if aoa_stats_path is None or not aoa_stats_path.exists():
        return base
    stats = pd.read_csv(aoa_stats_path, low_memory=False)
    if "story_id" in stats.columns:
        stats["story_id"] = stats["story_id"].map(_normalize_story_id)
    if "video_id" in stats.columns:
        stats["video_id"] = stats["video_id"].astype(str).str.strip()
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


def build_messages(text: str, rewrite_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You rewrite children's stories while preserving plot, characters, and meaning."},
        {"role": "user", "content": f"{rewrite_prompt}\n\nStory:\n{text.strip()}"},
    ]


def resolve_layers(model: Any):
    candidate_paths = (
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("layers",),
    )
    for path in candidate_paths:
        current = model
        try:
            for attr in path:
                current = getattr(current, attr)
        except AttributeError:
            continue
        if current is not None:
            return current
    raise AttributeError(f"Could not locate decoder layers for model class {model.__class__.__name__}")


def format_history(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    lines: list[str] = []
    for m in messages:
        prefix = "Assistant" if m.get("role") == "assistant" else "User"
        lines.append(f"{prefix}: {m.get('content','').strip()}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def normalized_direction(pos_mean: np.ndarray, neg_mean: np.ndarray) -> np.ndarray:
    v = np.asarray(pos_mean, dtype=float) - np.asarray(neg_mean, dtype=float)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


def assign_role_splits(
    labels: np.ndarray,
    seed: int,
    direction_train_frac: float,
    evaluator_val_size: float,
    rewrite_test_size: float,
) -> pd.DataFrame:
    if not (0.0 < direction_train_frac < 1.0):
        raise ValueError("--direction-train-frac must be between 0 and 1")
    if evaluator_val_size <= 0.0 or rewrite_test_size <= 0.0:
        raise ValueError("--evaluator-val-size and --rewrite-test-size must be positive")
    remaining = 1.0 - direction_train_frac
    if evaluator_val_size + rewrite_test_size >= remaining:
        raise ValueError("evaluator/rewrite fractions must fit within the held-out pool")

    idx_all = np.arange(len(labels))
    idx_dir, idx_rest = train_test_split(
        idx_all,
        train_size=float(direction_train_frac),
        random_state=int(seed),
        stratify=labels,
    )
    heldout_frac = float(evaluator_val_size + rewrite_test_size) / remaining
    idx_eval_train, idx_eval_holdout = train_test_split(
        idx_rest,
        test_size=heldout_frac,
        random_state=int(seed),
        stratify=labels[idx_rest],
    )
    rel_rewrite = float(rewrite_test_size / (evaluator_val_size + rewrite_test_size))
    idx_eval_val, idx_rewrite = train_test_split(
        idx_eval_holdout,
        test_size=rel_rewrite,
        random_state=int(seed),
        stratify=labels[idx_eval_holdout],
    )

    direction_split = np.full(len(labels), "heldout", dtype=object)
    direction_split[idx_dir] = "train"

    evaluator_split = np.full(len(labels), "excluded", dtype=object)
    evaluator_split[idx_eval_train] = "train"
    evaluator_split[idx_eval_val] = "val"
    evaluator_split[idx_rewrite] = "test"

    rewrite_split = np.full(len(labels), "not_rewritten", dtype=object)
    rewrite_split[idx_rewrite] = "test"

    legacy_split = evaluator_split.copy()

    return pd.DataFrame(
        {
            "split": legacy_split,
            "direction_split": direction_split,
            "evaluator_split": evaluator_split,
            "rewrite_split": rewrite_split,
        }
    )


def generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int, input_device: Any, hook_layer: Any = None, delta: Any = None) -> str:
    import torch  # type: ignore

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, padding=False)
    input_ids = enc["input_ids"].to(input_device)
    attention_mask = enc["attention_mask"].to(input_device)

    handle = None
    if hook_layer is not None and delta is not None:
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            edited = tensor.clone()
            edited[:, -1, :] = edited[:, -1, :] + delta.to(dtype=edited.dtype, device=edited.device).unsqueeze(0)
            return (edited, *output[1:]) if isinstance(output, tuple) else edited
        handle = hook_layer.register_forward_hook(hook)

    try:
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    finally:
        if handle is not None:
            handle.remove()

    new_tokens = generated[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate unsteered and activation-steered decoder rewrites for younger holdout stories.")
    ap.add_argument("--master", type=Path, default=Path("data/processed/master_stories.parquet"))
    ap.add_argument("--story-vectors", type=Path, required=True)
    ap.add_argument("--word-vectors", type=Path, required=True)
    ap.add_argument("--cleaned-meta", type=Path, default=Path("story_wAmazonData_042626/amazon_youtube_batch1_wid.csv"))
    ap.add_argument("--cleaned-col", type=str, default="cleaned")
    ap.add_argument("--story-aoa-stats", type=Path, default=Path("results/features/eda/corpus_aoa_cutoffs_cleaned_idchecked/story_aoa_stats.csv"))
    ap.add_argument("--text-col", type=str, default="transcript_best")
    ap.add_argument("--id-col", type=str, default="video_id")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--age-cutoff", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--direction-train-frac", type=float, default=0.6)
    ap.add_argument("--evaluator-val-size", type=float, default=0.2)
    ap.add_argument("--rewrite-test-size", type=float, default=0.2)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--max-items", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--rewrite-prompt", type=str, default="Rewrite this story for slightly older children using somewhat more advanced vocabulary and sentence structure, while preserving the same plot, characters, and meaning.")
    ap.add_argument("--reasoning", type=str, default="low", choices=["low", "medium", "high"])
    ap.add_argument("--device-map", type=str, choices=["auto", "none"], default="auto")
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    global torch
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    vec = np.load(args.story_vectors, allow_pickle=True)
    story_vecs = vec["vectors"]
    story_ids_np = [None if x is None or str(x).strip()=="" else str(x) for x in vec["story_id"].tolist()]
    item_ids_np = vec["item_id"].astype(str).tolist()
    ages_np = vec["age_mid"].astype(float)
    dfv = pd.DataFrame({"vec_row": np.arange(story_vecs.shape[0]), "item_id": item_ids_np, "story_id": story_ids_np, "video_id": item_ids_np, "age_mid": ages_np})

    master = read_table(args.master)
    if args.cleaned_meta is not None and args.cleaned_meta.exists():
        master = filter_cleaned_rows(master, args.cleaned_meta, args.cleaned_col)
    cols = [c for c in [args.id_col, "story_id", "age_lo", "age_hi", "age_mid", args.text_col] if c in master.columns]
    work = master[cols].copy().rename(columns={args.id_col: "item_id", args.text_col: "text"})
    work["item_id"] = work["item_id"].astype(str)
    if "story_id" in work.columns:
        work["story_id"] = work["story_id"].map(_normalize_story_id)
    work = merge_story_level_aoa_stats(work, args.story_aoa_stats)
    merged = dfv.merge(work, on="item_id", how="inner", suffixes=("", "_text"))
    if "story_id_text" in merged.columns:
        merged["story_id"] = merged["story_id"].fillna(merged["story_id_text"])
        merged = merged.drop(columns=["story_id_text"])
    if "age_mid_text" in merged.columns:
        merged["age_mid"] = pd.to_numeric(merged["age_mid"], errors="coerce").fillna(pd.to_numeric(merged["age_mid_text"], errors="coerce"))
        merged = merged.drop(columns=["age_mid_text"])
    merged["text"] = merged["text"].astype(str).fillna("").str.strip()
    merged["age_mid"] = pd.to_numeric(merged["age_mid"], errors="coerce")
    merged = merged[merged["text"].ne("") & merged["age_mid"].notna()].copy()
    merged["age_label"] = (merged["age_mid"] > float(args.age_cutoff)).astype(int)
    merged = merged.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    role_splits = assign_role_splits(
        merged["age_label"].to_numpy(dtype=int),
        seed=int(args.seed),
        direction_train_frac=float(args.direction_train_frac),
        evaluator_val_size=float(args.evaluator_val_size),
        rewrite_test_size=float(args.rewrite_test_size),
    )
    merged = pd.concat([merged, role_splits], axis=1)
    holdout = merged[(merged["rewrite_split"] == "test") & (merged["age_label"] == 0)].copy()
    if args.max_items > 0:
        holdout = holdout.head(int(args.max_items)).copy()

    word_npz = np.load(args.word_vectors, allow_pickle=True)
    word_vecs = word_npz["vectors"]
    word_labels = word_npz["aoa_label"].astype(int)
    if args.layer < 0 or args.layer >= word_vecs.shape[1]:
        raise ValueError(f"Layer {args.layer} out of range for word vectors shape {word_vecs.shape}")
    v = normalized_direction(word_vecs[word_labels == 1, args.layer, :].mean(axis=0), word_vecs[word_labels == 0, args.layer, :].mean(axis=0))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.device_map == "auto":
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        try:
            input_device = model.get_input_embeddings().weight.device
        except Exception:
            input_device = next(model.parameters()).device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", trust_remote_code=True)
        model.to(device)
        input_device = device
    model.eval()
    layers = resolve_layers(model)
    layer_idx = args.layer if args.layer >= 0 else len(layers) + args.layer
    hook_layer = layers[layer_idx]
    delta = torch.as_tensor(float(args.alpha) * v, dtype=torch.float32, device=input_device)

    rows: List[Dict[str, Any]] = []
    for row in holdout.itertuples(index=False):
        prompt = format_history(tokenizer, build_messages(str(row.text), args.rewrite_prompt))
        baseline = generate(model, tokenizer, prompt, args.max_new_tokens, input_device)
        steered = generate(model, tokenizer, prompt, args.max_new_tokens, input_device, hook_layer=hook_layer, delta=delta)
        rows.append(
            {
                "item_id": row.item_id,
                "story_id": row.story_id,
                "video_id": row.video_id,
                "age_mid": float(row.age_mid),
                "split": row.split,
                "direction_split": row.direction_split,
                "evaluator_split": row.evaluator_split,
                "rewrite_split": row.rewrite_split,
                "layer": int(args.layer),
                "alpha": float(args.alpha),
                "model": args.model,
                "rewrite_prompt": args.rewrite_prompt,
                "original_text": str(row.text),
                "baseline_rewrite": baseline,
                "rewritten_text": steered,
            }
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.outdir / "rewrites.csv", index=False)
    merged[
        [
            "item_id",
            "story_id",
            "video_id",
            "age_mid",
            "age_label",
            "split",
            "direction_split",
            "evaluator_split",
            "rewrite_split",
        ]
    ].to_csv(
        args.outdir / "split_assignments.csv", index=False
    )
    summary = {
        "n_items": int(len(rows)),
        "n_total_stories": int(len(merged)),
        "direction_split_counts": merged["direction_split"].value_counts().to_dict(),
        "evaluator_split_counts": merged["evaluator_split"].value_counts().to_dict(),
        "rewrite_split_counts": merged["rewrite_split"].value_counts().to_dict(),
        "model": args.model,
        "layer": int(args.layer),
        "alpha": float(args.alpha),
        "age_cutoff": float(args.age_cutoff),
        "max_new_tokens": int(args.max_new_tokens),
        "rewrite_prompt": args.rewrite_prompt,
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
