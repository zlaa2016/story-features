from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

from shared.io import parse_int_list, read_jsonl, write_json
from shared.research import train_story_keys


def import_jlens(path: Path):
    sys.path.insert(0, str(path))
    try:
        import jlens
    except ImportError as exc:
        raise RuntimeError(f"Clone anthropics/jacobian-lens into {path} first") from exc
    return jlens


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit one prompt shard of the corpus-averaged Jacobian lens.")
    parser.add_argument("--cohort", type=Path, default=Path("results/features/summaries/dataset_index/cohort.jsonl"))
    parser.add_argument("--jlens-repo", type=Path, default=Path(os.environ.get("JLENS_REPO", f"/scratch/{os.environ.get('USER', 'user')}/tools/jacobian-lens")))
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--source-layers", default="8,16,24,30")
    parser.add_argument("--prompt-count", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--dim-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--story-split", type=Path, default=Path("results/analysis/representation/probes/layerwise_detail/story_split.csv"))
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    jlens = import_jlens(args.jlens_repo)
    training = train_story_keys(args.story_split)
    rows = [row for row in read_jsonl(args.cohort) if row.get("record_type") == "original" and str(row.get("story_key")) in training]
    random.Random(args.seed).shuffle(rows)
    prompts = [str(row["text"]) for row in rows[: args.prompt_count]]
    prompts = prompts[args.start : args.start + args.limit]
    if not prompts:
        raise ValueError("This shard selected no prompts")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    model = jlens.from_hf(hf_model, tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix(".checkpoint.pt")
    lens = jlens.fit(
        model,
        prompts=prompts,
        source_layers=parse_int_list(args.source_layers),
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=str(checkpoint),
        checkpoint_every=1,
        resume=True,
    )
    file_descriptor, temporary_name = tempfile.mkstemp(dir=args.output.parent, suffix=".pt")
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        lens.save(str(temporary))
        temporary.replace(args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    write_json(args.output.with_suffix(".json"), {"model": args.model_name, "start": args.start, "prompts": lens.n_prompts, "layers": lens.source_layers, "fit_split": "train", "story_split": str(args.story_split), "seed": args.seed})
    print(f"Wrote {args.output} from {lens.n_prompts} prompts")


if __name__ == "__main__":
    main()
