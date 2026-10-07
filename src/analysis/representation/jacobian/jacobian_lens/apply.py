from __future__ import annotations

import argparse
import os
import sys
from itertools import islice
from pathlib import Path
from typing import Any

from shared.io import append_jsonl, read_jsonl


def detail_positions(tokenizer: Any, text: str, segments: list[dict[str, Any]], max_length: int) -> list[tuple[int, int]]:
    encoded = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length)
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    lowered = text.casefold()
    cursor = 0
    positions: list[tuple[int, int]] = []
    for detail_index, segment in enumerate(segments):
        value = str(segment.get("text", "")).strip()
        start = text.find(value, cursor)
        if start < 0:
            start = lowered.find(value.casefold(), cursor)
        if start < 0:
            continue
        end = start + len(value)
        token_indices = [index for index, (left, right) in enumerate(offsets) if right > start and left < end and right > left]
        if token_indices:
            positions.append((detail_index, token_indices[len(token_indices) // 2]))
        cursor = end
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a fitted J-lens at each detail midpoint.")
    parser.add_argument("--cohort", type=Path, default=Path("results/features/summaries/dataset_index/cohort.jsonl"))
    parser.add_argument("--lens", type=Path, default=Path("results/analysis/representation/jacobian/jacobian_lens/jacobian_lens.pt"))
    parser.add_argument("--jlens-repo", type=Path, default=Path(os.environ.get("JLENS_REPO", f"/scratch/{os.environ.get('USER', 'user')}/tools/jacobian-lens")))
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(args.jlens_repo))
    import jlens

    stop = None if args.limit <= 0 else args.start + args.limit
    records = list(islice(read_jsonl(args.cohort), args.start, stop))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(str(args.lens))
    completed = {(row.get("record_id"), int(row.get("detail_index", -1))) for row in read_jsonl(args.output)} if args.output.exists() else set()

    for record_number, record in enumerate(records, start=1):
        pairs = [(detail, position) for detail, position in detail_positions(tokenizer, record["text"], record.get("segments") or [], args.max_seq_len) if (record["record_id"], detail) not in completed]
        if not pairs:
            continue
        detail_ids, positions = zip(*pairs)
        lens_logits, _model_logits, input_ids = lens.apply(model, record["text"], positions=list(positions), max_seq_len=args.max_seq_len)
        output_rows = []
        for pair_index, detail_index in enumerate(detail_ids):
            for layer, logits in lens_logits.items():
                values, token_ids = torch.topk(logits[pair_index], args.top_k)
                output_rows.append({
                    "record_id": record["record_id"], "story_key": record["story_key"], "record_type": record["record_type"],
                    "condition": record.get("condition", ""), "summary_id": record.get("summary_id", 0),
                    "source_age": record.get("source_age"), "target_age": record.get("target_age"),
                    "detail_index": detail_index, "position": positions[pair_index], "layer": int(layer),
                    "token_ids": token_ids.cpu().tolist(), "tokens": tokenizer.convert_ids_to_tokens(token_ids.cpu().tolist()),
                    "logits": values.float().cpu().tolist(), "input_token_id": int(input_ids[0, positions[pair_index]]),
                })
        append_jsonl(args.output, output_rows)
        print(f"[{record_number}/{len(records)}] {record['record_id']} details={len(pairs)}", flush=True)


if __name__ == "__main__":
    main()
