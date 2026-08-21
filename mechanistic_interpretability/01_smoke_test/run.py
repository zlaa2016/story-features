from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechanistic_interpretability.shared.io import read_jsonl, write_json
from mechanistic_interpretability.shared.modeling import load_model, render_prompt, resolve_layers
from mechanistic_interpretability.shared.prompts import strict_summary_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model loading, hidden states, routing, and generation.")
    parser.add_argument("--cohort", type=Path, default=Path("mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl"))
    parser.add_argument("--model-key", default="gpt_oss_20b", choices=["qwen3_8b", "gpt_oss_20b", "qwen3_30b_a3b"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", type=Path, default=Path("mechanistic_interpretability/outputs/01_smoke_test/report.json"))
    args = parser.parse_args()

    import torch

    original = next(row for row in read_jsonl(args.cohort) if row.get("record_type") == "original")
    model, tokenizer, device, spec = load_model(
        args.model_key,
        model_name=args.model_name,
        dtype=args.dtype,
        output_router_logits=True,
    )
    layers = resolve_layers(model, spec)
    encoded = tokenizer(original["text"], return_tensors="pt", truncation=True, max_length=1024)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    forward_kwargs = {"output_hidden_states": True, "use_cache": False}
    if hasattr(model.config, "output_router_logits"):
        forward_kwargs["output_router_logits"] = True
    with torch.inference_mode():
        outputs = model(**encoded, **forward_kwargs)

    target_words = max(30, round(int(original["word_count"]) * 0.30))
    prompt = render_prompt(
        tokenizer,
        strict_summary_prompt(
            original["text"],
            target_words=target_words,
            target_age=original.get("source_age_range", ""),
            condition="actual",
        ),
    )
    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    prompt_inputs = {key: value.to(device) for key, value in prompt_inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **prompt_inputs,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, prompt_inputs["input_ids"].shape[1] :]
    router_logits = getattr(outputs, "router_logits", None)
    report = {
        "model_key": args.model_key,
        "model_name": args.model_name or spec.name,
        "story_key": original["story_key"],
        "n_layers": len(layers),
        "hidden_state_tensors": len(outputs.hidden_states or []),
        "hidden_shape_last": list(outputs.hidden_states[-1].shape),
        "router_available": router_logits is not None,
        "router_tensors": len(router_logits) if isinstance(router_logits, (tuple, list)) else int(router_logits is not None),
        "generation_preview": tokenizer.decode(new_tokens, skip_special_tokens=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
