from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and standardize an official Qwen-Scope SAE checkpoint.")
    parser.add_argument("--repo", default="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download

    source_path = hf_hub_download(args.repo, filename=f"layer{args.layer}.sae.pt")
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    required = {"W_enc", "W_dec", "b_enc", "b_dec"}
    if not required.issubset(source):
        raise ValueError(f"Unexpected Qwen-Scope checkpoint keys: {sorted(source)}")
    n_features, d_model = source["W_enc"].shape
    if source["W_dec"].shape != (d_model, n_features):
        raise ValueError("Qwen-Scope encoder/decoder dimensions disagree")
    payload = {
        "state_dict": {
            "encoder.weight": source["W_enc"], "encoder.bias": source["b_enc"],
            "decoder.weight": source["W_dec"], "decoder.bias": source["b_dec"],
        },
        "mean": torch.zeros(d_model), "scale": 1.0,
        "config": {"layer": args.layer, "d_model": d_model, "n_features": n_features, "k": 50, "relu": False, "expansion": n_features // d_model, "source": args.repo},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(dir=args.output.parent, suffix=".pt")
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Wrote {args.output}: d_model={d_model} features={n_features} layer={args.layer}")


if __name__ == "__main__":
    main()
