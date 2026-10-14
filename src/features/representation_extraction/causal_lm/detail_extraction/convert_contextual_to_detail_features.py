"""Write the per-detail feature files the probes read, from the contextual shards that already exist.

`extract.py` writes `features_NNN.npz` (`vectors`: n_details x n_layers x D, mean of the detail's tokens at each layer) and
`metadata_NNN.csv`. The global extraction (`extract_global.py`) already stores the same quantity for every detail as
`mean_control_vectors` in `contextual_detail_features_NNN.npz`, read inside the whole untruncated text. This converts one to the
other so layerwise probes (04), behavior-vs-representation (10) and the other consumers can run without a second GPU pass.

Differences from `extract.py` output that matter:
  * no router summaries (router_* columns are empty), so the router experiment (07) still needs a real extract.py run;
  * `video_id` and `compression_ratio` are filled from the cohort when --cohort is given;
  * `token_count` is the detail's token count from the global run.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from shared.io import read_jsonl, write_csv, write_npz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder with contextual_detail_features_*.npz and contextual_detail_metadata_*.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Becomes the detail_extraction/<model_key> folder the probes read")
    parser.add_argument("--model-key", default="gpt_oss_20b")
    parser.add_argument("--cohort", type=Path, default=Path("results/features/summaries/dataset_index/cohort.jsonl"))
    parser.add_argument("--array", default="mean_control_vectors", choices=["mean_control_vectors", "last_token_vectors"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cohort = {}
    if args.cohort and Path(args.cohort).exists():
        for record in read_jsonl(args.cohort):
            cohort[record["record_id"]] = {"video_id": record.get("video_id", ""), "compression_ratio": record.get("compression_ratio")}
    else:
        print(f"cohort not found at {args.cohort}; video_id and compression_ratio will be empty")

    shards = sorted(glob.glob(str(args.input_dir / "contextual_detail_features_*.npz")))
    if not shards:
        raise SystemExit(f"No contextual_detail_features_*.npz under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path_text in shards:
        path = Path(path_text)
        suffix = path.stem.removeprefix("contextual_detail_features_")
        meta_path = args.input_dir / f"contextual_detail_metadata_{suffix}.csv"
        out_features, out_meta = args.output_dir / f"features_{suffix}.npz", args.output_dir / f"metadata_{suffix}.csv"
        if out_features.exists() and out_meta.exists() and not args.overwrite:
            continue
        if not meta_path.exists():
            print(f"skip {suffix}: no metadata")
            continue
        metadata = pd.read_csv(meta_path, low_memory=False)
        with np.load(path, allow_pickle=False) as payload:
            vectors = payload[args.array]
            layers = payload["layers"]
        if len(vectors) != len(metadata):
            raise SystemExit(f"{path.name}: {len(vectors)} vectors but {len(metadata)} metadata rows")
        frame = pd.DataFrame({
            "feature_row": np.arange(len(metadata)),
            "record_id": metadata["record_id"], "record_type": metadata["record_type"], "story_key": metadata["story_key"],
            "video_id": metadata["record_id"].map(lambda r: cohort.get(r, {}).get("video_id", "")),
            "condition": metadata["condition"], "summary_id": metadata["summary_id"],
            "source_age": metadata["source_age"], "target_age": metadata["target_age"],
            "compression_ratio": metadata["record_id"].map(lambda r: cohort.get(r, {}).get("compression_ratio")),
            "detail_index": metadata["detail_index"], "detail_text": metadata["detail_text"],
            "token_count": metadata["detail_token_count"],
            "router_entropy": "[]", "router_top_expert": "[]", "router_top_probability": "[]",
        })
        write_npz(out_features, vectors=vectors.astype(np.float16), layers=layers, model_key=np.asarray(args.model_key))
        write_csv(out_meta, frame)
        written += 1
    print(f"wrote {written} shards to {args.output_dir} ({len(shards)} contextual shards found)")


if __name__ == "__main__":
    main()
