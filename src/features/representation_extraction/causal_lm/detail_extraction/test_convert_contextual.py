import importlib
import json
import sys

import numpy as np
import pandas as pd

convert = importlib.import_module("features.representation_extraction.causal_lm.detail_extraction.convert_contextual_to_detail_features")
probes = importlib.import_module("analysis.representation.probes.layerwise_detail.run")


def test_converted_shards_load_in_the_layerwise_probe_loader(tmp_path, monkeypatch):
    src, out = tmp_path / "ctx", tmp_path / "out"
    src.mkdir()
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({
        "record_id": ["s::original"] * 3 + ["s::actual::1"] * 2, "record_type": ["original"] * 3 + ["summary"] * 2, "story_key": "s",
        "condition": [""] * 3 + ["actual"] * 2, "summary_id": [0] * 3 + [1] * 2, "source_age": 5.0, "target_age": [np.nan] * 3 + [5.0] * 2,
        "detail_index": [0, 1, 2, 0, 1], "detail_text": list("abcde"), "detail_token_count": 4,
    })
    meta.to_csv(src / "contextual_detail_metadata_000.csv", index=False)
    vectors = rng.normal(size=(5, 3, 8)).astype(np.float16)
    np.savez(src / "contextual_detail_features_000.npz", mean_control_vectors=vectors, last_token_vectors=vectors * 2, layers=np.array([4, 12, 23], dtype=np.int16))
    cohort = tmp_path / "cohort.jsonl"
    cohort.write_text("\n".join(json.dumps({"record_id": r, "video_id": "v", "compression_ratio": 0.25}) for r in ["s::original", "s::actual::1"]) + "\n")
    monkeypatch.setattr(sys, "argv", ["x", "--input-dir", str(src), "--output-dir", str(out), "--cohort", str(cohort)])
    convert.main()
    loaded, metadata, layers = probes.load_shards(out)
    assert loaded.shape == (5, 3, 8) and layers.tolist() == [4, 12, 23]
    np.testing.assert_allclose(loaded, vectors.astype(np.float32))
    assert metadata["compression_ratio"].eq(0.25).all() and metadata["video_id"].eq("v").all()
    pooled, records = probes.aggregate_records(loaded, metadata)
    assert len(records) == 2 and pooled.shape == (2, 3, 8)
