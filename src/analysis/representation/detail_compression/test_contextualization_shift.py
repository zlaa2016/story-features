import importlib

import numpy as np
import pandas as pd

module = importlib.import_module("analysis.representation.detail_compression.contextualization_shift")


def frame(n):
    return pd.DataFrame({
        "story_key": "s", "record_type": "original", "condition": "", "summary_id": 0,
        "source_age": 5.0, "target_age": np.nan, "detail_index": range(n), "tokens": 4.0,
    })


def test_identical_vectors_have_zero_shift_and_centrality_matches_definition():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(5, 2, 8)).astype(np.float32)
    layers = np.array([4, 12])
    table = module.shift_rows("s::original", (frame(5), vectors), (frame(5), vectors.copy()), layers)
    assert len(table) == 10 and np.allclose(table["shift"], 0.0, atol=1e-6)
    unit = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    for layer_position, layer in enumerate(layers):
        sims = unit[:, layer_position] @ unit[:, layer_position].T
        expected = (sims.sum(axis=1) - np.diag(sims)) / 4
        got = table[table.layer.eq(layer)].sort_values("detail_index")["centrality"].to_numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
