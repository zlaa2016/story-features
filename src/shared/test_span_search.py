import numpy as np
import pandas as pd

from shared.span_search import pooled_span_vectors, ratio_rows, search_story, span_list, summary_row, unit_rows


def test_pooled_span_vectors_match_manual_weighted_mean():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(6, 4))
    weights = np.array([1, 2, 3, 4, 5, 6])
    spans = span_list(6, 3)
    pooled = pooled_span_vectors(vectors, weights, spans)
    for span, vector in zip(spans, pooled):
        sl = slice(span.start, span.end + 1)
        expected = (vectors[sl] * weights[sl, None]).sum(0) / weights[sl].sum()
        np.testing.assert_allclose(vector, expected, rtol=1e-5, atol=1e-6)


def test_search_recovers_planted_spans_and_ratios():
    rng = np.random.default_rng(1)
    source = unit_rows(rng.normal(size=(12, 64)))
    planted = [(0, 1), (3, 5), (8, 8)]
    summary = unit_rows(np.stack([source[a : b + 1].mean(0) for a, b in planted]))
    spans = span_list(12, 6)
    span_vectors = unit_rows(pooled_span_vectors(source, np.ones(12), spans))
    chosen, scores = search_story(
        summary @ span_vectors.T, spans, length_penalty=0.0, overlap_penalty=0.3, reversal_penalty=2.5, top_k=32
    )
    assert [(spans[i].start, spans[i].end) for i in chosen] == planted
    table = ratio_rows(
        chosen, spans, scores, source_words=np.full(12, 10.0), summary_words=np.array([10.0, 10.0, 10.0])
    )
    assert table["span_len"].tolist() == [2, 3, 1]
    assert table["span_share"].round(4).tolist() == [round(2 / 12, 4), round(3 / 12, 4), round(1 / 12, 4)]
    assert table["word_ratio"].tolist() == [2.0, 3.0, 1.0]
    row = summary_row(table, chosen, spans, 12)
    assert row["detail_compression"] == 4.0 and row["covered_fraction"] == 6 / 12 and row["inversion_rate"] == 0.0
