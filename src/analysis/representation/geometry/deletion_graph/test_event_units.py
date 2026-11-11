from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.representation.geometry.deletion_graph import compare_units, event_batch
from features.representation_extraction.causal_lm.detail_extraction import event_segments
from features.representation_extraction.causal_lm.detail_extraction.extract_global import locate_segment_spans

EVENT_FILE = """# 000552_Title

**4 sentences | 14 words | 2 events**

### Event 0 — sents 0–1 (2 sentences)
**exposition**
> [0] The bears visit the farm.
> [1] They feed the ducks.

### Event 1 — sents 2–3 (2 sentences)
> [2] The ducks follow them home.
> [3] Mama says no more food.
"""


class EventSegmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "000552_Title.md").write_text(EVENT_FILE, encoding="utf-8")
        (self.dir / "000552_Other.md").write_text(EVENT_FILE.replace("bears", "cats"), encoding="utf-8")
        (self.dir / "notes.md").write_text("not a story", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_parse_groups_sentences_by_event(self) -> None:
        events = event_segments.parse_event_file(self.dir / "000552_Title.md")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1], ["The ducks follow them home.", "Mama says no more food."])

    def test_missing_sentence_is_an_error(self) -> None:
        broken = self.dir / "000999_Broken.md"
        broken.write_text(EVENT_FILE.replace("> [1] They feed the ducks.\n", ""), encoding="utf-8")
        with self.assertRaises(ValueError):
            event_segments.parse_event_file(broken)

    def test_files_are_keyed_by_story_id_and_first_name_wins(self) -> None:
        files = event_segments.event_files(self.dir)
        self.assertEqual(list(files), [552])
        self.assertEqual(files[552].name, "000552_Other.md")

    def test_record_spans_cover_every_event_in_order(self) -> None:
        record = event_segments.event_record("vid__story01", 552, self.dir / "000552_Title.md", "vid")
        spans = locate_segment_spans(record["text"], record["segments"])
        self.assertTrue(all(spans))
        self.assertEqual(record["text"][spans[0][0]:spans[0][1]], "The bears visit the farm. They feed the ducks.")
        self.assertLess(spans[0][1], spans[1][0])
        self.assertEqual(record["record_type"], "event")

    def test_records_skip_stories_without_a_file(self) -> None:
        records = event_segments.event_records(self.dir, [("a__story01", 552, "a"), ("b__story01", 7, "b")])
        self.assertEqual([r["story_key"] for r in records], ["a__story01"])


class BatchTests(unittest.TestCase):
    def test_story_listed_at_every_covered_age_within_2_12(self) -> None:
        master = pd.DataFrame({
            "story_id": [1, 2, 3, 4], "video_id": ["a", "b", "c", "d"], "age_lo": [4.0, 0.0, 13.0, np.nan],
            "age_hi": [6.0, 3.0, 16.0, np.nan], "yt_view_count": [10, 30, 20, 5],
        })
        files = {1: Path("000001_a.md"), 2: Path("000002_b.md"), 3: Path("000003_c.md"), 4: Path("000004_d.md")}
        subset, membership = event_batch.build_batch(master, files)
        self.assertEqual(list(subset["story_key"]), ["b__story01", "a__story01"])  # most viewed first; 13-16 and no-age dropped
        self.assertEqual(sorted(membership[membership.story_key == "a__story01"]["age"]), [4, 5, 6])
        self.assertEqual(sorted(membership[membership.story_key == "b__story01"]["age"]), [2, 3])


class CompareTests(unittest.TestCase):
    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(0)
        n = 40
        keys = [f"s{i}" for i in range(n)]
        value = rng.normal(size=n)
        events = pd.DataFrame({"story_key": keys, "token_count": 500, "n_details": 9, "mean_weight": value})
        details = pd.DataFrame({"story_key": keys, "token_count": 510, "n_details": 40, "mean_weight": value + 1 + 0.1 * rng.normal(size=n)})
        details.loc[0, "token_count"] = 5000  # an event file of another story
        ages = pd.DataFrame({"story_key": keys, "age_mid": np.arange(n) / 4})
        return events, details, ages

    def test_mismatched_texts_are_dropped_and_agreement_is_found(self) -> None:
        table, overview = compare_units.compare(*self.frames())
        self.assertEqual(int(overview["dropped_text_mismatch"][0]), 1)
        row = table[table.measure == "mean_weight"].iloc[0]
        self.assertGreater(row["rho_units"], 0.9)
        self.assertGreater(row["median_detail"], row["median_event"])
        self.assertLess(row["wilcoxon_p"], 0.001)
        self.assertEqual(row["n"], 39)

    def test_bookkeeping_columns_are_not_measures(self) -> None:
        events, _, _ = self.frames()
        self.assertEqual(compare_units.measure_columns(events), ["mean_weight"])


if __name__ == "__main__":
    unittest.main()
