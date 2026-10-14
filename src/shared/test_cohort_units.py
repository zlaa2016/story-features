import pandas as pd

from shared.cohort_units import excluded_record_ids, story_units


def records():
    out = [{"record_id": f"{s}::original", "record_type": "original", "story_key": s, "source_age": age} for s, age in [("a", 3.0), ("b", 8.0), ("c", 5.0)]]
    out += [{"record_id": "a::actual::1", "record_type": "summary", "story_key": "a", "condition": "actual", "summary_id": 1},
            {"record_id": "a::older::2", "record_type": "summary", "story_key": "a", "condition": "older", "summary_id": 2},
            {"record_id": "a::older::1", "record_type": "summary", "story_key": "a", "condition": "older", "summary_id": 1},
            {"record_id": "b::actual::1", "record_type": "summary", "story_key": "b", "condition": "actual", "summary_id": 1},
            {"record_id": "b::older::1", "record_type": "summary", "story_key": "b", "condition": "weird_grid", "summary_id": 1}]
    return out


def test_keeps_every_story_with_a_summary_and_every_summary():
    units = story_units(records())
    assert [u["story_key"] for u in units] == ["a", "b"]                       # c has no summary
    assert [s["record_id"] for s in units[0]["summaries"]] == ["a::actual::1", "a::older::1", "a::older::2"]
    assert [s["record_id"] for s in units[1]["summaries"]] == ["b::actual::1"]  # unbalanced stories stay; non-confirmatory condition dropped


def test_no_age_balancing_or_completeness_requirement():
    ages = [u["source_age"] for u in story_units(records())]
    assert ages == [3.0, 8.0]


def test_sentence_level_summaries_are_excluded_only_by_method(tmp_path):
    meta = pd.DataFrame({"record_id": ["a::actual::1", "a::older::1", "a::older::2"], "segmentation_method": ["api_detail", "local_sentence", ""]})
    path = tmp_path / "cohort_metadata.csv"
    meta.to_csv(path, index=False)
    skip = excluded_record_ids(path)
    assert skip == {"a::older::1"}
    units = story_units(records(), skip)
    assert [s["record_id"] for s in units[0]["summaries"]] == ["a::actual::1", "a::older::2"]
