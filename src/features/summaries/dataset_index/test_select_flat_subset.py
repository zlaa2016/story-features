import importlib

import pandas as pd

select = importlib.import_module("features.summaries.dataset_index.select_flat_subset")


def make_metadata():
    rows = []
    for story, age in [("a", 3.2), ("b", 3.7), ("c", 3.9), ("d", 8.1), ("e", 8.4), ("f", 8.8)]:
        rows.append({"record_id": f"{story}::original", "record_type": "original", "story_key": story, "source_age": age, "condition": "", "summary_id": 0})
        for condition in ["younger", "actual", "older"]:
            if story == "f" and condition == "younger":
                continue
            for summary_id in (2, 1):
                rows.append({"record_id": f"{story}::{condition}::{summary_id}", "record_type": "summary", "story_key": story, "source_age": age, "condition": condition, "summary_id": summary_id})
    return pd.DataFrame(rows)


def test_complete_stories_and_lowest_summary_id():
    stories = select.complete_stories(make_metadata())
    assert sorted(stories["story_key"]) == ["a", "b", "c", "d", "e"]
    assert stories.loc[stories.story_key.eq("a"), "actual_record_id"].item() == "a::actual::1"


def test_flatten_equalises_age_bins():
    stories = select.complete_stories(make_metadata())
    subset = select.flatten_by_age(stories, bin_width=1.0, per_bin=0, min_bin=2, seed=3)
    assert subset["age_bin"].value_counts().to_dict() == {3.0: 2, 8.0: 2}


def test_require_method_keeps_only_matching_summaries():
    metadata = make_metadata()
    metadata["segmentation_method"] = ["" if r == "original" else "api_detail" for r in metadata.record_type]
    metadata.loc[metadata.record_id.eq("a::older::1") | metadata.record_id.eq("a::older::2"), "segmentation_method"] = "local_sentence"
    assert "a" in select.complete_stories(metadata)["story_key"].tolist()
    strict = select.complete_stories(metadata, require_method="api_detail")
    assert "a" not in strict["story_key"].tolist() and "b" in strict["story_key"].tolist()
    audit = select.method_audit(metadata)
    assert audit.set_index("segmentation_method").loc["local_sentence", "summaries"] == 2
