"""Group cohort records into stories, without choosing which stories to keep.

Every original with at least one summary in a confirmatory condition is used, with every summary it has. Stories
are not balanced across age and are not required to have all three summaries; the mixed models handle unbalanced
stories. The only exclusion is by *unit of segmentation*: summaries segmented at sentence level are a separate
measurement and are skipped by the detail-level analyses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

CONDITIONS = ("younger", "actual", "older")
NOT_DETAIL_LEVEL = ("local_sentence",)


def excluded_record_ids(cohort_metadata: Path | None, methods: Iterable[str] = NOT_DETAIL_LEVEL) -> set[str]:
    """Record ids of summaries whose segmentation method is in `methods` (read from cohort_metadata.csv)."""
    methods = tuple(m for m in methods if m)
    if not methods or cohort_metadata is None or not Path(cohort_metadata).exists():
        return set()
    meta = pd.read_csv(cohort_metadata, usecols=lambda c: c in {"record_id", "segmentation_method"}, low_memory=False)
    if "segmentation_method" not in meta:
        return set()
    return set(meta.loc[meta["segmentation_method"].fillna("").isin(methods), "record_id"])


def story_units(records: Iterable[dict[str, Any]], exclude_ids: Iterable[str] = (), conditions: Iterable[str] = CONDITIONS) -> list[dict[str, Any]]:
    """[{story_key, source_age, original, summaries[]}], ordered by story; stories without a usable summary are dropped."""
    exclude, wanted = set(exclude_ids), tuple(conditions)
    originals: dict[str, dict[str, Any]] = {}
    summaries: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("record_type") == "original":
            originals[str(record["story_key"])] = record
        elif record.get("record_type") == "summary" and record.get("condition") in wanted and record["record_id"] not in exclude:
            summaries.setdefault(str(record["story_key"]), []).append(record)
    units = []
    for story_key in sorted(originals):
        found = summaries.get(story_key)
        if not found:
            continue
        found.sort(key=lambda r: (wanted.index(r["condition"]), _summary_number(r.get("summary_id"))))
        original = originals[story_key]
        units.append({"story_key": story_key, "source_age": original.get("source_age"), "original": original, "summaries": found})
    return units


def _summary_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
