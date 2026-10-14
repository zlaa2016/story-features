#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


DEFAULT_MODEL = os.environ.get("OPENAI_DETAIL_SEGMENT_MODEL", "gpt-5.6-luna")

DETAIL_SYSTEM_PROMPT = """
Definition of a detail:
A detail is one minimal coherent piece of explicitly stated information in the text.
A detail may describe an action, state, attribute, relation, cause, outcome, spoken statement, thought, or small connected description.
A detail may consist of one or more sentences, as long as those sentences together express one single coherent piece of information.
A detail is one specific content unit within an ongoing situation.
An event is an ongoing coherent situation, while a detail is one specific content unit within that situation.

Segmentation instructions:
- Keep all text word-for-word exactly as written.
- Do not rewrite, normalize, correct, paraphrase, or remove any words.
- Break the text into distinct segments, where each segment contains one and only one detail.
- A segment may contain one or more sentences if they all belong to the same single coherent detail.
- Do not infer or add anything that is not explicitly stated.
- Preserve all words, including connection words such as "and", "so", "but", and "because".
- If a sentence contains multiple distinct details, split it into multiple segments.
- If multiple consecutive sentences express the same single coherent detail, keep them in the same segment.
- If the same detail is repeated later, include it again there.
- The segmentation must cover the entire text with no missing words.
- After segmenting the entire text, verify that all words from the text appear in the segments and that the total words across segments match the total words in the original text.
""".strip()

DETAIL_SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["segment_id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}


FULLTEXT_PROMPT = """
You will be given a story in full text.

Return valid JSON in this exact format:
{{
  "segments": [
    {{"segment_id": 1, "text": "..."}},
    {{"segment_id": 2, "text": "..."}}
  ]
}}

Story:
{story_text}
""".strip()


SUMMARY_PROMPT = """
You will be given a story summary written for a target child age range of {target_age_range}. Because it is a summary, it may contain simplification, omission, repetition, informal language, missing or incorrect punctuation, or redundant phrasing.

Return valid JSON in this exact format:
{{
  "target_age_range": "{target_age_range}",
  "segments": [
    {{"segment_id": 1, "text": "..."}},
    {{"segment_id": 2, "text": "..."}}
  ]
}}

Summary for target age {target_age_range}:
{summary_text}
""".strip()


@dataclass
class SelectedStory:
    source_age: int
    story_key: str
    video_id: str
    video_title: str
    text_path: str
    original_age_range: str


def load_api_key(key_path: Optional[Path]) -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    if key_path and key_path.exists():
        os.environ["OPENAI_API_KEY"] = key_path.read_text(encoding="utf-8").strip()
        return
    raise FileNotFoundError("OPENAI_API_KEY not set and key file not found")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def local_sentence_segments(text: str) -> List[Dict[str, Any]]:
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text.strip())
        if part.strip()
    ]
    if not parts and text.strip():
        parts = [text.strip()]
    return [
        {"segment_id": index, "text": part}
        for index, part in enumerate(parts, start=1)
    ]


def call_openai_json(prompt: str, model: str, schema_name: str = "detail_segments") -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DETAIL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": DETAIL_SEGMENT_SCHEMA, "strict": True},
        },
    )
    message = response.choices[0].message
    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    return json.loads(message.content or "{}")


def load_selected_stories(api_plan_path: Path) -> List[SelectedStory]:
    plan = pd.read_csv(api_plan_path, low_memory=False)
    keep = (
        plan[["source_age", "story_key", "video_id", "video_title", "text_path", "original_age_range"]]
        .drop_duplicates()
        .sort_values(["source_age", "story_key"])
    )
    first_per_age = keep.groupby("source_age", as_index=False).first()
    rows: List[SelectedStory] = []
    for row in first_per_age.itertuples(index=False):
        rows.append(
            SelectedStory(
                source_age=int(row.source_age),
                story_key=normalize_text(row.story_key),
                video_id=normalize_text(row.video_id),
                video_title=normalize_text(row.video_title),
                text_path=normalize_text(row.text_path),
                original_age_range=normalize_text(row.original_age_range),
            )
        )
    return rows


def load_summary_lookup(summary_outputs_path: Path) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    with summary_outputs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            response = row.get("response")
            if response is None:
                continue
            key = (
                normalize_text(row.get("story_key")),
                int(row.get("target_age")),
                normalize_text(row.get("compression_condition")),
                normalize_text(row.get("age_info_condition")),
            )
            lookup[key] = row
    return lookup


def load_current_summary_rows(summary_outputs_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with summary_outputs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_original_detail_lookup(path: Path) -> dict[str, Dict[str, Any]]:
    lookup: dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            story_key = normalize_text(row.get("story_key"))
            if story_key:
                lookup[story_key] = row
    return lookup


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_processed_original_story_keys(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            story_key = normalize_text(payload.get("story_key"))
            if story_key:
                out.add(story_key)
    return out


def load_processed_summary_keys(path: Path) -> set[tuple[str, int, str, str, int]]:
    out: set[tuple[str, int, str, str, int]] = set()
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            key = (
                normalize_text(payload.get("story_key")),
                int(payload.get("target_age")),
                normalize_text(payload.get("compression_condition")),
                normalize_text(payload.get("age_info_condition")),
                int(payload.get("summary_id", 1)),
            )
            out.add(key)
    return out


def parse_summary_ids(value: str) -> Optional[set[int]]:
    cleaned = normalize_text(value).lower()
    if cleaned == "all":
        return None
    out: set[int] = set()
    for part in cleaned.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    if not out:
        raise ValueError("--summary-ids must be a comma-separated list like 1,2,3 or all")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Detail-segment one story per source age from age-steering run 1.")
    ap.add_argument("--api-plan", type=Path, required=False, default=None)
    ap.add_argument("--summary-outputs", type=Path, required=True)
    ap.add_argument("--original-detail-jsonl", type=Path, default=None, help="Existing original detail-segment JSONL to reuse instead of segmenting originals from text_path.")
    ap.add_argument("--current-summary-format", action="store_true", help="Use current summary output format with story_id, target_age_range, and target_age_label.")
    ap.add_argument("--key-path", type=Path, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--compression-condition", default="30_percent")
    ap.add_argument("--age-info-condition", default="without_original_age")
    ap.add_argument("--summary-id", type=int, default=1)
    ap.add_argument("--summary-ids", default=None, help="For --current-summary-format only: comma-separated summary IDs to segment, e.g. 1,2,3, or all. Defaults to --summary-id.")
    ap.add_argument("--local-summary-segmentation", action="store_true", help="For --current-summary-format only: split summaries locally into sentence segments instead of calling the API.")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--match-source-target-only", action="store_true")
    args = ap.parse_args()

    if not (args.current_summary_format and args.local_summary_segmentation):
        load_api_key(args.key_path)
    current_summary_ids = parse_summary_ids(args.summary_ids or str(args.summary_id))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("results/features/summaries/runs") / f"detail_segment_run1_{stamp}"
    ensure_dir(out_dir)

    manifest_rows: List[Dict[str, Any]] = []
    originals_path = out_dir / "detail_segments_original.jsonl"
    summaries_path = out_dir / "detail_segments_summaries.jsonl"

    if args.overwrite:
        for p in [originals_path, summaries_path]:
            if p.exists():
                p.unlink()

    done_originals = load_processed_original_story_keys(originals_path)
    done_summaries = load_processed_summary_keys(summaries_path)

    if args.current_summary_format:
        if args.original_detail_jsonl is None:
            raise ValueError("--original-detail-jsonl is required with --current-summary-format")
        original_lookup = load_original_detail_lookup(args.original_detail_jsonl)
        summary_rows = load_current_summary_rows(args.summary_outputs)

        for row in summary_rows:
            story_key = normalize_text(row.get("story_id"))
            if not story_key:
                continue
            original = original_lookup.get(story_key)
            if not original:
                continue

            if story_key not in done_originals:
                with originals_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "source_age": row.get("source_age_midpoint"),
                                "story_key": story_key,
                                "video_id": row.get("video_id"),
                                "video_title": row.get("video_title"),
                                "text_path": original.get("cleaned_path", ""),
                                "original_age_range": row.get("source_age_bounds") or row.get("age_range", ""),
                                "segments": original.get("segments", []),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                done_originals.add(story_key)

            response = row.get("response") or {}
            summaries = response.get("summaries") or []
            if not summaries:
                continue
            target_age_range = normalize_text(row.get("target_age_range"))

            for index, candidate in enumerate(summaries, start=1):
                summary_id = int(candidate.get("summary_id", index))
                if current_summary_ids is not None and summary_id not in current_summary_ids:
                    continue
                done_summary_key = (
                    story_key,
                    int(float(row.get("target_age"))),
                    normalize_text(row.get("target_age_label")),
                    target_age_range,
                    summary_id,
                )
                if done_summary_key in done_summaries:
                    continue

                summary_text = normalize_text(candidate.get("summary"))
                if not summary_text:
                    continue

                if args.local_summary_segmentation:
                    summary_payload = {"segments": local_sentence_segments(summary_text)}
                else:
                    summary_payload = call_openai_json(
                        SUMMARY_PROMPT.format(target_age_range=target_age_range, summary_text=summary_text),
                        model=args.model,
                        schema_name="summary_detail_segments",
                    )
                with summaries_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "source_age": row.get("source_age_midpoint"),
                                "target_age": row.get("target_age"),
                                "story_key": story_key,
                                "video_id": row.get("video_id"),
                                "video_title": row.get("video_title"),
                                "compression_condition": "customer_pm3_discrete",
                                "age_info_condition": normalize_text(row.get("target_age_label")),
                                "summary_id": summary_id,
                                "summary_text": summary_text,
                                "segmentation_method": "local_sentence" if args.local_summary_segmentation else "api_detail",
                                "target_age_range": target_age_range,
                                "segments": summary_payload.get("segments", []),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                done_summaries.add(done_summary_key)
                manifest_rows.append(
                    {
                        "source_age": row.get("source_age_midpoint"),
                        "target_age": row.get("target_age"),
                        "story_key": story_key,
                        "video_id": row.get("video_id"),
                        "compression_condition": "customer_pm3_discrete",
                        "age_info_condition": normalize_text(row.get("target_age_label")),
                        "target_age_range": target_age_range,
                        "summary_id": summary_id,
                        "summary_found": True,
                        "summary_segmented": True,
                    }
                )
                if args.sleep > 0:
                    time.sleep(args.sleep)

        pd.DataFrame(manifest_rows).to_csv(out_dir / "detail_segmentation_manifest.csv", index=False)
        print(f"Wrote: {originals_path}")
        print(f"Wrote: {summaries_path}")
        print(f"Wrote: {out_dir / 'detail_segmentation_manifest.csv'}")
        return

    if args.api_plan is None:
        raise ValueError("--api-plan is required unless --current-summary-format is used")
    stories = load_selected_stories(args.api_plan)
    summary_lookup = load_summary_lookup(args.summary_outputs)

    for story in stories:
        if story.story_key not in done_originals:
            story_text = Path(story.text_path).read_text(encoding="utf-8").strip()
            full_payload = call_openai_json(FULLTEXT_PROMPT.format(story_text=story_text), model=args.model, schema_name="fulltext_detail_segments")
            with originals_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "source_age": story.source_age,
                            "story_key": story.story_key,
                            "video_id": story.video_id,
                            "video_title": story.video_title,
                            "text_path": story.text_path,
                            "original_age_range": story.original_age_range,
                            "segments": full_payload.get("segments", []),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            done_originals.add(story.story_key)
            if args.sleep > 0:
                time.sleep(args.sleep)

        for target_age in range(1, 19):
            if args.match_source_target_only and int(target_age) != int(story.source_age):
                continue
            lookup_key = (
                story.story_key,
                int(target_age),
                normalize_text(args.compression_condition),
                normalize_text(args.age_info_condition),
            )
            summary_row = summary_lookup.get(lookup_key)
            done_summary_key = (
                story.story_key,
                int(target_age),
                normalize_text(args.compression_condition),
                normalize_text(args.age_info_condition),
                int(args.summary_id),
            )
            if done_summary_key in done_summaries:
                continue
            if not summary_row:
                manifest_rows.append(
                    {
                        "source_age": story.source_age,
                        "target_age": target_age,
                        "story_key": story.story_key,
                        "video_id": story.video_id,
                        "compression_condition": args.compression_condition,
                        "age_info_condition": args.age_info_condition,
                        "summary_found": False,
                        "summary_segmented": False,
                    }
                )
                continue
            summaries = summary_row.get("response", {}).get("summaries", [])
            summary_text = ""
            for candidate in summaries:
                if int(candidate.get("summary_id", -1)) == args.summary_id:
                    summary_text = normalize_text(candidate.get("summary"))
                    break
            if not summary_text:
                manifest_rows.append(
                    {
                        "source_age": story.source_age,
                        "target_age": target_age,
                        "story_key": story.story_key,
                        "video_id": story.video_id,
                        "compression_condition": args.compression_condition,
                        "age_info_condition": args.age_info_condition,
                        "summary_found": True,
                        "summary_segmented": False,
                    }
                )
                continue
            summary_payload = call_openai_json(
                SUMMARY_PROMPT.format(target_age_range=target_age, summary_text=summary_text),
                model=args.model,
                schema_name="summary_detail_segments",
            )
            with summaries_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "source_age": story.source_age,
                            "target_age": target_age,
                            "story_key": story.story_key,
                            "video_id": story.video_id,
                            "video_title": story.video_title,
                            "compression_condition": args.compression_condition,
                            "age_info_condition": args.age_info_condition,
                            "summary_id": args.summary_id,
                            "summary_text": summary_text,
                            "segments": summary_payload.get("segments", []),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            done_summaries.add(done_summary_key)
            manifest_rows.append(
                {
                    "source_age": story.source_age,
                    "target_age": target_age,
                    "story_key": story.story_key,
                    "video_id": story.video_id,
                    "compression_condition": args.compression_condition,
                    "age_info_condition": args.age_info_condition,
                    "summary_found": True,
                    "summary_segmented": True,
                }
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

    pd.DataFrame(manifest_rows).to_csv(out_dir / "detail_segmentation_manifest.csv", index=False)
    print(f"Wrote: {originals_path}")
    print(f"Wrote: {summaries_path}")
    print(f"Wrote: {out_dir / 'detail_segmentation_manifest.csv'}")


if __name__ == "__main__":
    main()
