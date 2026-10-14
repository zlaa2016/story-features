#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # src/, so `shared` and `data_pipeline` resolve when run directly

from shared.age_utils import format_age_range, normalize_age_text, parse_age_bounds, parse_age_midpoint
from data_pipeline.cleaning.transcript_cleanup.run_transcript_cleanup_segmentation import (
    CLEAN_AND_PUNCTUATE_PROMPT,
    StoryRow,
    clean_input_text,
    load_api_key,
    load_labcomputer_rows,
    load_manifest_video_ids,
    load_title_success_video_ids,
    normalize_clean_stories,
    normalize_text,
    record_key,
    load_jsonl_cache,
    append_jsonl,
    word_count,
    write_cleaned_txt,
)


DEFAULT_CLEAN_MODEL = "gpt-5.4-nano"
DEFAULT_DETAIL_MODEL = "gpt-5.6-luna"
CLEAN_PROMPT_VERSION = "v2"
DETAIL_PROMPT_VERSION = "v1"
SUMMARY_CLEAN_PROMPT_VERSION = "v2"

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


DETAIL_JSON_SCHEMA = {
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


CLEAN_WITH_SUMMARY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "stories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "story_title_guess": {"type": "string"},
                    "cleaned_story_verbatim": {"type": "string"},
                    "punctuated_story": {"type": "string"},
                    "age_steered_summaries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "target_age_label": {"type": "string"},
                                "target_age_setting": {"type": "string"},
                                "summary": {"type": "string"},
                            },
                            "required": ["target_age_label", "target_age_setting", "summary"],
                            "additionalProperties": False,
                        },
                    },
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
                    },
                },
                "required": [
                    "story_title_guess",
                    "cleaned_story_verbatim",
                    "punctuated_story",
                    "age_steered_summaries",
                    "segments",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["stories"],
    "additionalProperties": False,
}


CLEAN_AND_SUMMARIZE_PROMPT = """
You will be given an automatically generated transcript of a YouTube video. These are usually direct readings of children's books or story texts, not personal recalls or free retellings.

Your tasks:
1. Identify whether the transcript contains one story, multiple stories, or no real story at all.
2. Remove only content that is clearly NOT part of the story itself.
3. Restore readable punctuation and sentence boundaries for the story-only text.
4. For each recovered story, segment the cleaned/punctuated full story into detail segments.
5. For each recovered story, create exactly three age-steered summaries:
   - `younger`: for children age {younger_target_age}
   - `actual`: for children age {actual_target_age_range}
   - `older`: for children age {older_target_age}

Remove:
- greetings, channel promotion, title/author introductions
- "the end", "thanks for watching", subscribe requests, link mentions
- narrator commentary that is not part of the story
- subtitle headers like "Kind: captions Language: en"
- sound-effect markers like [Music], [Applause] when they are not story text
- framing/setup lines that introduce the moral, lesson, character identity, or theme before the actual story begins
- promotional or branded story framing when it functions as series intro rather than story content

Important rule for all tasks:
- Do NOT add any new words, facts, events, names, or interpretations.
- You may only remove non-story material, add punctuation/capitalization, and split text into segments.
- When uncertain whether a line is part of the story, prefer keeping it.

Story splitting rules:
- Some videos may contain no real story.
- Some videos may contain non-narrative children's books.
- Some videos may contain multiple separate stories.
- If the video contains one clear story, return exactly one item in `stories`.
- If the video contains multiple distinct stories, chapters, or anthology entries, split them into separate `stories` items in chronological order.
- Do NOT split a single continuous story into multiple items just because it has several scenes.
- If a transcript contains almost no recoverable story content, return an empty `stories` list.

Requirements for `cleaned_story_verbatim`:
- Keep ONLY the story content for that story.
- Keep the surviving story words in their original order.
- Do NOT add new story content.
- Do NOT rewrite or paraphrase the story.
- Do NOT summarize the story.

Requirements for `punctuated_story`:
- First clean away clearly non-story content, then punctuate the remaining story text.
- Use the same story content after cleaning.
- Add punctuation, capitalization, and sentence boundaries.
- Do NOT add or remove story facts.
- Keep the wording as close as possible while making it readable.

Requirements for each item in `age_steered_summaries`:
- Return exactly three items, with `target_age_label` values `younger`, `actual`, and `older`.
- `younger` must use target age setting `{younger_target_age}`.
- `actual` must use target age setting `{actual_target_age_range}`.
- `older` must use target age setting `{older_target_age}`.
- Each summary should preserve approximately 30% of the length of the cleaned story text.
- Each summary should still include the beginning, middle, and end of the story arc.
- Include key events in sequence and the main characters/actions.
- Preserve cause-and-effect relationships and any explicit central message.
- Use only the recovered story content; do not add opinions, interpretations, motives, morals, or facts not present in the story.
- Do not use bullet points.

Requirements for `segments`:
- Segment the `punctuated_story` into minimal coherent detail units.
- A detail is one explicitly stated action, state, attribute, relation, cause, outcome, spoken statement, thought, or small connected description.
- Keep all segment text word-for-word exactly as it appears in `punctuated_story`.
- Do not rewrite, normalize, correct, paraphrase, infer, or add words.
- If a sentence contains multiple distinct details, split it into multiple segments.
- The segments must cover the full `punctuated_story` in order with no missing story text.

Return valid JSON with exactly this structure:
{{
  "stories": [
    {{
      "story_title_guess": "...",
      "cleaned_story_verbatim": "...",
      "punctuated_story": "...",
      "age_steered_summaries": [
        {{"target_age_label": "younger", "target_age_setting": "{younger_target_age}", "summary": "..."}},
        {{"target_age_label": "actual", "target_age_setting": "{actual_target_age_range}", "summary": "..."}},
        {{"target_age_label": "older", "target_age_setting": "{older_target_age}", "summary": "..."}}
      ],
      "segments": [
        {{"segment_id": 1, "text": "..."}},
        {{"segment_id": 2, "text": "..."}}
      ]
    }}
  ]
}}

Transcript:
{story}
""".strip()


DETAIL_PROMPT = """
You will be given a story.

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


@dataclass
class DetailRow:
    story_id: str
    root_video_id: str
    video_title: str
    source_file: str
    substory_index: int
    n_substories: int
    story_title_guess: str
    cleaned_story_verbatim: str
    punctuated_story: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_existing_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path, low_memory=False)
    records = df.to_dict(orient="records")
    cleaned: List[Dict[str, Any]] = []
    for record in records:
        cleaned.append({k: v for k, v in record.items() if not (isinstance(v, float) and pd.isna(v))})
    return cleaned


def upsert_manifest_row(manifest_rows: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
    row_key = normalize_text(row.get("story_key")) or normalize_text(row.get("video_id"))
    for idx, existing in enumerate(manifest_rows):
        existing_key = normalize_text(existing.get("story_key")) or normalize_text(existing.get("video_id"))
        if existing_key == row_key:
            manifest_rows[idx] = row
            return
    manifest_rows.append(row)


def story_key(video_id: str, substory_index: int) -> str:
    return f"{video_id}__story{substory_index:02d}"


def extract_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def extract_json_candidate(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return "{}"
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def call_openai_json(prompt: str, model: str, *, schema_name: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_prompt = prompt if attempt == 1 else f"{prompt}\n\nReturn valid JSON only. Do not include any prose."
        messages.append({"role": "user", "content": user_prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": DETAIL_JSON_SCHEMA if schema_name.startswith("detail") else None, "strict": True},
            } if schema_name.startswith("detail") else {"type": "json_object"},
        )
        message = response.choices[0].message
        parsed = getattr(message, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        try:
            return json.loads(extract_json_candidate(extract_message_text(message)))
        except Exception as exc:
            last_err = exc
            time.sleep(min(attempt, 3))
    if last_err is not None:
        raise last_err
    raise RuntimeError("OpenAI call failed without a parseable response")


def call_clean_json(prompt: str, model: str) -> Dict[str, Any]:
    from openai import OpenAI
    from run_transcript_cleanup_segmentation import CLEAN_JSON_SCHEMA

    client = OpenAI()
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        user_prompt = prompt if attempt == 1 else f"{prompt}\n\nReturn valid JSON only. Do not include any prose."
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "clean_transcript_payload", "schema": CLEAN_JSON_SCHEMA, "strict": True},
            },
        )
        message = response.choices[0].message
        parsed = getattr(message, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        try:
            return json.loads(extract_json_candidate(extract_message_text(message)))
        except Exception as exc:
            last_err = exc
            time.sleep(min(attempt, 3))
    if last_err is not None:
        raise last_err
    raise RuntimeError("OpenAI clean call failed without a parseable response")


def call_clean_with_summary_json(prompt: str, model: str) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        user_prompt = prompt if attempt == 1 else f"{prompt}\n\nReturn valid JSON only. Do not include any prose."
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "clean_transcript_with_summary_payload",
                    "schema": CLEAN_WITH_SUMMARY_JSON_SCHEMA,
                    "strict": True,
                },
            },
        )
        message = response.choices[0].message
        parsed = getattr(message, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        try:
            return json.loads(extract_json_candidate(extract_message_text(message)))
        except Exception as exc:
            last_err = exc
            time.sleep(min(attempt, 3))
    if last_err is not None:
        raise last_err
    raise RuntimeError("OpenAI clean+summary call failed without a parseable response")


def load_target_age_ranges(manifest_csv: Path, video_id_col: str, age_col: str) -> Dict[str, str]:
    df = pd.read_csv(manifest_csv, low_memory=False)
    if video_id_col not in df.columns or age_col not in df.columns:
        return {}
    out: Dict[str, str] = {}
    for _, row in df[[video_id_col, age_col]].iterrows():
        video_id = normalize_text(row.get(video_id_col))
        age_range = normalize_age_text(row.get(age_col))
        if video_id and age_range:
            out[video_id] = age_range
    return out


def load_processed_summary_keys(path: Path) -> Set[str]:
    out: Set[str] = set()
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            story_id = normalize_text(row.get("story_id"))
            condition = normalize_text(row.get("condition"))
            target_age_range = normalize_text(row.get("target_age_range"))
            if story_id and condition and target_age_range:
                out.add(f"{story_id}\t{condition}\t{target_age_range}")
    return out


def append_age_steered_summaries(
    summary_jsonl: Optional[Path],
    processed_summary_keys: Set[str],
    *,
    story_id: str,
    video_id: str,
    video_title: str,
    model: str,
    age_range: str,
    summaries: List[Dict[str, Any]],
) -> None:
    if summary_jsonl is None or not summaries:
        return
    source_bounds = parse_age_bounds(age_range)
    source_mid = parse_age_midpoint(age_range)
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        label = normalize_text(summary.get("target_age_label")).lower()
        target_setting = normalize_text(summary.get("target_age_setting"))
        summary_text = normalize_text(summary.get("summary"))
        if not label or not target_setting or not summary_text:
            continue
        key = f"{story_id}\tage\t{target_setting}"
        if key in processed_summary_keys:
            continue
        target_mid = parse_age_midpoint(target_setting)
        append_jsonl(
            summary_jsonl,
            {
                "story_id": story_id,
                "video_id": video_id,
                "video_title": video_title,
                "condition": "age",
                "model": model,
                "age_range": age_range,
                "age_source": "customer",
                "target_age": target_mid,
                "target_age_range": target_setting,
                "target_age_label": label,
                "source_age_midpoint": source_mid,
                "source_age_bounds": format_age_range(*source_bounds) if source_bounds else age_range,
                "abs_age_gap": None if target_mid is None or source_mid is None else abs(float(target_mid) - float(source_mid)),
                "prompt_path": "",
                "response": {
                    "target_age_setting": target_setting,
                    "summaries": [{"summary_id": 1, "summary": summary_text}],
                },
            },
        )
        processed_summary_keys.add(key)


def build_summary_json_examples(n_summaries: int) -> str:
    return ",\n".join(
        f'        {{"summary_id": {summary_id}, "summary": "..."}}'
        for summary_id in range(1, n_summaries + 1)
    )


def normalize_clean_stories_with_summary(clean_payload: Dict[str, Any]) -> List[Dict[str, str]]:
    stories = clean_payload.get("stories") if isinstance(clean_payload.get("stories"), list) else []
    normalized: List[Dict[str, str]] = []
    for story in stories:
        cleaned_story = normalize_text(story.get("cleaned_story_verbatim", ""))
        punctuated_story = normalize_text(story.get("punctuated_story", ""))
        story_title_guess = normalize_text(story.get("story_title_guess", "")) or "N/A"
        age_steered_summaries = story.get("age_steered_summaries")
        if not isinstance(age_steered_summaries, list):
            actual_age_summaries = story.get("actual_age_summaries")
            if not isinstance(actual_age_summaries, list):
                legacy_summary = normalize_text(story.get("actual_age_summary", ""))
                actual_age_summaries = [{"summary_id": 1, "summary": legacy_summary}] if legacy_summary else []
            age_steered_summaries = []
            for index, summary in enumerate(actual_age_summaries, start=1):
                if not isinstance(summary, dict):
                    continue
                summary_text = normalize_text(summary.get("summary"))
                if not summary_text:
                    continue
                age_steered_summaries.append(
                    {
                        "target_age_label": "actual",
                        "target_age_setting": "",
                        "summary": summary_text,
                    }
                )
        normalized_summaries: List[Dict[str, Any]] = []
        for summary in age_steered_summaries:
            if not isinstance(summary, dict):
                continue
            summary_text = normalize_text(summary.get("summary"))
            if not summary_text:
                continue
            normalized_summaries.append(
                {
                    "target_age_label": normalize_text(summary.get("target_age_label")).lower() or "actual",
                    "target_age_setting": normalize_text(summary.get("target_age_setting")),
                    "summary": summary_text,
                }
            )
        raw_segments = story.get("segments")
        normalized_segments: List[Dict[str, Any]] = []
        if isinstance(raw_segments, list):
            for index, segment in enumerate(raw_segments, start=1):
                if not isinstance(segment, dict):
                    continue
                segment_text = normalize_text(segment.get("text"))
                if not segment_text:
                    continue
                segment_id = segment.get("segment_id", index)
                try:
                    segment_id = int(segment_id)
                except Exception:
                    segment_id = index
                normalized_segments.append({"segment_id": segment_id, "text": segment_text})
        if not cleaned_story and not punctuated_story:
            continue
        normalized.append(
            {
                "story_title_guess": story_title_guess,
                "cleaned_story_verbatim": cleaned_story,
                "punctuated_story": punctuated_story,
                "age_steered_summaries": normalized_summaries,
                "segments": normalized_segments,
            }
        )
    return normalized


def load_rows_from_input_dirs(input_dirs: Iterable[Path], min_words: int) -> List[StoryRow]:
    merged: Dict[str, StoryRow] = {}
    for input_dir in input_dirs:
        for row in load_labcomputer_rows(input_dir, min_words=min_words):
            prev = merged.get(row.video_id)
            if prev is None or len(row.transcript_text) > len(prev.transcript_text):
                merged[row.video_id] = row
    return list(merged.values())


def build_selected_rows(
    input_dirs: List[Path],
    manifest_csv: Path,
    manifest_video_id_col: str,
    titles_dir: Optional[Path],
    require_title_success: bool,
    require_book_metadata: bool,
    min_words: int,
) -> List[StoryRow]:
    rows = load_rows_from_input_dirs(input_dirs, min_words=min_words)
    keep_ids = load_manifest_video_ids(manifest_csv, manifest_video_id_col)
    rows = [row for row in rows if row.video_id in keep_ids]
    if titles_dir is not None and (require_title_success or require_book_metadata):
        title_ids = load_title_success_video_ids(
            titles_dir,
            require_success=require_title_success,
            require_book_metadata=require_book_metadata,
        )
        rows = [row for row in rows if row.video_id in title_ids]
    return rows


def processed_story_keys(detail_jsonl: Path) -> Set[str]:
    out: Set[str] = set()
    if not detail_jsonl.exists():
        return out
    with detail_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            key = normalize_text(payload.get("story_key"))
            if key:
                out.add(key)
    return out


def process_row(
    row: StoryRow,
    *,
    clean_model: str,
    detail_model: str,
    clean_cache_path: Path,
    detail_cache_path: Path,
    cleaned_dir: Path,
    details_jsonl: Path,
    manifest_rows: List[Dict[str, Any]],
    sleep_seconds: float,
    already_done: Set[str],
    summarize_in_clean: bool,
    target_age_range: str,
    summary_jsonl: Optional[Path],
    processed_summary_keys: Set[str],
    detail_in_clean: bool,
    n_summaries_in_clean: int,
) -> None:
    clean_cache = load_jsonl_cache(clean_cache_path)
    detail_cache = load_jsonl_cache(detail_cache_path)

    clean_version = f"{SUMMARY_CLEAN_PROMPT_VERSION}_n{n_summaries_in_clean}" if summarize_in_clean else CLEAN_PROMPT_VERSION
    clean_mode = "clean_summary" if summarize_in_clean else "clean"
    clean_key = record_key(clean_mode, row.video_id, row.transcript_text, clean_model, clean_version)
    clean_entry = clean_cache.get(clean_key)
    if clean_entry is None:
        if summarize_in_clean:
            bounds = parse_age_bounds(target_age_range)
            if bounds is None:
                younger_target_age = "N/A"
                actual_target_age_range = normalize_text(target_age_range) or "N/A"
                older_target_age = "N/A"
            else:
                lower, upper = bounds
                younger_target_age = str(lower - 3)
                actual_target_age_range = format_age_range(lower, upper)
                older_target_age = str(upper + 3)
            clean_payload = call_clean_with_summary_json(
                CLEAN_AND_SUMMARIZE_PROMPT.format(
                    story=row.transcript_text,
                    younger_target_age=younger_target_age,
                    actual_target_age_range=actual_target_age_range,
                    older_target_age=older_target_age,
                ),
                clean_model,
            )
        else:
            clean_payload = call_clean_json(CLEAN_AND_PUNCTUATE_PROMPT.format(story=row.transcript_text), clean_model)
        clean_entry = {"key": clean_key, "video_id": row.video_id, "payload": clean_payload}
        append_jsonl(clean_cache_path, clean_entry)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    else:
        clean_payload = clean_entry["payload"]

    stories = normalize_clean_stories_with_summary(clean_payload) if summarize_in_clean else normalize_clean_stories(clean_payload)
    for substory_index, story in enumerate(stories, start=1):
        current_key = story_key(row.video_id, substory_index)
        punctuated_story = normalize_text(story.get("punctuated_story"))
        if not punctuated_story or word_count(punctuated_story) < 20:
            upsert_manifest_row(
                manifest_rows,
                {
                    "video_id": row.video_id,
                    "story_key": current_key,
                    "status": "cleaned_but_too_short",
                    "substory_index": substory_index,
                    "n_substories": len(stories),
                },
            )
            continue
        cleaned_path = write_cleaned_txt(cleaned_dir, current_key, story.get("story_title_guess", row.video_title), punctuated_story)
        if current_key in already_done:
            upsert_manifest_row(
                manifest_rows,
                {
                    "video_id": row.video_id,
                    "story_key": current_key,
                    "status": "already_segmented",
                    "substory_index": substory_index,
                    "n_substories": len(stories),
                    "cleaned_path": str(cleaned_path),
                },
            )
            continue

        if detail_in_clean and story.get("segments"):
            detail_payload = {"segments": story.get("segments", [])}
        else:
            detail_key = record_key("detail", current_key, punctuated_story, detail_model, DETAIL_PROMPT_VERSION)
            detail_entry = detail_cache.get(detail_key)
            if detail_entry is None:
                detail_payload = call_openai_json(
                    DETAIL_PROMPT.format(story_text=punctuated_story),
                    detail_model,
                    schema_name="detail_segments",
                    system_prompt=DETAIL_SYSTEM_PROMPT,
                )
                detail_entry = {"key": detail_key, "story_key": current_key, "payload": detail_payload}
                append_jsonl(detail_cache_path, detail_entry)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            else:
                detail_payload = detail_entry["payload"]

        payload = {
            "story_key": current_key,
            "video_id": row.video_id,
            "video_title": row.video_title,
            "source_file": row.source_file,
            "substory_index": substory_index,
            "n_substories": len(stories),
            "story_title_guess": normalize_text(story.get("story_title_guess")) or "N/A",
            "cleaned_story_verbatim": normalize_text(story.get("cleaned_story_verbatim")),
            "punctuated_story": punctuated_story,
            "cleaned_path": str(cleaned_path),
            "age_steered_summaries": story.get("age_steered_summaries", []),
            "summary_target_age_range": target_age_range,
            "segments": detail_payload.get("segments", []),
        }
        append_jsonl(details_jsonl, payload)
        append_age_steered_summaries(
            summary_jsonl,
            processed_summary_keys,
            story_id=current_key,
            video_id=row.video_id,
            video_title=row.video_title,
            model=clean_model,
            age_range=target_age_range,
            summaries=story.get("age_steered_summaries", []),
        )
        already_done.add(current_key)
        upsert_manifest_row(
            manifest_rows,
            {
                "video_id": row.video_id,
                "story_key": current_key,
                "status": "segmented",
                "substory_index": substory_index,
                "n_substories": len(stories),
                "cleaned_path": str(cleaned_path),
                "n_segments": len(detail_payload.get("segments", [])),
            },
        )

    if not stories:
        upsert_manifest_row(
            manifest_rows,
            {
                "video_id": row.video_id,
                "story_key": story_key(row.video_id, 1),
                "status": "no_recoverable_story",
                "substory_index": 0,
                "n_substories": 0,
            },
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean transcripts and segment them into details only.")
    ap.add_argument("--input-dir", type=Path, action="append", required=True, help="Repeat for multiple transcript source directories.")
    ap.add_argument("--manifest-csv", type=Path, required=True)
    ap.add_argument("--manifest-video-id-col", default="video_id")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--output-prefix", default="transcript_cleanup_detail")
    ap.add_argument("--key-path", type=Path, default=None)
    ap.add_argument("--clean-model", default=DEFAULT_CLEAN_MODEL)
    ap.add_argument("--detail-model", default=DEFAULT_DETAIL_MODEL)
    ap.add_argument("--titles-dir", type=Path, default=None)
    ap.add_argument("--require-title-success", action="store_true")
    ap.add_argument("--require-book-metadata", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--min-words", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--summarize-in-clean", action="store_true", help="Also generate three actual-age summaries in the cleanup call to avoid a separate summary API pass.")
    ap.add_argument("--detail-in-clean", action="store_true", help="Also generate full-text detail segments in the cleanup call, avoiding the separate detail API call when segments are returned.")
    ap.add_argument("--n-summaries-in-clean", type=int, default=1, help="Number of actual-age summaries to request when --summarize-in-clean is set.")
    ap.add_argument("--summary-jsonl", type=Path, default=None, help="Optional JSONL output for summaries generated by --summarize-in-clean.")
    ap.add_argument("--target-age-range-col", default="best_Reading Age")
    args = ap.parse_args()

    load_api_key(args.key_path)
    ensure_dir(args.out_dir)

    selected = build_selected_rows(
        args.input_dir,
        args.manifest_csv,
        args.manifest_video_id_col,
        args.titles_dir,
        args.require_title_success,
        args.require_book_metadata,
        args.min_words,
    )
    if args.start:
        selected = selected[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    clean_cache_path = args.out_dir / f"{args.output_prefix}_clean_cache.jsonl"
    detail_cache_path = args.out_dir / f"{args.output_prefix}_detail_cache.jsonl"
    details_jsonl = args.out_dir / f"{args.output_prefix}_detail_segments.jsonl"
    cleaned_dir = args.out_dir / "cleaned_txt"
    manifest_path = args.out_dir / f"{args.output_prefix}_manifest.csv"
    already_done = processed_story_keys(details_jsonl)
    manifest_rows = load_existing_manifest_rows(manifest_path)
    target_age_ranges = load_target_age_ranges(
        args.manifest_csv,
        args.manifest_video_id_col,
        args.target_age_range_col,
    )
    processed_summary_keys = load_processed_summary_keys(args.summary_jsonl) if args.summary_jsonl else set()

    print(f"Run dir: {args.out_dir}")
    print("Input dirs:")
    for input_dir in args.input_dir:
        print(f"  - {input_dir}")
    print(f"Stories selected: {len(selected)}")
    print(f"Start offset: {args.start}")
    print(f"Clean model: {args.clean_model}")
    print(f"Detail model: {args.detail_model}")
    print(f"Min words: {args.min_words}")
    print(f"Summarize in clean: {args.summarize_in_clean}")
    print(f"Detail in clean: {args.detail_in_clean}")
    print(f"N summaries in clean: {args.n_summaries_in_clean}")
    if args.summary_jsonl:
        print(f"Summary JSONL: {args.summary_jsonl}")

    for idx, row in enumerate(selected, start=1):
        try:
            process_row(
                row,
                clean_model=args.clean_model,
                detail_model=args.detail_model,
                clean_cache_path=clean_cache_path,
                detail_cache_path=detail_cache_path,
                cleaned_dir=cleaned_dir,
                details_jsonl=details_jsonl,
                manifest_rows=manifest_rows,
                sleep_seconds=args.sleep,
                already_done=already_done,
                summarize_in_clean=args.summarize_in_clean,
                target_age_range=target_age_ranges.get(row.video_id, ""),
                summary_jsonl=args.summary_jsonl,
                processed_summary_keys=processed_summary_keys,
                detail_in_clean=args.detail_in_clean,
                n_summaries_in_clean=args.n_summaries_in_clean,
            )
        except Exception as exc:
            upsert_manifest_row(
                manifest_rows,
                {
                    "video_id": row.video_id,
                    "story_key": story_key(row.video_id, 1),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            if args.stop_on_error:
                raise
        last = manifest_rows[-1] if manifest_rows else {}
        print(
            f"[{idx}/{len(selected)}] {row.video_id} | "
            f"status={last.get('status', 'unknown')} | "
            f"story_key={last.get('story_key', '')}"
        )
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print(f"Wrote: {details_jsonl}")
    print(f"Wrote: {manifest_path}")
    print(f"Cleaned txt dir: {cleaned_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
