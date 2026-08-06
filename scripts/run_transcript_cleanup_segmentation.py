#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd


CLEAN_PROMPT_VERSION = "v1"
SEGMENT_PROMPT_VERSION = "v1"
DEFAULT_CLEAN_MODEL = os.environ.get("OPENAI_CLEAN_MODEL", "gpt-5.4-nano")
DEFAULT_SEGMENT_MODEL = os.environ.get("OPENAI_SEGMENT_MODEL", "gpt-5.4-mini")


CLEAN_JSON_SCHEMA = {
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
                },
                "required": ["story_title_guess", "cleaned_story_verbatim", "punctuated_story"],
                "additionalProperties": False,
            }
        }
    },
    "required": ["stories"],
    "additionalProperties": False,
}


SEGMENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_sent_idx": {"type": "integer"},
                    "end_sent_idx": {"type": "integer"},
                    "boundary_cue": {"type": "string"},
                },
                "required": ["start_sent_idx", "end_sent_idx", "boundary_cue"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["events"],
    "additionalProperties": False,
}


CLEAN_AND_PUNCTUATE_PROMPT = """
You will be given an automatically generated transcript of a YouTube video that tells a children's story.

Your tasks:
1. Identify the distinct story narrative(s) present in the transcript, if any.
2. Remove all content that is NOT part of each story itself.
3. Restore readable punctuation and sentence boundaries for each story-only text.

Remove:
- greetings, channel promotion, title/author introductions
- "the end", "thanks for watching", subscribe requests, link mentions
- narrator commentary that is not part of the story
- subtitle headers like "Kind: captions Language: en"
- sound-effect markers like [Music], [Applause] when they are not story text

Story splitting rules:
- If the video contains one clear story, return exactly one item in `stories`.
- If the video contains multiple distinct stories, chapters, or anthology entries, split them into separate `stories` items in chronological order.
- Do NOT split a single continuous story into multiple items just because it has several scenes.
- If a transcript contains almost no recoverable story content, return an empty `stories` list.

Requirements for each `cleaned_story_verbatim`:
- Keep ONLY the story content for that story.
- Keep the surviving story words in their original order.
- Do NOT add new story content.
- Do NOT rewrite or paraphrase the story.
- Do NOT summarize the story.

Requirements for each `punctuated_story`:
- Use the same story content after cleaning.
- Add punctuation, capitalization, and sentence boundaries.
- Do NOT add or remove story facts.
- Keep the wording as close as possible while making it readable.

For `story_title_guess`:
- Use a short title-like label if inferable from the transcript.
- Otherwise return "N/A".

Return valid JSON with exactly this structure:
{{
  "stories": [
    {{
      "story_title_guess": "...",
      "cleaned_story_verbatim": "...",
      "punctuated_story": "..."
    }}
  ]
}}

Transcript:
{story}
""".strip()


SEGMENT_PROMPT = """
You are segmenting a children's story into coherent narrative events using the same event-boundary logic as Method O v2.1.

Theoretical grounding:
- Treat event boundaries as situation-model updates in the sense of Event Segmentation Theory.

Definition of a narrative event:
- An event is a contiguous block of sentences unified by the current characters, location, time, goal, activity, or causal situation.

Segmentation rules:
- Cover every sentence exactly once.
- Events must be contiguous and non-overlapping.
- The first event must start at sentence 0.
- The last event must end at the final sentence.
- Start a new event only when there is a meaningful shift in time, location, characters present, goal, activity, or causal situation.
- Keep dialogue together when it belongs to the same ongoing situation.
- Do NOT rely on punctuation alone as a boundary signal.
- Do NOT assign narrative arc labels such as exposition, climax, or initiating_event.

For each event after the first, briefly note the main cue that triggered the boundary in `boundary_cue`.
Use `story_start` for the first event.

Return valid JSON only:
{{
  "events": [
    {{"start_sent_idx": 0, "end_sent_idx": 4, "boundary_cue": "story_start"}},
    {{"start_sent_idx": 5, "end_sent_idx": 11, "boundary_cue": "location shift"}}
  ]
}}

Story sentences:
{numbered_sentences}
""".strip()


@dataclass
class StoryRow:
    story_id: str
    video_id: str
    video_title: str
    transcript_source: str
    duration: Optional[float]
    transcript_text: str
    source_file: str


def load_api_key(key_path: Optional[Path]) -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    if key_path and key_path.exists():
        key = key_path.read_text(encoding="utf-8").strip()
        os.environ["OPENAI_API_KEY"] = key
        return key
    raise FileNotFoundError("OPENAI_API_KEY not set and key file not found")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def clean_input_text(text: str) -> str:
    text = text.replace("\ufeff", " ").replace("\xa0", " ")
    text = re.sub(r"Kind:\s*captions\s*Language:\s*[A-Za-z-]+", " ", text, flags=re.I)
    text = re.sub(r"\[(music|applause|laughter|laughs|cheering|sound effects?)\]", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len([token for token in re.split(r"\s+", text.strip()) if token])


def load_labcomputer_rows(input_dir: Path, min_words: int = 1) -> List[StoryRow]:
    rows: List[StoryRow] = []
    for csv_path in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, low_memory=False)
        if "id" not in df.columns or "cleaned_subtitles" not in df.columns:
            continue
        for row in df.itertuples(index=False):
            video_id = normalize_text(getattr(row, "id", ""))
            transcript_text = clean_input_text(normalize_text(getattr(row, "cleaned_subtitles", "")))
            if not video_id or not transcript_text or word_count(transcript_text) < min_words:
                continue
            rows.append(
                StoryRow(
                    story_id="",
                    video_id=video_id,
                    video_title=normalize_text(getattr(row, "title", "")),
                    transcript_source=normalize_text(getattr(row, "transcript_source", "")) or "missing",
                    duration=_coerce_float(getattr(row, "duration", None)),
                    transcript_text=transcript_text,
                    source_file=csv_path.name,
                )
            )
    deduped: Dict[str, StoryRow] = {}
    for row in rows:
        prev = deduped.get(row.video_id)
        if prev is None or len(row.transcript_text) > len(prev.transcript_text):
            deduped[row.video_id] = row
    return list(deduped.values())


def load_title_success_video_ids(
    titles_dir: Path,
    require_success: bool = True,
    require_book_metadata: bool = False,
) -> Set[str]:
    video_ids: Set[str] = set()
    for csv_path in sorted(titles_dir.glob("**/*_titles.csv")):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue
        if "video_id" not in df.columns:
            continue
        mask = pd.Series([True] * len(df), index=df.index)
        if require_success and "extraction_status" in df.columns:
            status = df["extraction_status"].fillna("").astype(str).str.lower().str.strip()
            mask &= status.eq("success")
        if require_book_metadata:
            book_ok = pd.Series([False] * len(df), index=df.index)
            if "book_title" in df.columns:
                book_ok |= df["book_title"].fillna("").astype(str).str.strip().str.upper().ne("N/A")
                book_ok &= df["book_title"].fillna("").astype(str).str.strip().ne("")
            if "story_title" in df.columns:
                story_ok = df["story_title"].fillna("").astype(str).str.strip()
                book_ok |= story_ok.ne("") & story_ok.str.upper().ne("N/A")
            mask &= book_ok
        ids = df.loc[mask, "video_id"].fillna("").astype(str).str.strip()
        video_ids.update(video_id for video_id in ids if video_id)
    return video_ids


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except Exception:
        return None


def sanitize_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", normalize_text(text)).strip("_")
    return (text[:max_len] or "story")


def assign_story_ids(rows: List[StoryRow]) -> List[StoryRow]:
    assigned: List[StoryRow] = []
    for idx, row in enumerate(sorted(rows, key=lambda item: (item.video_id, item.source_file)), start=1):
        assigned.append(
            StoryRow(
                story_id=f"{idx:06d}",
                video_id=row.video_id,
                video_title=row.video_title,
                transcript_source=row.transcript_source,
                duration=row.duration,
                transcript_text=row.transcript_text,
                source_file=row.source_file,
            )
        )
    return assigned


def story_stub(story_id: str, title_text: str) -> str:
    title_part = sanitize_filename(title_text)
    return f"{story_id}_{title_part}" if title_part else story_id


def local_event_summary(sentences: List[str], start: int, end: int, max_words: int = 24) -> str:
    text = " ".join(sentences[start : end + 1]).strip()
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:") + "..."


def write_cleaned_txt(cleaned_dir: Path, story_id: str, title_text: str, punctuated_story: str) -> Path:
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    path = cleaned_dir / f"{story_stub(story_id, title_text)}_cleaned.txt"
    path.write_text((punctuated_story.strip() + "\n") if punctuated_story.strip() else "", encoding="utf-8")
    return path


def write_review_sample_md(review_dir: Path, story_id: str, title_text: str, sentences: List[str], events: List[Dict[str, Any]], method: str = "Method O v2.1") -> Path:
    review_dir.mkdir(parents=True, exist_ok=True)
    path = review_dir / f"{story_stub(story_id, title_text)}.md"
    lines = [
        f"# {story_stub(story_id, title_text)}",
        "",
        f"**{len(sentences)} sentences | {sum(len(s.split()) for s in sentences)} words | {len(events)} events**",
        "",
        f"## {method} — {len(events)} events",
        "",
    ]
    for event_idx, event in enumerate(events):
        start_sent = int(event["start_sent_idx"])
        end_sent = int(event["end_sent_idx"])
        n_sent = end_sent - start_sent + 1
        boundary = (normalize_text(event.get("boundary_cue", "")) or ("story_start" if event_idx == 0 else "unspecified")).replace("_", " ")
        summary = local_event_summary(sentences, start_sent, end_sent)
        lines.extend(
            [
                f"### Event {event_idx} — sents {start_sent}–{end_sent} ({n_sent} sentences)",
                "**event**",
                f"*Boundary: {boundary}*",
                f"*{summary}*",
                "",
            ]
        )
        for sent_idx in range(start_sent, end_sent + 1):
            lines.append(f"> [{sent_idx}] {sentences[sent_idx]}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def build_substory_id(root_story_id: str, substory_index: int, n_substories: int) -> str:
    if n_substories <= 1:
        return root_story_id
    return f"{root_story_id}_{substory_index:02d}"


def normalize_clean_stories(clean_payload: Dict[str, Any]) -> List[Dict[str, str]]:
    if isinstance(clean_payload.get("stories"), list):
        stories = clean_payload.get("stories", [])
    else:
        stories = [
            {
                "story_title_guess": "N/A",
                "cleaned_story_verbatim": clean_payload.get("cleaned_story_verbatim", ""),
                "punctuated_story": clean_payload.get("punctuated_story", ""),
            }
        ]
    normalized: List[Dict[str, str]] = []
    for story in stories:
        cleaned_story = normalize_text(story.get("cleaned_story_verbatim", ""))
        punctuated_story = normalize_text(story.get("punctuated_story", ""))
        story_title_guess = normalize_text(story.get("story_title_guess", "")) or "N/A"
        if not cleaned_story and not punctuated_story:
            continue
        normalized.append(
            {
                "story_title_guess": story_title_guess,
                "cleaned_story_verbatim": cleaned_story,
                "punctuated_story": punctuated_story,
            }
        )
    return normalized


def record_key(step: str, story_id: str, text: str, model: str, prompt_version: str) -> str:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{step}|{story_id}|{text_hash}|{model}|{prompt_version}"


def load_jsonl_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = entry.get("key")
            if key:
                cache[key] = entry
    return cache


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
    sentences = [part.strip() for part in parts if part.strip()]
    if len(sentences) <= 1:
        parts = re.split(r"(?<=[.!?])\s+", text)
        sentences = [part.strip() for part in parts if part.strip()]
    return sentences or [text]


def make_numbered_sentences(sentences: Iterable[str]) -> str:
    return " ".join(f"[{idx}] {sentence}" for idx, sentence in enumerate(sentences))


def validate_events(events: List[Dict[str, Any]], n_sentences: int) -> List[Dict[str, Any]]:
    if n_sentences <= 0:
        return []
    cleaned: List[Dict[str, Any]] = []
    for idx, event in enumerate(events):
        try:
            start = int(event["start_sent_idx"])
            end = int(event["end_sent_idx"])
        except Exception as exc:
            raise ValueError(f"Invalid event indices at position {idx}: {event}") from exc
        cue = normalize_text(event.get("boundary_cue", "")) or ("story_start" if idx == 0 else "unspecified")
        cleaned.append({"start_sent_idx": start, "end_sent_idx": end, "boundary_cue": cue})
    cleaned.sort(key=lambda event: (event["start_sent_idx"], event["end_sent_idx"]))
    if cleaned[0]["start_sent_idx"] != 0:
        raise ValueError("First event does not start at sentence 0")
    if cleaned[-1]["end_sent_idx"] != n_sentences - 1:
        raise ValueError("Last event does not end at final sentence")
    expected = 0
    for event in cleaned:
        if event["start_sent_idx"] != expected:
            raise ValueError(f"Gap or overlap before sentence {expected}")
        if event["end_sent_idx"] < event["start_sent_idx"]:
            raise ValueError("Event end before start")
        expected = event["end_sent_idx"] + 1
    return cleaned


def _extract_json_candidate(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def call_openai_json(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_attempts: int = 3,
    schema_name: str = "response_payload",
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt if attempt == 1 else (prompt + "\n\nReturn valid JSON only. Do not include any extra text."),
                }
            ],
            temperature=temperature,
            response_format=(
                {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}}
                if schema is not None
                else {"type": "json_object"}
            ),
        )
        message = response.choices[0].message
        parsed = getattr(message, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        candidate = _extract_json_candidate(message.content or "")
        try:
            return json.loads(candidate)
        except Exception as exc:
            last_err = exc
            time.sleep(min(1.0 * attempt, 3.0))
    if last_err is not None:
        raise last_err
    raise RuntimeError("OpenAI JSON call failed without an exception")


def process_story(
    row: StoryRow,
    clean_model: str,
    segment_model: str,
    clean_cache_path: Path,
    segment_cache_path: Path,
    output_jsonl_path: Path,
    openai_schema_jsonl_path: Path,
    cleaned_dir: Path,
    review_dir: Path,
    sleep_seconds: float,
    max_calls_remaining: Optional[List[int]],
) -> List[Dict[str, Any]]:
    clean_cache = load_jsonl_cache(clean_cache_path)
    segment_cache = load_jsonl_cache(segment_cache_path)
    clean_key = record_key("clean", row.video_id, row.transcript_text, clean_model, CLEAN_PROMPT_VERSION)

    if clean_key in clean_cache:
        clean_payload = clean_cache[clean_key]["payload"]
        clean_from_cache = True
    else:
        if max_calls_remaining is not None and max_calls_remaining[0] <= 0:
            raise RuntimeError("Max API call budget reached before clean step")
        clean_payload = call_openai_json(
            CLEAN_AND_PUNCTUATE_PROMPT.format(story=row.transcript_text),
            model=clean_model,
            temperature=0.0,
            schema_name="cleanup_payload",
            schema=CLEAN_JSON_SCHEMA,
        )
        append_jsonl(
            clean_cache_path,
            {
                "key": clean_key,
                "story_id": row.video_id,
                "step": "clean",
                "model": clean_model,
                "prompt_version": CLEAN_PROMPT_VERSION,
                "timestamp": time.time(),
                "payload": clean_payload,
            },
        )
        clean_from_cache = False
        if max_calls_remaining is not None:
            max_calls_remaining[0] -= 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    clean_stories = normalize_clean_stories(clean_payload)
    n_substories = len(clean_stories)
    records: List[Dict[str, Any]] = []
    for substory_index, story_payload in enumerate(clean_stories, start=1):
        cleaned_story = story_payload["cleaned_story_verbatim"]
        punctuated_story = story_payload["punctuated_story"]
        story_title_guess = story_payload["story_title_guess"]
        substory_id = build_substory_id(row.story_id, substory_index, n_substories)
        effective_title = story_title_guess if story_title_guess != "N/A" else row.video_title
        sentences = sentence_split(punctuated_story)

        segment_key = record_key("segment", f"{row.video_id}:{substory_index}", punctuated_story, segment_model, SEGMENT_PROMPT_VERSION)
        if segment_key in segment_cache:
            segment_payload = segment_cache[segment_key]["payload"]
            segment_from_cache = True
        else:
            if not sentences:
                segment_payload = {"events": []}
                segment_from_cache = True
            else:
                if max_calls_remaining is not None and max_calls_remaining[0] <= 0:
                    raise RuntimeError("Max API call budget reached before segmentation step")
                segment_payload = call_openai_json(
                    SEGMENT_PROMPT.format(numbered_sentences=make_numbered_sentences(sentences)),
                    model=segment_model,
                    temperature=0.0,
                    schema_name="segment_payload",
                    schema=SEGMENT_JSON_SCHEMA,
                )
                append_jsonl(
                    segment_cache_path,
                    {
                        "key": segment_key,
                        "story_id": substory_id,
                        "story_root_id": row.story_id,
                        "step": "segment",
                        "model": segment_model,
                        "prompt_version": SEGMENT_PROMPT_VERSION,
                        "timestamp": time.time(),
                        "payload": segment_payload,
                    },
                )
                segment_from_cache = False
                if max_calls_remaining is not None:
                    max_calls_remaining[0] -= 1
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

        events = validate_events(segment_payload.get("events", []), len(sentences)) if sentences else []
        cleaned_txt_path = write_cleaned_txt(cleaned_dir, substory_id, effective_title, punctuated_story)
        review_md_path = write_review_sample_md(review_dir, substory_id, effective_title, sentences, events)
        openai_schema_record = {
            "story_id": substory_id,
            "story_root_id": row.story_id,
            "substory_index": substory_index,
            "n_substories_in_video": n_substories,
            "video_id": row.video_id,
            "video_title": row.video_title,
            "story_title_guess": story_title_guess,
            "n_sentences": len(sentences),
            "n_events": len(events),
            "events": [
                {
                    "start_sent_idx": int(event["start_sent_idx"]),
                    "end_sent_idx": int(event["end_sent_idx"]),
                    "summary": local_event_summary(sentences, int(event["start_sent_idx"]), int(event["end_sent_idx"])),
                    "label": "event",
                    "boundary_cue": event["boundary_cue"],
                }
                for event in events
            ],
        }
        append_jsonl(openai_schema_jsonl_path, openai_schema_record)
        record = {
            "story_id": substory_id,
            "story_root_id": row.story_id,
            "substory_index": substory_index,
            "n_substories_in_video": n_substories,
            "video_id": row.video_id,
            "video_title": row.video_title,
            "story_title_guess": story_title_guess,
            "transcript_source": row.transcript_source,
            "duration": row.duration,
            "source_file": row.source_file,
            "cleaned_txt_path": str(cleaned_txt_path),
            "review_md_path": str(review_md_path),
            "clean_model": clean_model,
            "segment_model": segment_model,
            "clean_from_cache": clean_from_cache,
            "segment_from_cache": segment_from_cache,
            "raw_transcript": row.transcript_text,
            "cleaned_story_verbatim": cleaned_story,
            "punctuated_story": punctuated_story,
            "n_sentences": len(sentences),
            "n_events": len(events),
            "sentences": sentences,
            "events": events,
        }
        append_jsonl(output_jsonl_path, record)
        records.append(record)
    return records


def load_processed_story_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            for field in ["story_id", "story_root_id", "video_id"]:
                value = normalize_text(payload.get(field, ""))
                if value:
                    ids.add(value)
    return ids


def build_run_dir(prefix: str, out_dir: Optional[Path]) -> Path:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("pipeline_output") / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_output_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def write_summary(path: Path, records: List[Dict[str, Any]]) -> None:
    rows = []
    for record in records:
        rows.append(
            {
                "story_id": record["story_id"],
                "story_root_id": record.get("story_root_id"),
                "substory_index": record.get("substory_index"),
                "n_substories_in_video": record.get("n_substories_in_video"),
                "video_title": record["video_title"],
                "story_title_guess": record.get("story_title_guess"),
                "transcript_source": record["transcript_source"],
                "duration": record["duration"],
                "source_file": record["source_file"],
                "cleaned_txt_path": record.get("cleaned_txt_path"),
                "review_md_path": record.get("review_md_path"),
                "n_sentences": record["n_sentences"],
                "n_events": record["n_events"],
                "clean_from_cache": record["clean_from_cache"],
                "segment_from_cache": record["segment_from_cache"],
                "clean_chars": len(record["cleaned_story_verbatim"]),
                "punct_chars": len(record["punctuated_story"]),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transcript cleanup + punctuation + event segmentation over labcomputer_output stories.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/labcomputer_output"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default="transcript_cleanup_segment")
    parser.add_argument("--key-path", type=Path, default=Path("/scratch") / os.environ.get("USER", "") / ".cache" / "openai" / "hartley_api_key.txt")
    parser.add_argument("--clean-model", default=DEFAULT_CLEAN_MODEL)
    parser.add_argument("--segment-model", default=DEFAULT_SEGMENT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-calls", type=int, default=None, help="Hard cap on API calls across both steps.")
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--transcript-source", choices=["manual", "auto_generated", "all"], default="all")
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--titles-dir", type=Path, default=None, help="Optional extracted titles directory to filter stories by successful title extraction.")
    parser.add_argument("--require-title-success", action="store_true", help="Keep only stories whose video_id appears in extracted title outputs with extraction_status=success.")
    parser.add_argument("--require-book-metadata", action="store_true", help="When filtering by titles, require a non-N/A book_title or story_title.")
    args = parser.parse_args()

    load_api_key(args.key_path)
    run_dir = build_run_dir(args.output_prefix, args.out_dir)
    clean_cache_path = run_dir / "clean_cache.jsonl"
    segment_cache_path = run_dir / "segment_cache.jsonl"
    output_jsonl_path = run_dir / "segmented_events.jsonl"
    openai_schema_jsonl_path = run_dir / "segmented_events_openai_v2.1.jsonl"
    summary_csv_path = run_dir / "segmentation_summary.csv"
    cleaned_dir = run_dir / "cleaned_idchecked"
    review_dir = run_dir / "review_samples"
    story_index_path = run_dir / "story_index.csv"

    stories = assign_story_ids(load_labcomputer_rows(args.input_dir, min_words=args.min_words))
    if args.transcript_source != "all":
        stories = [row for row in stories if row.transcript_source == args.transcript_source]
    if args.require_title_success:
        titles_dir = args.titles_dir or Path("data/extracted_titles")
        title_video_ids = load_title_success_video_ids(
            titles_dir,
            require_success=True,
            require_book_metadata=args.require_book_metadata,
        )
        stories = [row for row in stories if row.video_id in title_video_ids]
    stories = sorted(stories, key=lambda row: (row.story_id, row.video_id, row.source_file))
    if args.start:
        stories = stories[args.start :]
    if args.limit is not None:
        stories = stories[: args.limit]

    pd.DataFrame([
        {
            "story_id": row.story_id,
            "video_id": row.video_id,
            "video_title": row.video_title,
            "transcript_source": row.transcript_source,
            "duration": row.duration,
            "source_file": row.source_file,
        }
        for row in stories
    ]).to_csv(story_index_path, index=False)

    processed_ids = set() if args.overwrite else load_processed_story_ids(output_jsonl_path)
    stories = [row for row in stories if row.story_id not in processed_ids and row.video_id not in processed_ids]

    print(f"Run dir: {run_dir}")
    print(f"Stories selected: {len(stories)}")
    print(f"Clean model: {args.clean_model}")
    print(f"Segment model: {args.segment_model}")
    print(f"Min words: {args.min_words}")

    max_calls_remaining = [args.max_calls] if args.max_calls is not None else None
    for idx, row in enumerate(stories, start=1):
        try:
            records = process_story(
                row=row,
                clean_model=args.clean_model,
                segment_model=args.segment_model,
                clean_cache_path=clean_cache_path,
                segment_cache_path=segment_cache_path,
                output_jsonl_path=output_jsonl_path,
                openai_schema_jsonl_path=openai_schema_jsonl_path,
                cleaned_dir=cleaned_dir,
                review_dir=review_dir,
                sleep_seconds=args.sleep,
                max_calls_remaining=max_calls_remaining,
            )
            latest = records[-1] if records else None
            if idx % 25 == 0:
                write_summary(summary_csv_path, load_output_records(output_jsonl_path))
            calls_left = "unbounded" if max_calls_remaining is None else str(max_calls_remaining[0])
            print(
                f"[{idx}/{len(stories)}] {row.video_id} | subtories={len(records)}"
                + (f" | last_sents={latest['n_sentences']} last_events={latest['n_events']}" if latest is not None else "")
                + f" | calls_left={calls_left}"
            )
        except RuntimeError as exc:
            if "Max API call budget reached" in str(exc):
                print(str(exc))
                break
            raise

    write_summary(summary_csv_path, load_output_records(output_jsonl_path))
    print(f"Wrote: {output_jsonl_path}")
    print(f"Wrote: {openai_schema_jsonl_path}")
    print(f"Wrote: {summary_csv_path}")
    print(f"Wrote dir: {cleaned_dir}")
    print(f"Wrote dir: {review_dir}")
    print(f"Wrote: {story_index_path}")


if __name__ == "__main__":
    sys.exit(main())
