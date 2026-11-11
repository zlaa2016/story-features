"""Legacy event-segmented stories as deletion-graph units.

``data/legacy/review_samples_event_segments/NNNNNN_<title>.md`` lists a story's sentences grouped into events
(``### Event k — sents a–b`` followed by ``> [i] sentence`` lines).  ``NNNNNN`` is the story's ``story_id`` in the master index.
An event record has the same shape as a detail-segmented cohort record -- ``text`` plus ``segments`` -- so the deletion-graph
extractor treats an event exactly like a detail: the text is the sentences joined by single spaces, a segment is the event's sentences.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

EVENT_HEADER = re.compile(r"^### Event (\d+) — sents (\d+)[–-](\d+)", re.MULTILINE)
SENTENCE_LINE = re.compile(r"^> \[(\d+)\] (.*)$", re.MULTILINE)


def parse_event_file(path: Path) -> list[list[str]]:
    """Sentences of every event, in order.  Raises ValueError when the sentence numbers and the event ranges disagree."""
    text = Path(path).read_text(encoding="utf-8")
    sentences = {int(number): line.strip() for number, line in SENTENCE_LINE.findall(text)}
    events: list[list[str]] = []
    for _, first, last in EVENT_HEADER.findall(text):
        wanted = range(int(first), int(last) + 1)
        missing = [number for number in wanted if number not in sentences]
        if missing:
            raise ValueError(f"{path}: event sentences {missing[:3]} not listed")
        events.append([sentences[number] for number in wanted])
    return events


def event_files(event_dir: Path) -> dict[int, Path]:
    """story_id -> event file (the first file by name when an id has several)."""
    found: dict[int, Path] = {}
    for path in sorted(Path(event_dir).glob("*.md")):
        prefix = path.name.split("_", 1)[0]
        if prefix.isdigit():
            found.setdefault(int(prefix), path)
    return found


def event_record(story_key: str, story_id: int, path: Path, video_id: str = "") -> dict[str, Any]:
    events = [" ".join(sentences) for sentences in parse_event_file(path)]
    return {
        "story_key": story_key, "record_id": f"event_{story_id:06d}", "video_id": video_id, "record_type": "event",
        "text": " ".join(events), "segments": [{"text": event} for event in events],
    }


def event_records(event_dir: Path, wanted: list[tuple[str, int, str]]) -> list[dict[str, Any]]:
    """Records for ``wanted`` = [(story_key, story_id, video_id)], skipping stories without an event file."""
    files = event_files(event_dir)
    return [event_record(key, story_id, files[story_id], video_id) for key, story_id, video_id in wanted if story_id in files]
