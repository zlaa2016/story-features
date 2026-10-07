from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_index.py")
SPEC = importlib.util.spec_from_file_location("dataset_build_index", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_repair_title_fragments_merges_without_dropping_words() -> None:
    segments = [
        {"segment_id": 1, "text": "Mr."},
        {"segment_id": 2, "text": "Watson opened the door."},
        {"segment_id": 3, "text": "No."},
    ]

    repaired = MODULE.repair_title_fragments(segments)

    assert repaired == [
        {"segment_id": 1, "text": "Mr. Watson opened the door."},
        {"segment_id": 2, "text": "No."},
    ]
    assert " ".join(item["text"] for item in repaired) == " ".join(item["text"] for item in segments)


def test_repair_title_fragments_preserves_terminal_title() -> None:
    segments = [{"segment_id": 1, "text": "The visitor was Dr."}]

    assert MODULE.repair_title_fragments(segments) == segments
