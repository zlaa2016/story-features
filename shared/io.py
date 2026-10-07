from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd
import numpy as np


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:[-']\w+)?\b", normalize_text(text)))


def text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", normalize_text(text)).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield value


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_npy(path: Path, array: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        np.save(temporary, array)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]
