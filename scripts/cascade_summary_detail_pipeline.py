#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MODEL = os.environ.get("OPENAI_SUMMARY_DETAIL_MODEL", "gpt-5.6-luna")

SUMMARY_DETAIL_PROMPT = """
Summarize story full text.

Create exactly one simple summary of approximately 150 words for children in this target age setting: {target_age_range}.

The summary should:
- Use vocabulary and sentence structure appropriate for children in this target age setting
- Keep important character names, places, repeated phrases, and story-specific terms from the original story when needed
- Include the beginning, middle, and end of the story arc
- Include key events in sequence and the main characters/actions
- Preserve cause-and-effect relationships and any explicit central message
- Use only the story content; do not add opinions, interpretations, motives, morals, or facts not present in the story
- Do not use bullet points
- Do not mention the target age setting in the summary itself

After writing the summary, segment that summary into minimal coherent detail units.

Detail segmentation rules for the summary:
- Keep all segment text word-for-word exactly as it appears in the summary
- Do not rewrite, normalize, correct, paraphrase, infer, or add words
- Each segment should contain one explicitly stated action, state, attribute, relation, cause, outcome, spoken statement, thought, or small connected description
- If a sentence contains multiple distinct details, split it into multiple segments
- The segments must cover the full summary in order with no missing summary text

Return valid JSON in this exact shape:
{{
  "target_age_setting": "{target_age_range}",
  "summaries": [
    {{
      "summary_id": 1,
      "summary": "...",
      "segments": [
        {{"segment_id": 1, "text": "..."}},
        {{"segment_id": 2, "text": "..."}}
      ]
    }}
  ]
}}

Story:
{story_text}
""".strip()


SEGMENT_EXISTING_SUMMARY_PROMPT = """
You will be given a story summary written for a target child age range of {target_age_range}.

Segment the summary into minimal coherent detail units.

Detail segmentation rules:
- Keep all segment text word-for-word exactly as it appears in the summary
- Do not rewrite, normalize, correct, paraphrase, infer, or add words
- Each segment should contain one explicitly stated action, state, attribute, relation, cause, outcome, spoken statement, thought, or small connected description
- If a sentence contains multiple distinct details, split it into multiple segments
- The segments must cover the full summary in order with no missing summary text

Return valid JSON in this exact shape:
{{
  "segments": [
    {{"segment_id": 1, "text": "..."}},
    {{"segment_id": 2, "text": "..."}}
  ]
}}

Summary:
{summary_text}
""".strip()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def parse_json_object_loose(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise json.JSONDecodeError("empty response", text, 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def load_api_key(key_path: Path | None) -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    if key_path and key_path.exists():
        os.environ["OPENAI_API_KEY"] = key_path.read_text(encoding="utf-8").strip()
        return
    raise FileNotFoundError("OPENAI_API_KEY not set and key file not found")


def call_openai_json(prompt: str, model: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    last_error: Exception | None = None
    for attempt_index in range(3):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                text={"format": {"type": "json_object"}},
                store=False,
            )
            return parse_json_object_loose(response.output_text)
        except Exception as error:
            last_error = error
            time.sleep(1.0 + attempt_index)
    if last_error is not None:
        raise last_error
    raise RuntimeError("OpenAI call failed without an exception")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_condition_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path, low_memory=False)
    required = {"story_key", "condition", "target_age_range"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Condition rows missing columns: {sorted(missing)}")
    rows["story_key"] = rows["story_key"].map(normalize_text)
    rows["condition"] = rows["condition"].map(normalize_text)
    rows["target_age_range"] = rows["target_age_range"].map(normalize_text)
    if "target_age_numeric" not in rows.columns:
        rows["target_age_numeric"] = pd.NA
    rows = rows.loc[rows["story_key"].ne("") & rows["condition"].ne("")].copy()
    return rows.drop_duplicates(["story_key", "condition"])


def load_manifest_lookup(path: Path) -> dict[str, dict[str, Any]]:
    rows = pd.read_csv(path, low_memory=False)
    rows["story_key"] = rows["story_key"].map(normalize_text)
    if "status" in rows.columns:
        rows = rows.loc[rows["status"].fillna("").astype(str).str.strip().isin(["segmented", "already_segmented"])].copy()
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows.to_dict(orient="records"):
        story_key = normalize_text(row.get("story_key"))
        if story_key and story_key not in lookup:
            lookup[story_key] = row
    return lookup


def read_story_text(manifest_row: dict[str, Any]) -> str:
    cleaned_path = Path(normalize_text(manifest_row.get("cleaned_path")))
    if cleaned_path.exists():
        return cleaned_path.read_text(encoding="utf-8").strip()
    text = normalize_text(manifest_row.get("punctuated_story"))
    if text:
        return text
    return normalize_text(manifest_row.get("story_text"))


def load_summary_lookup(paths: list[Path], summary_id: int) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                story_key = normalize_text(row.get("story_id") or row.get("story_key"))
                condition = normalize_text(row.get("target_age_label") or row.get("age_info_condition"))
                if not story_key or not condition:
                    continue
                summaries = (row.get("response") or {}).get("summaries") or []
                selected_summary = None
                for fallback_index, summary in enumerate(summaries, start=1):
                    if not isinstance(summary, dict):
                        continue
                    current_id = int(summary.get("summary_id", fallback_index))
                    summary_text = normalize_text(summary.get("summary"))
                    if (summary_id == 0 or current_id == summary_id) and summary_text:
                        selected_summary = {
                            "summary_id": current_id,
                            "summary": summary_text,
                            "source_record": row,
                        }
                        break
                if selected_summary is not None:
                    lookup[(story_key, condition)] = selected_summary
    return lookup


def load_segmented_summary_keys(paths: list[Path], summary_id: int) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                method = normalize_text(row.get("segmentation_method") or "api_detail")
                if method and method != "api_detail":
                    continue
                story_key = normalize_text(row.get("story_key") or row.get("story_id"))
                condition = normalize_text(row.get("age_info_condition") or row.get("target_age_label"))
                current_id = int(row.get("summary_id", 1))
                if story_key and condition and (summary_id == 0 or current_id == summary_id):
                    keys.add((story_key, condition, current_id))
    return keys


def build_jobs(
    condition_rows: pd.DataFrame,
    manifest_lookup: dict[str, dict[str, Any]],
    summary_lookup: dict[tuple[str, str], dict[str, Any]],
    segmented_keys: set[tuple[str, str, int]],
    summary_id: int,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row in condition_rows.to_dict(orient="records"):
        story_key = normalize_text(row.get("story_key"))
        condition = normalize_text(row.get("condition"))
        target_age_range = normalize_text(row.get("target_age_range"))
        target_age = row.get("target_age_numeric")
        key = (story_key, condition)
        existing_summary = summary_lookup.get(key)
        selected_summary_id = int(existing_summary["summary_id"]) if existing_summary else 1
        if summary_id == 0:
            has_segment = any(
                segmented_story == story_key and segmented_condition == condition
                for segmented_story, segmented_condition, _segmented_summary_id in segmented_keys
            )
        else:
            has_segment = (story_key, condition, selected_summary_id) in segmented_keys
        if existing_summary and has_segment:
            mode = "complete"
        elif existing_summary:
            mode = "segment_existing_summary"
        else:
            mode = "generate_summary_and_segment"
        jobs.append(
            {
                "mode": mode,
                "story_key": story_key,
                "condition": condition,
                "target_age_range": target_age_range,
                "target_age": target_age,
                "summary_id": selected_summary_id,
                "manifest": manifest_lookup.get(story_key, {}),
                "existing_summary": existing_summary,
            }
        )
    return jobs


def summary_output_record(
    job: dict[str, Any],
    model: str,
    summary_text: str,
) -> dict[str, Any]:
    manifest_row = job["manifest"]
    target_age = job.get("target_age")
    return {
        "story_id": job["story_key"],
        "video_id": normalize_text(manifest_row.get("video_id")),
        "video_title": normalize_text(manifest_row.get("video_title")),
        "condition": "age",
        "model": model,
        "age_range": normalize_text(manifest_row.get("source_age_range") or manifest_row.get("source_age_range_clean")),
        "age_source": "customer",
        "target_age": None if pd.isna(target_age) else target_age,
        "target_age_range": job["target_age_range"],
        "target_age_label": job["condition"],
        "source_age_midpoint": normalize_text(manifest_row.get("source_age_midpoint")),
        "source_age_bounds": normalize_text(manifest_row.get("source_age_range") or manifest_row.get("source_age_range_clean")),
        "abs_age_gap": None,
        "prompt_path": "",
        "response": {
            "target_age_setting": job["target_age_range"],
            "summaries": [{"summary_id": 1, "summary": summary_text}],
        },
    }


def segment_output_record(
    job: dict[str, Any],
    summary_text: str,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_row = job["manifest"]
    target_age = job.get("target_age")
    return {
        "source_age": normalize_text(manifest_row.get("source_age_midpoint")),
        "target_age": None if pd.isna(target_age) else target_age,
        "story_key": job["story_key"],
        "video_id": normalize_text(manifest_row.get("video_id")),
        "video_title": normalize_text(manifest_row.get("video_title")),
        "compression_condition": "customer_pm3_discrete",
        "age_info_condition": job["condition"],
        "summary_id": int(job.get("summary_id", 1)),
        "summary_text": summary_text,
        "segmentation_method": "api_detail",
        "target_age_range": job["target_age_range"],
        "segments": segments,
    }


def print_audit(jobs: list[dict[str, Any]]) -> None:
    by_mode = Counter(job["mode"] for job in jobs)
    print("Total expected story-condition rows:", len(jobs))
    print("By mode:", dict(by_mode))
    for mode in ["complete", "segment_existing_summary", "generate_summary_and_segment"]:
        subset = [job for job in jobs if job["mode"] == mode]
        print(f"{mode} by condition:", dict(Counter(job["condition"] for job in subset)))
    api_calls = by_mode["segment_existing_summary"] + by_mode["generate_summary_and_segment"]
    print("API calls needed:", api_calls)
    print("  segment existing summary calls:", by_mode["segment_existing_summary"])
    print("  generate summary + segment calls:", by_mode["generate_summary_and_segment"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable cascade: generate missing summaries and GPT-detail-segment summaries.")
    parser.add_argument("--condition-rows", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--existing-summary-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--existing-summary-segments-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--key-path", type=Path, default=None)
    parser.add_argument("--summary-model", default=DEFAULT_MODEL)
    parser.add_argument("--segment-model", default=DEFAULT_MODEL)
    parser.add_argument("--summary-id", type=int, default=0, help="Summary replicate to use. Use 0 for first available existing summary; generated summaries use 1.")
    parser.add_argument(
        "--condition",
        choices=["younger", "actual", "older"],
        default=None,
        help="Optionally process only one age-steering condition.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Segment existing natural-language summaries but do not generate missing summaries.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("pipeline_output") / f"cascade_summary_detail_{args.summary_model.replace('/', '_')}_{stamp}"
    ensure_dir(out_dir)
    generated_summary_path = out_dir / "generated_summary_outputs.jsonl"
    segmented_summary_path = out_dir / "detail_segments_summaries.jsonl"
    manifest_path = out_dir / "cascade_manifest.csv"

    if args.overwrite:
        for output_path in [generated_summary_path, segmented_summary_path, manifest_path]:
            if output_path.exists():
                output_path.unlink()

    condition_rows = load_condition_rows(args.condition_rows)
    if args.condition:
        condition_rows = condition_rows.loc[condition_rows["condition"].eq(args.condition)].copy()
    manifest_lookup = load_manifest_lookup(args.manifest_csv)
    summary_paths = list(args.existing_summary_jsonl)
    if generated_summary_path.exists():
        summary_paths.append(generated_summary_path)
    segment_paths = list(args.existing_summary_segments_jsonl)
    if segmented_summary_path.exists():
        segment_paths.append(segmented_summary_path)

    summary_lookup = load_summary_lookup(summary_paths, args.summary_id)
    segmented_keys = load_segmented_summary_keys(segment_paths, args.summary_id)
    jobs = build_jobs(condition_rows, manifest_lookup, summary_lookup, segmented_keys, args.summary_id)
    print("Output directory:", out_dir)
    print_audit(jobs)

    runnable_jobs = [job for job in jobs if job["mode"] != "complete"]
    if args.existing_only:
        runnable_jobs = [job for job in runnable_jobs if job["mode"] == "segment_existing_summary"]
    if args.start:
        runnable_jobs = runnable_jobs[args.start :]
    if args.limit is not None:
        runnable_jobs = runnable_jobs[: args.limit]
    print("Runnable jobs this invocation:", len(runnable_jobs))

    if args.dry_run:
        return

    load_api_key(args.key_path)
    manifest_rows: list[dict[str, Any]] = []
    for job_index, job in enumerate(runnable_jobs, start=1):
        story_key = job["story_key"]
        condition = job["condition"]
        mode = job["mode"]
        try:
            if mode == "segment_existing_summary":
                summary_text = normalize_text(job["existing_summary"]["summary"])
                payload = call_openai_json(
                    SEGMENT_EXISTING_SUMMARY_PROMPT.format(
                        target_age_range=job["target_age_range"],
                        summary_text=summary_text,
                    ),
                    args.segment_model,
                )
                segments = payload.get("segments", [])
            elif mode == "generate_summary_and_segment":
                story_text = read_story_text(job["manifest"])
                if not story_text:
                    raise ValueError(f"No cleaned story text available for {story_key}")
                payload = call_openai_json(
                    SUMMARY_DETAIL_PROMPT.format(
                        target_age_range=job["target_age_range"],
                        story_text=story_text,
                    ),
                    args.summary_model,
                )
                summaries = payload.get("summaries") or []
                if not summaries:
                    raise ValueError("No summaries returned")
                selected_summary = summaries[0]
                summary_text = normalize_text(selected_summary.get("summary"))
                segments = selected_summary.get("segments", [])
                if not summary_text:
                    raise ValueError("Empty summary returned")
                append_jsonl(generated_summary_path, summary_output_record(job, args.summary_model, summary_text))
            else:
                continue

            append_jsonl(segmented_summary_path, segment_output_record(job, summary_text, segments))
            status = "segmented_existing_summary" if mode == "segment_existing_summary" else "generated_and_segmented_summary"
            manifest_rows.append(
                {
                    "story_key": story_key,
                    "condition": condition,
                    "target_age_range": job["target_age_range"],
                    "mode": mode,
                    "status": status,
                    "summary_segmented": True,
                    "n_segments": len(segments),
                }
            )
            print(f"[{job_index}/{len(runnable_jobs)}] {story_key} | {condition} | {status}")
        except Exception as error:
            manifest_rows.append(
                {
                    "story_key": story_key,
                    "condition": condition,
                    "target_age_range": job["target_age_range"],
                    "mode": mode,
                    "status": "error",
                    "error": str(error),
                }
            )
            print(f"[{job_index}/{len(runnable_jobs)}] {story_key} | {condition} | error: {error}")
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        if args.sleep > 0:
            time.sleep(args.sleep)

    print("Wrote:", generated_summary_path)
    print("Wrote:", segmented_summary_path)
    print("Wrote:", manifest_path)


if __name__ == "__main__":
    main()
