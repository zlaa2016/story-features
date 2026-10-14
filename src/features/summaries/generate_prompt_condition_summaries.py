#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from age_utils import format_age_range, normalize_age_text, parse_age_bounds, parse_age_midpoint


MODEL = "gpt-4o-mini"
N_SUMMARIES = 5


NO_AGE_PROMPT = """Summarize story full text.

Create exactly {n_summaries} independent summaries of approximately 150 words each for the following story.

Each summary should:
- Include all key events in their correct sequence
- Mention the main characters and their actions
- Preserve the central message or lesson of the story, if explicitly present
- Capture the cause-and-effect relationships between events
- Include the beginning, middle, and end of the narrative arc
- Use clear, simple language
- Focus only on what explicitly happens in the story
- Avoid adding opinions, interpretations, motives, morals, or elements not present in the original text
- Maintain a straightforward, factual tone throughout

Do not copy long passages from the story. Do not include bullet points. Do not explain your reasoning.

Return valid JSON in this exact shape:
{{
  "summaries": [
{summary_json_examples}
  ]
}}

Story:
{story_text}
"""


AGE_PROMPT = """Summarize story full text.

Create exactly {n_summaries} independent summaries of approximately 150 words each for children in this target age setting: {target_age_spec}.

Each summary should:
- Use vocabulary and sentence structure appropriate for children in this target age setting
- Keep important character names, places, repeated phrases, and story-specific terms from the original story when needed
- Include all key events in their correct sequence
- Mention the main characters and their actions
- Preserve the central message or lesson of the story, if explicitly present
- Capture the cause-and-effect relationships between events
- Include the beginning, middle, and end of the narrative arc
- Focus only on what explicitly happens in the story
- Avoid adding opinions, interpretations, motives, morals, or elements not present in the original text
- Maintain a straightforward, factual tone throughout

Do not copy long passages from the story. Do not include bullet points. Do not explain your reasoning. Do not mention the target age setting in the summary itself.

Return valid JSON in this exact shape:
{{
  "target_age_setting": "{target_age_spec}",
  "summaries": [
{summary_json_examples}
  ]
}}

Story:
{story_text}
"""


def load_api_key(key_path: Path) -> None:
    if os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY already set")
        return
    if key_path.exists():
        key = key_path.read_text(encoding="utf-8").strip()
        os.environ["OPENAI_API_KEY"] = key
        print(f"API key loaded from {key_path} ({'*' * 4}{key[-4:]})")
    else:
        print(f"API key file not found at {key_path}; set OPENAI_API_KEY manually")


def parse_json_object_loose(text: str) -> dict:
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


def read_zip_stories(zip_path: Path) -> dict[str, str]:
    stories: dict[str, str] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir() or name.startswith((".", "__")) or not name.lower().endswith(".txt"):
                continue
            raw = zf.read(info.filename)
            for encoding in ("utf-8", "latin-1", "cp1252"):
                try:
                    text = raw.decode(encoding).strip()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = raw.decode("utf-8", errors="replace").strip()
            story_id = Path(name).stem
            if text and (story_id not in stories or len(text) > len(stories[story_id])):
                stories[story_id] = text
    return stories


def normalize_story_id(value: object) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.strip()


def load_master(master_path: Path) -> pd.DataFrame:
    if master_path.suffix == ".parquet":
        return pd.read_parquet(master_path)
    return pd.read_csv(master_path)


def load_story_rows(zip_path: Path, master_path: Path) -> pd.DataFrame:
    master = load_master(master_path)
    if "story_id_str" in master.columns:
        master["story_id_key"] = master["story_id_str"].map(normalize_story_id)
    elif "story_id" in master.columns:
        master["story_id_key"] = master["story_id"].map(normalize_story_id)
    else:
        master["story_id_key"] = master.index.astype(str)
    if "video_id" in master.columns:
        missing_id = master["story_id_key"].isin(["", "nan", "None"]) | master["story_id_key"].isna()
        master.loc[missing_id, "story_id_key"] = master.loc[missing_id, "video_id"].map(normalize_story_id)
    missing_id = master["story_id_key"].isin(["", "nan", "None"]) | master["story_id_key"].isna()
    master.loc[missing_id, "story_id_key"] = master.index[missing_id].astype(str)

    if "age_range" not in master.columns:
        master["age_range"] = master.get("Reading Age", "")

    if zip_path.exists():
        stories = read_zip_stories(zip_path)
        rows = pd.DataFrame({"story_id_key": list(stories), "story_text": list(stories.values())})
        rows["id_prefix"] = rows["story_id_key"].str.extract(r"^(\d+)", expand=False)
        age_cols = master[["story_id_key", "age_range"]].dropna().drop_duplicates("story_id_key")
        rows = rows.merge(age_cols, left_on="id_prefix", right_on="story_id_key", how="left", suffixes=("", "_meta"))
        rows["story_id"] = rows["story_id_key"]
        rows["age_range"] = rows["age_range"].fillna("")
        return rows[["story_id", "age_range", "story_text"]]

    text_col = "transcript_best" if "transcript_best" in master.columns else "yt_cleaned_subtitles"
    rows = master[["story_id_key", "age_range", text_col]].rename(
        columns={"story_id_key": "story_id", text_col: "story_text"}
    )
    rows = rows.dropna(subset=["story_text"]).drop_duplicates("story_id")
    return rows


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes"])


def load_cleaned_manifest_rows(
    detail_manifest_path: Path,
    catalog_path: Path,
    *,
    age_source_filter: str | None,
    require_status: str = "segmented",
) -> pd.DataFrame:
    manifest = pd.read_csv(detail_manifest_path, low_memory=False)
    if require_status and "status" in manifest.columns:
        manifest = manifest.loc[manifest["status"].fillna("").astype(str).str.strip().eq(require_status)].copy()
    manifest["video_id"] = manifest["video_id"].fillna("").astype(str).str.strip()
    manifest["story_id"] = manifest.get("story_key", manifest["video_id"]).fillna("").astype(str).str.strip()
    manifest["story_text"] = manifest["cleaned_path"].map(
        lambda p: Path(p).read_text(encoding="utf-8").strip() if isinstance(p, str) and p and Path(p).exists() else ""
    )
    manifest = manifest.loc[manifest["story_text"].ne("")].copy()

    catalog = pd.read_csv(catalog_path, low_memory=False)
    catalog["video_id"] = catalog["video_id"].fillna("").astype(str).str.strip()
    if "has_amazon_age" in catalog.columns:
        catalog = catalog.loc[normalize_bool_series(catalog["has_amazon_age"])].copy()

    merge_cols = ["video_id"]
    for col in [
        "best_Reading Age",
        "best_Reading Age Source",
        "best_Title",
        "best_Author",
        "video_title",
        "youtube_url",
    ]:
        if col in catalog.columns:
            merge_cols.append(col)
    catalog = catalog[merge_cols].drop_duplicates("video_id")
    rows = manifest.merge(catalog, on="video_id", how="left")
    rows["age_range_raw"] = rows.get("best_Reading Age", "").fillna("").astype(str).str.strip()
    rows["age_range"] = rows["age_range_raw"].map(normalize_age_text)
    rows["age_source"] = rows.get("best_Reading Age Source", "").fillna("").astype(str).str.strip().str.lower()

    if age_source_filter:
        wanted = age_source_filter.strip().lower()
        rows = rows.loc[rows["age_source"].eq(wanted)].copy()

    rows = rows.loc[rows["age_range"].ne("")].copy()
    rows["target_age"] = rows["age_range"].map(parse_age_midpoint)
    return rows.drop_duplicates("story_id")


def call_openai(prompt: str, model: str, temperature: float) -> dict:
    from openai import OpenAI

    client = OpenAI()
    last_err = None
    for _attempt in range(3):
        kwargs = {
            "model": model,
            "input": prompt,
            "text": {"format": {"type": "json_object"}},
            "store": False,
        }
        if temperature is not None and not str(model).startswith("gpt-5.6"):
            kwargs["temperature"] = temperature
        response = client.responses.create(**kwargs)
        try:
            return parse_json_object_loose(response.output_text)
        except json.JSONDecodeError as err:
            last_err = err
            time.sleep(1.0)
    if last_err is not None:
        raise last_err
    raise RuntimeError("OpenAI response parsing failed")


def build_summary_json_examples(n_summaries: int) -> str:
    return ",\n".join(
        f'    {{"summary_id": {summary_id}, "summary": "..."}}'
        for summary_id in range(1, n_summaries + 1)
    )


def build_prompt(condition: str, story_text: str, age_range: str, n_summaries: int) -> str:
    summary_json_examples = build_summary_json_examples(n_summaries)
    if condition == "age":
        return AGE_PROMPT.format(
            n_summaries=n_summaries,
            target_age_spec=age_range,
            summary_json_examples=summary_json_examples,
            story_text=story_text,
        )
    return NO_AGE_PROMPT.format(
        n_summaries=n_summaries,
        summary_json_examples=summary_json_examples,
        story_text=story_text,
    )


def parse_target_age_grid(value: str | None) -> list[int]:
    if not value:
        return []
    out: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def filter_rows_by_age_bounds(rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if rows.empty or "age_range" not in rows.columns:
        return rows

    work = rows.copy()
    work["__age_bounds__"] = work["age_range"].map(parse_age_bounds)
    work = work.loc[work["__age_bounds__"].notna()].copy()

    if getattr(args, "require_lower_minus_3_at_least", None) is not None:
        threshold = int(args.require_lower_minus_3_at_least)
        work = work.loc[work["__age_bounds__"].map(lambda b: (b[0] - 3) >= threshold)].copy()

    if getattr(args, "require_upper_plus_3_at_most", None) is not None:
        threshold = int(args.require_upper_plus_3_at_most)
        work = work.loc[work["__age_bounds__"].map(lambda b: (b[1] + 3) <= threshold)].copy()

    return work.drop(columns=["__age_bounds__"])


def build_generation_jobs(rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.condition != "age":
        return rows.copy()

    fixed_target_ages = parse_target_age_grid(args.target_age_grid)
    records: list[dict] = []
    if fixed_target_ages:
        for _, row in rows.iterrows():
            base = row.to_dict()
            source_mid = row.get("target_age", None)
            for target_age in fixed_target_ages:
                rec = dict(base)
                rec["source_age_midpoint"] = source_mid
                rec["target_age"] = int(target_age)
                rec["target_age_label"] = f"fixed_{int(target_age)}"
                rec["target_age_range"] = str(int(target_age))
                if source_mid is None or pd.isna(source_mid):
                    rec["abs_age_gap"] = None
                else:
                    rec["abs_age_gap"] = abs(float(target_age) - float(source_mid))
                records.append(rec)
        return pd.DataFrame.from_records(records)

    if args.target_age_relative:
        lower = int(args.target_age_min)
        upper = int(args.target_age_max)
        for _, row in rows.iterrows():
            base = row.to_dict()
            source_mid = row.get("target_age", None)
            bounds = parse_age_bounds(str(row.get("age_range", "")))
            if bounds is None:
                continue
            src_lower, src_upper = bounds
            younger_age = src_lower - 3
            older_age = src_upper + 3
            candidate_specs = [
                ("younger", younger_age if younger_age >= lower else None),
                ("actual", format_age_range(max(lower, src_lower), min(upper, src_upper))),
                ("older", older_age if older_age <= upper else None),
            ]
            seen: set[str] = set()
            for label, target_spec in candidate_specs:
                if args.relative_label and label != args.relative_label:
                    continue
                if target_spec is None:
                    continue
                target_spec_str = str(target_spec)
                if target_spec_str in seen:
                    continue
                seen.add(target_spec_str)
                rec = dict(base)
                rec["source_age_midpoint"] = source_mid
                rec["source_age_bounds"] = format_age_range(src_lower, src_upper)
                rec["target_age"] = parse_age_midpoint(target_spec_str)
                rec["target_age_label"] = label
                rec["target_age_range"] = target_spec_str
                rec["abs_age_gap"] = None if source_mid is None or pd.isna(source_mid) or rec["target_age"] is None else abs(float(rec["target_age"]) - float(source_mid))
                records.append(rec)
        return pd.DataFrame.from_records(records)

    return rows.copy()


def load_existing_job_keys(paths: list[str]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                summaries = (record.get("response") or {}).get("summaries") or []
                if not any(str(summary.get("summary", "")).strip() for summary in summaries if isinstance(summary, dict)):
                    continue
                story_id = str(record.get("story_id", "")).strip()
                condition = str(record.get("condition", "")).strip()
                target_age = str(record.get("target_age_label") or record.get("target_age_range") or record.get("target_age") or "").strip()
                if story_id and condition:
                    keys.add((story_id, condition, target_age))
    return keys


def load_existing_story_ids(paths: list[str]) -> set[str]:
    story_ids: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                story_id = str(record.get("story_id", "")).strip()
                if story_id:
                    story_ids.add(story_id)
    return story_ids


def filter_existing_jobs(rows: pd.DataFrame, existing_keys: set[tuple[str, str, str]], condition: str) -> pd.DataFrame:
    if not existing_keys or rows.empty:
        return rows
    work = rows.copy()
    work["__story_id__"] = work["story_id"].astype(str).str.strip()
    if "target_age_label" in work.columns:
        work["__target_age__"] = work["target_age_label"].fillna("").astype(str).str.strip()
    elif "target_age_range" in work.columns:
        work["__target_age__"] = work["target_age_range"].fillna("").astype(str).str.strip()
    else:
        work["__target_age__"] = work.get("target_age", "").map(lambda x: "" if pd.isna(x) else str(int(x)) if isinstance(x, (int, float)) else str(x))
    keep_mask = []
    for _, row in work.iterrows():
        key = (row["__story_id__"], condition, row["__target_age__"])
        keep_mask.append(key not in existing_keys)
    work = work.loc[keep_mask].copy()
    return work.drop(columns=["__story_id__", "__target_age__"])


def filter_existing_stories(rows: pd.DataFrame, existing_story_ids: set[str]) -> pd.DataFrame:
    if not existing_story_ids or rows.empty:
        return rows
    work = rows.copy()
    work["__story_id__"] = work["story_id"].astype(str).str.strip()
    work = work.loc[~work["__story_id__"].isin(existing_story_ids)].copy()
    return work.drop(columns=["__story_id__"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate age/no-age GPT summaries for story texts.")
    parser.add_argument("--zip-path", default="cleaned_idchecked_punctuated.zip")
    parser.add_argument("--master-path", default="data/processed/master_stories.parquet")
    parser.add_argument("--detail-manifest-path", default=None, help="Use cleaned detail-manifest rows instead of zip/master input.")
    parser.add_argument("--catalog-path", default="data/processed/youtube_catalog.csv.gz", help="Catalog used to attach best Amazon age metadata.")
    parser.add_argument("--age-source-filter", default=None, help="Optional age source filter, e.g. customer or publisher.")
    parser.add_argument("--condition", choices=["age", "no_age", "both"], default="both")
    parser.add_argument("--key-path", default="../hartley_api_key.txt")
    parser.add_argument("--limit", type=int, default=3, help="Number of stories to process.")
    parser.add_argument("--start", type=int, default=0, help="Start offset in loaded story rows.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between API calls.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_SUMMARY_MODEL", MODEL))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("SUMMARY_TEMPERATURE", "0.8")))
    parser.add_argument("--output-prefix", default="summary_prompt_test")
    parser.add_argument("--n-summaries", type=int, default=N_SUMMARIES)
    parser.add_argument("--target-age-grid", default="", help="Comma-separated fixed target ages, e.g. 3,5,7,9,11,13,15,18.")
    parser.add_argument("--target-age-relative", action="store_true", help="Generate summaries at source age rounded, source-3, and source+3.")
    parser.add_argument(
        "--relative-label",
        choices=["actual", "younger", "older"],
        default=None,
        help="When using --target-age-relative, generate only this relative condition.",
    )
    parser.add_argument("--target-age-min", type=int, default=2)
    parser.add_argument("--target-age-max", type=int, default=18)
    parser.add_argument("--require-lower-minus-3-at-least", type=int, default=None, help="Keep only stories whose source lower bound minus 3 is at least this value.")
    parser.add_argument("--require-upper-plus-3-at-most", type=int, default=None, help="Keep only stories whose source upper bound plus 3 is at most this value.")
    parser.add_argument("--skip-existing-jsonl", action="append", default=[], help="Existing summary JSONL files whose story_id/condition/target_age combos should be skipped.")
    parser.add_argument("--skip-existing-story-only", action="store_true", help="Skip an entire story if it appears at all in any --skip-existing-jsonl file.")
    parser.add_argument("--dry-run", action="store_true", help="Write prompts without calling the API.")
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = Path("results/features/summaries/runs") / f"{args.output_prefix}_{args.model.replace('/', '_')}_{run_timestamp}"
    prompt_dir = pipeline_dir / "prompts"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(exist_ok=True)
    out_jsonl = pipeline_dir / "summary_outputs.jsonl"

    print(f"Output directory: {pipeline_dir}")
    load_api_key(Path(args.key_path))

    if args.detail_manifest_path:
        rows = load_cleaned_manifest_rows(
            Path(args.detail_manifest_path),
            Path(args.catalog_path),
            age_source_filter=args.age_source_filter,
        )
    else:
        rows = load_story_rows(Path(args.zip_path), Path(args.master_path))
    rows = rows.iloc[args.start : args.start + args.limit].copy()
    rows = filter_rows_by_age_bounds(rows, args)
    rows = build_generation_jobs(rows, args)
    if args.skip_existing_story_only:
        existing_story_ids = load_existing_story_ids(args.skip_existing_jsonl)
        rows = filter_existing_stories(rows, existing_story_ids)
    else:
        existing_keys = load_existing_job_keys(args.skip_existing_jsonl)
        if args.condition == "both":
            rows_age = filter_existing_jobs(rows, existing_keys, "age")
            rows_no_age = filter_existing_jobs(rows, existing_keys, "no_age")
            rows = pd.concat([rows_age, rows_no_age]).drop_duplicates().reset_index(drop=True)
        else:
            rows = filter_existing_jobs(rows, existing_keys, args.condition)
    print(f"Loaded {len(rows)} generation jobs for this run")

    plan_cols = [col for col in [
        "story_id", "video_id", "video_title", "best_Title", "best_Author", "age_range", "age_source", "target_age"
    ] if col in rows.columns]
    rows[plan_cols].to_csv(pipeline_dir / "input_rows.csv", index=False)

    with out_jsonl.open("w", encoding="utf-8") as out:
        for _, row in rows.iterrows():
            story_id = str(row["story_id"])
            age_range = "" if pd.isna(row.get("age_range")) else str(row.get("age_range", ""))
            story_text = str(row["story_text"]).strip()

            if args.condition == "both":
                conditions = ("no_age", "age")
            else:
                conditions = (args.condition,)
            for condition in conditions:
                target_prompt_age = str(row.get("target_age_range", age_range)) if condition == "age" else age_range
                prompt = build_prompt(condition, story_text, target_prompt_age, args.n_summaries)
                prompt_suffix = f"_{row.get('target_age_label')}" if condition == "age" and row.get("target_age_label") else ""
                prompt_path = prompt_dir / f"{story_id}_{condition}{prompt_suffix}.txt"
                prompt_path.write_text(prompt, encoding="utf-8")

                record = {
                    "story_id": story_id,
                    "video_id": row.get("video_id", ""),
                    "video_title": row.get("video_title", ""),
                    "condition": condition,
                    "model": args.model,
                    "age_range": age_range,
                    "age_source": row.get("age_source", ""),
                    "target_age": row.get("target_age", None),
                    "target_age_range": row.get("target_age_range", age_range),
                    "target_age_label": row.get("target_age_label", ""),
                    "source_age_midpoint": row.get("source_age_midpoint", row.get("target_age", None)),
                    "source_age_bounds": row.get("source_age_bounds", age_range),
                    "abs_age_gap": row.get("abs_age_gap", None),
                    "prompt_path": str(prompt_path),
                }
                if args.dry_run:
                    record["response"] = None
                else:
                    record["response"] = call_openai(prompt, args.model, args.temperature)
                    time.sleep(args.sleep)

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(f"wrote {story_id} / {condition} / {record.get('target_age_label','')} / {record.get('target_age_range','')}")

    print(f"Results: {out_jsonl}")


if __name__ == "__main__":
    main()
