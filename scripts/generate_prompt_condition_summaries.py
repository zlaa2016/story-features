#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd


MODEL = "gpt-4o-mini"
N_SUMMARIES = 5


NO_AGE_PROMPT = """Summarize story full text.

Create {n_summaries} independent summaries of approximately 150 words each for the following story.

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
    {{"summary_id": 1, "summary": "..."}},
    {{"summary_id": 2, "summary": "..."}},
    {{"summary_id": 3, "summary": "..."}},
    {{"summary_id": 4, "summary": "..."}},
    {{"summary_id": 5, "summary": "..."}}
  ]
}}

Story:
{story_text}
"""


AGE_PROMPT = """Summarize story full text.

The following story is intended for children in this age range: {age_range}.

Create {n_summaries} independent summaries of approximately 150 words each for an audience in this age range.

Each summary should:
- Use vocabulary and sentence structure appropriate for children in the target age range
- Keep important character names, places, repeated phrases, and story-specific terms from the original story when needed
- Include all key events in their correct sequence
- Mention the main characters and their actions
- Preserve the central message or lesson of the story, if explicitly present
- Capture the cause-and-effect relationships between events
- Include the beginning, middle, and end of the narrative arc
- Focus only on what explicitly happens in the story
- Avoid adding opinions, interpretations, motives, morals, or elements not present in the original text
- Maintain a straightforward, factual tone throughout

Do not copy long passages from the story. Do not include bullet points. Do not explain your reasoning. Do not mention the age range in the summary itself.

Return valid JSON in this exact shape:
{{
  "age_range": "{age_range}",
  "summaries": [
    {{"summary_id": 1, "summary": "..."}},
    {{"summary_id": 2, "summary": "..."}},
    {{"summary_id": 3, "summary": "..."}},
    {{"summary_id": 4, "summary": "..."}},
    {{"summary_id": 5, "summary": "..."}}
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


def call_openai(prompt: str, model: str, temperature: float) -> dict:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        text={"format": {"type": "json_object"}},
        store=False,
    )
    return json.loads(response.output_text)


def build_prompt(condition: str, story_text: str, age_range: str) -> str:
    if condition == "age":
        return AGE_PROMPT.format(
            n_summaries=N_SUMMARIES,
            age_range=age_range or "unknown",
            story_text=story_text,
        )
    return NO_AGE_PROMPT.format(n_summaries=N_SUMMARIES, story_text=story_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate age/no-age GPT summaries for story texts.")
    parser.add_argument("--zip-path", default="cleaned_idchecked_punctuated.zip")
    parser.add_argument("--master-path", default="data/processed/master_stories.parquet")
    parser.add_argument("--key-path", default="../hartley_api_key.txt")
    parser.add_argument("--limit", type=int, default=3, help="Number of stories to process.")
    parser.add_argument("--start", type=int, default=0, help="Start offset in loaded story rows.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between API calls.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_SUMMARY_MODEL", MODEL))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("SUMMARY_TEMPERATURE", "0.8")))
    parser.add_argument("--output-prefix", default="summary_prompt_test")
    parser.add_argument("--dry-run", action="store_true", help="Write prompts without calling the API.")
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = Path("pipeline_output") / f"{args.output_prefix}_{args.model.replace('/', '_')}_{run_timestamp}"
    prompt_dir = pipeline_dir / "prompts"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(exist_ok=True)
    out_jsonl = pipeline_dir / "summary_outputs.jsonl"

    print(f"Output directory: {pipeline_dir}")
    load_api_key(Path(args.key_path))

    rows = load_story_rows(Path(args.zip_path), Path(args.master_path))
    rows = rows.iloc[args.start : args.start + args.limit].copy()
    print(f"Loaded {len(rows)} stories for this run")

    with out_jsonl.open("w", encoding="utf-8") as out:
        for _, row in rows.iterrows():
            story_id = str(row["story_id"])
            age_range = "" if pd.isna(row.get("age_range")) else str(row.get("age_range", ""))
            story_text = str(row["story_text"]).strip()

            for condition in ("no_age", "age"):
                prompt = build_prompt(condition, story_text, age_range)
                prompt_path = prompt_dir / f"{story_id}_{condition}.txt"
                prompt_path.write_text(prompt, encoding="utf-8")

                record = {
                    "story_id": story_id,
                    "condition": condition,
                    "model": args.model,
                    "age_range": age_range,
                    "prompt_path": str(prompt_path),
                }
                if args.dry_run:
                    record["response"] = None
                else:
                    record["response"] = call_openai(prompt, args.model, args.temperature)
                    time.sleep(args.sleep)

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(f"wrote {story_id} / {condition}")

    print(f"Results: {out_jsonl}")


if __name__ == "__main__":
    main()
