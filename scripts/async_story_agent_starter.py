#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: Dict[str, Any]
    command_builder: Callable[[Dict[str, Any]], List[str]]


def _repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def _tool_schema(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def build_tool_registry() -> Dict[str, ToolSpec]:
    python_bin = os.environ.get("PYTHON_BIN", "python")
    return {
        "compile_youtube_catalog": ToolSpec(
            name="compile_youtube_catalog",
            description="Build the all-YouTube catalog with transcript scrape status, GPT title extraction status, and Amazon cross-reference columns.",
            schema=_tool_schema(
                "compile_youtube_catalog",
                "Compile the YouTube catalog used to track scraped videos and title extraction coverage.",
                {
                    "out": {"type": "string", "description": "Output CSV or CSV.GZ path."},
                    "with_amazon": {"type": "string", "description": "Optional Amazon merge CSV path."},
                },
                ["out"],
            ),
            command_builder=lambda args: [
                python_bin,
                _repo_path("scripts", "compile_youtube_catalog.py"),
                "--out",
                args["out"],
                *(
                    ["--with-amazon", args["with_amazon"]]
                    if args.get("with_amazon")
                    else []
                ),
            ],
        ),
        "build_master_dataset": ToolSpec(
            name="build_master_dataset",
            description="Build the story-level master dataset that joins Amazon, YouTube outputs, transcripts, and review_samples events.",
            schema=_tool_schema(
                "build_master_dataset",
                "Build the story master dataset used for downstream feature extraction and probes.",
                {
                    "base": {"type": "string", "description": "Base CSV/XLSX input."},
                    "out": {"type": "string", "description": "Output master CSV.GZ or parquet path."},
                    "out_index": {"type": "string", "description": "Output index CSV path."},
                },
                ["base", "out", "out_index"],
            ),
            command_builder=lambda args: [
                python_bin,
                _repo_path("scripts", "build_master_dataset.py"),
                "--base",
                args["base"],
                "--out",
                args["out"],
                "--out-index",
                args["out_index"],
            ],
        ),
        "analyze_text_features": ToolSpec(
            name="analyze_text_features",
            description="Compute story-level non-LLM features such as readability, AoA, frequency, sentiment, and concept/event features.",
            schema=_tool_schema(
                "analyze_text_features",
                "Run text feature extraction on a prepared master file.",
                {
                    "master": {"type": "string", "description": "Master CSV.GZ/parquet path."},
                    "out": {"type": "string", "description": "Output feature CSV path."},
                    "extra_nlp": {"type": "boolean", "description": "Enable extra NLP features."},
                    "concepts_from_events": {"type": "boolean", "description": "Extract concept/event features from review_samples event markup."},
                },
                ["master", "out"],
            ),
            command_builder=lambda args: [
                python_bin,
                _repo_path("scripts", "analyze_text_features.py"),
                "--master",
                args["master"],
                "--out",
                args["out"],
                *(["--extra-nlp"] if args.get("extra_nlp") else []),
                *(["--concepts-from-events"] if args.get("concepts_from_events") else []),
            ],
        ),
        "run_linear_probe": ToolSpec(
            name="run_linear_probe",
            description="Run layerwise age-group classification probes over saved vector activations.",
            schema=_tool_schema(
                "run_linear_probe",
                "Train and evaluate a linear probe over per-layer vector activations.",
                {
                    "vectors": {"type": "string", "description": "NPZ vectors path."},
                    "outdir": {"type": "string", "description": "Probe output directory."},
                    "bins": {"type": "string", "description": "Age bins such as 0-3,4-6,7-12."},
                    "features_csv": {"type": "string", "description": "Optional features CSV for label lookup and metadata."},
                },
                ["vectors", "outdir", "bins"],
            ),
            command_builder=lambda args: [
                python_bin,
                _repo_path("scripts", "run_linear_probe.py"),
                "--vectors",
                args["vectors"],
                "--outdir",
                args["outdir"],
                "--bins",
                args["bins"],
                *(
                    ["--features-csv", args["features_csv"]]
                    if args.get("features_csv")
                    else []
                ),
            ],
        ),
        "run_layerwise_regression_probe": ToolSpec(
            name="run_layerwise_regression_probe",
            description="Probe which layers encode a continuous confounder such as aoa_mean or readability.",
            schema=_tool_schema(
                "run_layerwise_regression_probe",
                "Train a layerwise linear regression probe for a selected continuous target.",
                {
                    "vectors": {"type": "string", "description": "NPZ vectors path."},
                    "features_csv": {"type": "string", "description": "Feature CSV containing the target column."},
                    "target": {"type": "string", "description": "Continuous target column name."},
                    "out_csv": {"type": "string", "description": "Output probe CSV path."},
                },
                ["vectors", "features_csv", "target", "out_csv"],
            ),
            command_builder=lambda args: [
                python_bin,
                _repo_path("scripts", "run_layerwise_regression_probe.py"),
                "--vectors",
                args["vectors"],
                "--features-csv",
                args["features_csv"],
                "--target",
                args["target"],
                "--out-csv",
                args["out_csv"],
            ],
        ),
    }


async def run_local_command(command: List[str], cwd: Path, timeout_s: int = 3600) -> Dict[str, Any]:
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {
            "ok": False,
            "error": "timeout",
            "command": command,
        }
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
        "command": command,
    }


class AsyncStoryAgent:
    BASE_SYSTEM_PROMPT = (
        "You are an autonomous research engineering agent for the story-features project. "
        "Use tools when they are needed. Prefer the smallest valid next step. "
        "When the task is complete, answer directly and do not call more tools."
    )

    def __init__(
        self,
        model: str,
        tool_registry: Dict[str, ToolSpec],
        reasoning_effort: str = "low",
        max_turns: int = 8,
        dry_run: bool = False,
        mock_mode: bool = False,
    ) -> None:
        self.model = model
        self.tool_registry = dict(tool_registry)
        self.tool_schemas = [spec.schema for spec in self.tool_registry.values()]
        self.reasoning_effort = reasoning_effort
        self.max_turns = max_turns
        self.dry_run = dry_run
        self.mock_mode = mock_mode

    async def call_model(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.mock_mode:
            return self.mock_response(messages)
        from openai import AsyncOpenAI  # imported lazily so dry runs do not require SDK

        client = AsyncOpenAI()
        response = await client.responses.create(
            model=self.model,
            instructions=self.BASE_SYSTEM_PROMPT,
            input=messages,
            tools=self.tool_schemas,
            reasoning={"effort": self.reasoning_effort},
            store=False,
        )
        output = []
        for item in response.output:
            payload = {"type": item.type}
            for key in ["name", "arguments", "call_id", "role", "content"]:
                if hasattr(item, key):
                    value = getattr(item, key)
                    if value is not None:
                        payload[key] = value
            output.append(payload)
        return {
            "output": output,
            "output_text": response.output_text,
        }

    def mock_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        have_catalog = any(
            item.get("type") == "function_call_output"
            and "compile_youtube_catalog" in item.get("output", "")
            for item in messages
        )
        if not have_catalog:
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "compile_youtube_catalog",
                        "arguments": json.dumps(
                            {
                                "out": _repo_path("data", "processed", "youtube_catalog.csv.gz"),
                                "with_amazon": _repo_path(
                                    "data", "amazon_data", "all_extracted_titles_with_amazon.csv"
                                ),
                            }
                        ),
                        "call_id": "mock-call-1",
                    }
                ],
                "output_text": "",
            }
        return {
            "output": [],
            "output_text": "The YouTube catalog has been compiled. Next, inspect the title-extraction coverage and the missing videos.",
        }

    async def execute_tool_call(self, call: Dict[str, Any]) -> Dict[str, Any]:
        name = call["name"]
        args = json.loads(call["arguments"])
        if self.dry_run:
            result = {
                "ok": True,
                "dry_run": True,
                "tool": name,
                "args": args,
                "command": self.tool_registry[name].command_builder(args),
            }
        else:
            if name not in self.tool_registry:
                result = {"ok": False, "error": f"unknown tool: {name}"}
            else:
                command = self.tool_registry[name].command_builder(args)
                result = await run_local_command(command, cwd=REPO_ROOT)
        return {
            "type": "function_call_output",
            "call_id": call["call_id"],
            "output": json.dumps({name: result}),
        }

    async def run(self, user_prompt: str) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]
        for turn in range(self.max_turns):
            response = await self.call_model(messages)
            function_calls = [item for item in response["output"] if item.get("type") == "function_call"]
            if not function_calls:
                messages.append({"role": "assistant", "content": response.get("output_text", "")})
                return {
                    "messages": messages,
                    "final_text": response.get("output_text", ""),
                    "turns": turn + 1,
                }

            for call in function_calls:
                messages.append(
                    {
                        "type": "function_call",
                        "name": call["name"],
                        "arguments": call["arguments"],
                        "call_id": call["call_id"],
                    }
                )
            outputs = await asyncio.gather(*(self.execute_tool_call(call) for call in function_calls))
            messages.extend(outputs)

        raise RuntimeError(f"agent exceeded max_turns={self.max_turns}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Async starter agent for story-features tool integration practice.")
    ap.add_argument(
        "--prompt",
        type=str,
        default="Compile the YouTube catalog and tell me what fraction of scraped transcript videos already have successful title extraction.",
    )
    ap.add_argument("--model", type=str, default="gpt-5")
    ap.add_argument("--reasoning-effort", type=str, default="low")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="Do not execute tools; just show planned calls.")
    ap.add_argument("--mock", action="store_true", help="Do not call the API; use a built-in mock tool-use cycle.")
    ap.add_argument("--list-tools", action="store_true")
    return ap


async def main_async() -> None:
    args = build_parser().parse_args()
    tool_registry = build_tool_registry()
    if args.list_tools:
        print(json.dumps({name: spec.schema for name, spec in tool_registry.items()}, indent=2))
        return

    agent = AsyncStoryAgent(
        model=args.model,
        tool_registry=tool_registry,
        reasoning_effort=args.reasoning_effort,
        max_turns=args.max_turns,
        dry_run=args.dry_run,
        mock_mode=args.mock,
    )
    result = await agent.run(args.prompt)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main_async())
