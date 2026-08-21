from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect lightweight HPC reports into a Git-friendly folder.")
    parser.add_argument("--outputs", type=Path, default=Path("mechanistic_interpretability/outputs"))
    parser.add_argument("--destination", type=Path, default=Path("mechanistic_interpretability/reports/latest"))
    parser.add_argument("--max-bytes", type=int, default=5_000_000)
    args = parser.parse_args()

    allowed = {".csv", ".json", ".txt"}
    copied = 0
    for path in args.outputs.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed or path.stat().st_size > args.max_bytes:
            continue
        if path.name.startswith(("metadata_", "detail_")):
            continue
        relative = path.relative_to(args.outputs)
        destination = args.destination / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied += 1
    print(f"Copied {copied} reports to {args.destination}")


if __name__ == "__main__":
    main()
