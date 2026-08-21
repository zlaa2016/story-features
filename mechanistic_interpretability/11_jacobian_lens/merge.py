from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independently fitted Jacobian-lens shards.")
    parser.add_argument("--input-glob", default="mechanistic_interpretability/outputs/11_jacobian_lens/shards/lens_*.pt")
    parser.add_argument("--jlens-repo", type=Path, default=Path(os.environ.get("JLENS_REPO", f"/scratch/{os.environ.get('USER', 'user')}/tools/jacobian-lens")))
    parser.add_argument("--output", type=Path, default=Path("mechanistic_interpretability/outputs/11_jacobian_lens/jacobian_lens.pt"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.jlens_repo))
    import jlens

    paths = [Path(path) for path in sorted(glob.glob(args.input_glob)) if ".checkpoint." not in path]
    if not paths:
        raise FileNotFoundError(f"No lens shards match {args.input_glob}")
    lenses = [jlens.JacobianLens.load(str(path)) for path in paths]
    merged = jlens.JacobianLens.merge(lenses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(dir=args.output.parent, suffix=".pt")
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        merged.save(str(temporary))
        temporary.replace(args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Wrote {args.output}: prompts={merged.n_prompts} layers={merged.source_layers}")


if __name__ == "__main__":
    main()
