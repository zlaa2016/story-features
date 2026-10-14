#!/usr/bin/env python3
"""Read-only status of the experiments on this machine: which outputs exist, how big, how recent, plus recent job states.

Standard library only, so it runs on a login node with the system python3:
    python3 hpc/check_status.py
Looks in the new layout (results/...) and in the old one (mechanistic_interpretability/outputs/...) because outputs written
before the reorganisation have not necessarily been copied yet. Deletes and moves nothing.
"""

import collections
import glob
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = "mechanistic_interpretability/outputs"
R = "results"
EXTR = f"{R}/features/representation_extraction/causal_lm/detail_extraction"
# (label, new location, old location, file patterns that must exist for the step to count as done)
STEPS = [
    ("00 cohort", f"{R}/features/summaries/dataset_index", f"{OLD}/00_dataset_index", ["cohort.jsonl", "cohort_metadata.csv"]),
    ("02 extraction gpt_oss contextual", f"{EXTR}/gpt_oss_20b_global_current", f"{OLD}/02_representation_extraction/gpt_oss_20b_global_current", ["contextual_detail_features_*.npz", "contextual_detail_metadata_*.csv"]),
    ("02 extraction gpt_oss isolated", f"{EXTR}/gpt_oss_20b_detail_controls", f"{OLD}/02_representation_extraction/gpt_oss_20b_detail_controls", ["isolated_detail_features_*.npz", "isolated_detail_metadata_*.csv"]),
    ("02 extraction gpt_oss deletion", f"{EXTR}/gpt_oss_20b_detail_controls", f"{OLD}/02_representation_extraction/gpt_oss_20b_detail_controls", ["deletion_features_*.npz", "deletion_metadata_*.csv"]),
    ("02 extraction qwen3 contextual", f"{EXTR}/qwen3_8b_aug27_global", f"{OLD}/02_representation_extraction/qwen3_8b_aug27_global", ["contextual_detail_features_*.npz"]),
    ("02 extraction qwen3 isolated/deletion", f"{EXTR}/qwen3_8b_aug27_detail_controls", f"{OLD}/02_representation_extraction/qwen3_8b_aug27_detail_controls", ["isolated_detail_features_*.npz", "deletion_features_*.npz"]),
    ("03 native generation", f"{R}/features/summaries/native_generation", f"{OLD}/03_native_generation", []),
    ("13 detail alignment (mpnet)", f"{R}/analysis/natural_language/detail_alignment", f"{OLD}/13_structured_detail_alignment", ["original_detail_retention.csv", "shard_*/original_detail_retention.csv"]),
    ("14 detail characteristics", f"{R}/analysis/natural_language/detail_characteristics", f"{OLD}/14_detail_characteristics", []),
    ("15 mobius integration", f"{R}/analysis/natural_language/mobius_integration", f"{OLD}/15_mobius_detail_integration", []),
    ("04 layerwise probes", f"{R}/analysis/representation/probes/layerwise_detail", f"{OLD}/04_layerwise_probes", ["story_split.csv"]),
    ("05 causal patching", f"{R}/analysis/representation/causal_patching", f"{OLD}/05_causal_patching", []),
    ("06 activation steering", f"{R}/analysis/representation/steering/activation_steering", f"{OLD}/06_activation_steering", []),
    ("07 router analysis", f"{R}/analysis/representation/router", f"{OLD}/07_router_analysis", []),
    ("08 demixing", f"{R}/analysis/representation/demixing", f"{OLD}/08_demixing_normalization", []),
    ("09 directional jacobian", f"{R}/analysis/representation/jacobian/directional_jacobian", f"{OLD}/09_directional_jacobian", []),
    ("10 behavior vs representation", f"{R}/analysis/representation/behavior_vs_representation", f"{OLD}/10_behavior_representation_dissociation", []),
    ("11 jacobian lens", f"{R}/analysis/representation/jacobian/jacobian_lens", f"{OLD}/11_jacobian_lens", []),
    ("12 sparse autoencoder", f"{R}/analysis/representation/sparse_autoencoder", f"{OLD}/12_sparse_autoencoder", []),
    ("16 compression geometry", f"{R}/analysis/representation/geometry/compression_geometry", f"{OLD}/16_causal_compression_geometry", []),
    ("17 covariance geometry", f"{R}/analysis/representation/geometry/covariance", f"{OLD}/17_covariance_detail_geometry", []),
    ("17 cross-model concordance", f"{R}/analysis/representation/geometry/cross_model_concordance", f"{OLD}/17_cross_model_concordance", []),
    ("18 prefix probes", f"{R}/analysis/representation/probes/prefix_detail", f"{OLD}/18_prefix_detail_probes", []),
    ("NEW span search, natural language", f"{R}/analysis/natural_language/detail_compression/span_search", "", ["summaries.csv", "details.csv"]),
    ("NEW span search, sentence level", f"{R}/analysis/natural_language/detail_compression/span_search_sentence", "", ["summaries.csv", "details.csv"]),
    ("NEW span search, contextual", f"{R}/analysis/representation/detail_compression/span_search/contextual_mean", "", ["summaries.csv", "details.csv"]),
    ("NEW span search, isolated", f"{R}/analysis/representation/detail_compression/span_search/isolated_mean", "", ["summaries.csv", "details.csv"]),
    ("NEW contextual-vs-isolated shift", f"{R}/analysis/representation/detail_compression/contextualization_shift", "", ["details.csv"]),
]


def stats(path: Path):
    files = [Path(d) / f for d, _, fs in os.walk(path, followlinks=True) for f in fs]
    if not files:
        return 0, 0, None
    size = sum(f.stat().st_size for f in files if f.exists())
    newest = max(f.stat().st_mtime for f in files if f.exists())
    return len(files), size, newest


def fmt_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def main():
    print(f"{'step':40s} {'where':6s} {'files':>7s} {'size':>9s}  {'newest':10s}  required files")
    for label, new, old, required in STEPS:
        for where, rel in (("new", new), ("old", old)):
            if not rel:
                continue
            path = ROOT / rel
            if not path.exists():
                continue
            n, size, newest = stats(path)
            if n == 0:
                continue
            have = []
            for pat in required:
                count = len(glob.glob(str(path / pat)))
                have.append(f"{pat}={count}")
            when = time.strftime("%Y-%m-%d", time.localtime(newest)) if newest else "-"
            print(f"{label:40s} {where:6s} {n:7d} {fmt_size(size):>9s}  {when:10s}  {' '.join(have)}")
            break
        else:
            print(f"{label:40s} {'MISSING':6s}")
    print("\nJobs in the last 45 days (by name and final state):")
    try:
        out = subprocess.run(["sacct", "-X", "-u", os.environ.get("USER", ""), "-S", "now-45days", "-n", "-P", "--format=JobName,State"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        out = ""
    if not out.strip():
        print("  (sacct not available here)")
        return
    table = collections.defaultdict(collections.Counter)
    for line in out.splitlines():
        name, _, state = line.partition("|")
        table[name][state.split()[0] if state else "?"] += 1
    for name in sorted(table):
        print(f"  {name:28s} " + "  ".join(f"{s}={c}" for s, c in table[name].most_common()))


if __name__ == "__main__":
    main()
