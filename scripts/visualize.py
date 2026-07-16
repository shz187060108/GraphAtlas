#!/usr/bin/env python
from __future__ import annotations

import os
from pathlib import Path

# Matplotlib may spend a long time rebuilding its font cache on shared or
# read-only home directories. Keep all caches inside the project so plotting
# is deterministic on Windows, Linux, containers, and cluster workers.
_PROJECT_HINT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_HINT / "outputs" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PROJECT_HINT / "outputs" / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from _bootstrap import PROJECT_ROOT  # noqa: F401

import argparse
import json
import sys

from graphatlas.visualization import visualize_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-ready GraphAtlas result figures.")
    parser.add_argument("--results", default="outputs/best_config_search/search_summary.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-model", default="graphatlas_c_oracle")
    args = parser.parse_args()
    manifest = visualize_results(Path(args.results), args.output_dir, target_model=args.target_model)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
