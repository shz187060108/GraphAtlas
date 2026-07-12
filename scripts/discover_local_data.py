#!/usr/bin/env python
from __future__ import annotations

try:
    from _bootstrap import PROJECT_ROOT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT_ROOT

import argparse
import json
from pathlib import Path

from graphatlas.datasets.local import discover_local_datasets, write_discovery_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover graph datasets already present on local disks.")
    parser.add_argument("--search-root", action="append", required=True)
    parser.add_argument("--output", default="outputs/manifests/local_candidates.json")
    args = parser.parse_args()
    candidates = discover_local_datasets(args.search_root)
    path = write_discovery_manifest(candidates, PROJECT_ROOT / args.output)
    print(json.dumps({"found": len(candidates), "manifest": str(path)}, ensure_ascii=False))
    for item in candidates:
        print(f"{item.dataset:20s} {item.format:16s} {item.path}")

if __name__ == "__main__":
    main()
