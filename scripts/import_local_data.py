#!/usr/bin/env python
from __future__ import annotations

try:
    from _bootstrap import PROJECT_ROOT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT_ROOT

import argparse
import json
from pathlib import Path

from graphatlas.datasets.local import LocalDatasetCandidate, convert_candidate
from graphatlas.datasets.real import validate_dataset_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert discovered local graph datasets to GraphAtlas NPZ format.")
    parser.add_argument("--manifest", default="outputs/manifests/local_candidates.json")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--destination", default="data")
    parser.add_argument("--prefer", choices=["first", "largest", "smallest"], default="first")
    args = parser.parse_args()
    records = json.loads((PROJECT_ROOT / args.manifest).read_text(encoding="utf-8"))
    if args.dataset:
        allowed = {name.lower().replace("-", "_") for name in args.dataset}
        records = [row for row in records if row["dataset"] in allowed]
    grouped: dict[str, list[dict]] = {}
    for row in records:
        grouped.setdefault(row["dataset"], []).append(row)
    selected = []
    for name, rows in grouped.items():
        if args.prefer == "largest":
            rows.sort(key=lambda row: row["size_bytes"], reverse=True)
        elif args.prefer == "smallest":
            rows.sort(key=lambda row: row["size_bytes"])
        selected.append(rows[0])
    failures = []
    for row in selected:
        try:
            path = convert_candidate(LocalDatasetCandidate(**row), PROJECT_ROOT / args.destination)
            info = validate_dataset_file(path)
            print(f"imported {row['dataset']}: {path} {info}")
        except Exception as error:
            failures.append({"dataset": row["dataset"], "path": row["path"], "error": repr(error)})
            print(f"FAILED {row['dataset']}: {error}")
    failure_path = PROJECT_ROOT / "outputs" / "status" / "local_import_failures.json"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
