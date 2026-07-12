#!/usr/bin/env python
from __future__ import annotations

try:
    from _bootstrap import PROJECT_ROOT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT_ROOT

import argparse
from dataclasses import replace

from graphatlas.config import DatasetConfig
from graphatlas.dataset_audit import audit_graph, write_audit
from graphatlas.datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit graph datasets before benchmark execution.")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--root", default="data")
    parser.add_argument("--output-dir", default="outputs/reports/data_audit")
    args = parser.parse_args()
    records = []
    for name in args.dataset:
        config = DatasetConfig(name=name, root=args.root)
        data = load_dataset(config, seed=0)
        records.append(audit_graph(data, name))
    csv_path, json_path = write_audit(records, PROJECT_ROOT / args.output_dir)
    print(csv_path)
    print(json_path)

if __name__ == "__main__":
    main()
