#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect direct same-split external baseline results.")
    parser.add_argument("--runs", default="outputs/external_runs")
    parser.add_argument("--output", default="outputs/results/external_direct.csv")
    parser.add_argument("--graphatlas-results", default=None)
    parser.add_argument("--combined-output", default="outputs/results/submission_complete.csv")
    args = parser.parse_args()
    root = Path(PROJECT_ROOT)
    runs = root / args.runs if not Path(args.runs).is_absolute() else Path(args.runs)
    rows = []
    for path in runs.rglob("metrics.json") if runs.exists() else []:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") == "direct_same_split":
            rows.append(payload)
    frame = pd.DataFrame(rows)
    output = root / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    if args.graphatlas_results:
        graphatlas_path = root / args.graphatlas_results if not Path(args.graphatlas_results).is_absolute() else Path(args.graphatlas_results)
        combined = pd.concat([pd.read_csv(graphatlas_path), frame], ignore_index=True, sort=False)
        combined_output = root / args.combined_output if not Path(args.combined_output).is_absolute() else Path(args.combined_output)
        combined.to_csv(combined_output, index=False)
    print(f"collected {len(frame)} direct external runs")


if __name__ == "__main__":
    main()
