#!/usr/bin/env python
from __future__ import annotations

try:
    from _bootstrap import PROJECT_ROOT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT_ROOT

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Select model variants by validation metric only.")
    parser.add_argument("--results", default="outputs/results/tuning.csv")
    parser.add_argument("--output", default="outputs/reports/tuning/selected_variants.csv")
    args = parser.parse_args()
    frame = pd.read_csv(PROJECT_ROOT / args.results)
    required = {"dataset", "model", "model_family", "val_metric"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    aggregated = (
        frame.groupby(["dataset", "model_family", "model"], as_index=False)
        .agg(val_metric_mean=("val_metric", "mean"), val_metric_std=("val_metric", "std"), runs=("val_metric", "count"))
    )
    selected = aggregated.loc[aggregated.groupby(["dataset", "model_family"])["val_metric_mean"].idxmax()].copy()
    selected = selected.sort_values(["dataset", "model_family"]).reset_index(drop=True)
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    print(output)

if __name__ == "__main__":
    main()
