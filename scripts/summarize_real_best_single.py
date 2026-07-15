#!/usr/bin/env python
"""Rebuild concise run-level and dataset-level real-best-single summaries."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"


def main() -> None:
    paths = sorted(
        path for path in RESULTS.glob("real_best_single_*.csv")
        if not path.name.endswith("_concise.csv")
        and path.name != "real_best_single_summary.csv"
    )
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise SystemExit("No completed real_best_single result files were found.")
    runs = pd.concat(frames, ignore_index=True, sort=False)
    keys = [key for key in ("dataset", "model", "seed", "split") if key in runs]
    runs = runs.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
    runs.to_csv(RESULTS / "real_best_single.csv", index=False)

    columns = [
        column for column in (
            "dataset", "model", "metric_name", "seed", "split", "val_metric",
            "test_metric", "best_epoch", "runtime_seconds", "parameters", "device",
            "num_nodes", "num_edges", "transportability_mean", "distortion_mean",
            "fraction_q_below_025", "fraction_q_above_075",
        ) if column in runs
    ]
    summary = runs.loc[:, columns].copy()
    summary.insert(3, "completed_runs", 1)
    summary.to_csv(RESULTS / "real_best_single_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nCompleted datasets: {summary['dataset'].nunique()}")


if __name__ == "__main__":
    main()
