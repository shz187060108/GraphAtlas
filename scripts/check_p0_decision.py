#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASELINES = (
    "graphatlas_no_metric",
    "graphatlas_no_cocycle",
    "ambient_vector_gnn",
    "geometry_moe",
)


def paired_summary(frame: pd.DataFrame, baseline: str) -> dict[str, object]:
    keys = [column for column in ("dataset", "task", "seed", "split") if column in frame]
    full = frame[frame["model"] == "graphatlas"][keys + ["test_metric"]]
    other = frame[frame["model"] == baseline][keys + ["test_metric"]]
    paired = full.merge(other, on=keys, suffixes=("_full", "_baseline")).dropna()
    delta = paired["test_metric_full"] - paired["test_metric_baseline"]
    t_p = float(stats.ttest_rel(paired["test_metric_full"], paired["test_metric_baseline"]).pvalue) if len(paired) >= 2 else float("nan")
    try:
        w_p = float(stats.wilcoxon(delta).pvalue) if len(paired) >= 2 and np.any(delta != 0) else float("nan")
    except ValueError:
        w_p = float("nan")
    return {
        "baseline": baseline,
        "n_pairs": len(paired),
        "mean_delta": float(delta.mean()) if len(delta) else float("nan"),
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Roman Empire P0 expansion decision.")
    parser.add_argument("--results", default="outputs/results/roman_p0.csv")
    parser.add_argument("--output-dir", default="outputs/reports/roman_p0")
    args = parser.parse_args()
    root = Path(PROJECT_ROOT)
    results = Path(args.results)
    if not results.is_absolute():
        results = root / results
    frame = pd.read_csv(results)

    blocking: list[str] = []
    reasons: list[str] = []
    unique_runs = int(frame["run_id"].nunique()) if "run_id" in frame else 0
    if unique_runs != 100 or len(frame) != 100:
        blocking.append(f"expected 100 unique runs, found {unique_runs} unique in {len(frame)} rows")
    split_counts = frame.groupby("model")["split"].nunique().to_dict() if {"model", "split"}.issubset(frame) else {}
    if len(split_counts) != 10 or any(count != 10 for count in split_counts.values()):
        blocking.append("each of the ten models must contain ten splits")

    comparisons = {name: paired_summary(frame, name) for name in BASELINES}
    for name in ("graphatlas_no_metric", "graphatlas_no_cocycle"):
        if not np.isfinite(comparisons[name]["mean_delta"]) or comparisons[name]["mean_delta"] < 0:
            blocking.append(f"full test metric is below {name}")
    if not np.isfinite(comparisons["ambient_vector_gnn"]["mean_delta"]) or comparisons["ambient_vector_gnn"]["mean_delta"] <= 0:
        blocking.append("full mean test metric does not exceed Ambient Vector GNN")

    means = frame.groupby("model", as_index=True).mean(numeric_only=True)
    path_full = float(means.at["graphatlas", "path_consistency_error"]) if "path_consistency_error" in means else float("nan")
    path_no_cocycle = float(means.at["graphatlas_no_cocycle", "path_consistency_error"]) if "path_consistency_error" in means else float("nan")
    metric_full = float(means.at["graphatlas", "metric_compatibility_error"]) if "metric_compatibility_error" in means else float("nan")
    metric_no_metric = float(means.at["graphatlas_no_metric", "metric_compatibility_error"]) if "metric_compatibility_error" in means else float("nan")
    if not (np.isfinite(path_full) and np.isfinite(path_no_cocycle) and path_full < path_no_cocycle):
        blocking.append("full path consistency is not below no-cocycle")
    if not (np.isfinite(metric_full) and np.isfinite(metric_no_metric) and metric_full < metric_no_metric):
        blocking.append("full metric compatibility is not below no-metric")

    full_rows = frame[frame["model"] == "graphatlas"]
    capacity_runtime = {
        "parameters_mean": float(full_rows["parameters"].mean()) if "parameters" in full_rows else None,
        "runtime_seconds_mean": float(full_rows["runtime_seconds"].mean()) if "runtime_seconds" in full_rows else None,
    }
    if not blocking:
        reasons.append("all Roman P0 expansion requirements are satisfied")
    payload = {
        "expand_to_more_datasets": not blocking,
        "reasons": reasons,
        "blocking_failures": blocking,
        "unique_runs": unique_runs,
        "split_counts": split_counts,
        "paired_comparisons": comparisons,
        "consistency": {
            "path_full": path_full,
            "path_no_cocycle": path_no_cocycle,
            "metric_full": metric_full,
            "metric_no_metric": metric_no_metric,
        },
        "full_model_capacity_runtime": capacity_runtime,
    }
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "p0_decision.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Roman Empire P0 decision",
        "",
        f"Expand to more datasets: **{payload['expand_to_more_datasets']}**",
        "",
        "## Blocking failures",
        "",
        *(f"- {item}" for item in blocking),
        "",
        "## Paired comparisons",
        "",
        pd.DataFrame(comparisons.values()).to_markdown(index=False),
    ]
    (output_dir / "p0_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
