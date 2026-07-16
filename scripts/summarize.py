#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from graphatlas.reporting import submission_readiness_gates, summarize_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GraphAtlas experiment results.")
    parser.add_argument("--results", default="outputs/best_config_search/search_summary.csv")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--strict-gates", action="store_true")
    parser.add_argument("--strict-submission-gates", action="store_true")
    parser.add_argument("--target-model", default="graphatlas_c_oracle")
    args = parser.parse_args()
    summary, _, gates = summarize_results(
        Path(args.results), plots=not args.no_plots, output_dir=args.output_dir, target_model=args.target_model
    )
    display_columns = [
        column
        for column in [
            "dataset", "model", "test_metric_count", "test_metric_mean", "test_metric_std",
            "test_metric_ci95", "boundary_accuracy_mean", "invariance_error_nonlinear_mean",
            "runtime_seconds_mean",
        ]
        if column in summary
    ]
    print(summary[display_columns].to_string(index=False))
    print(f"Scientific gates: {gates['passed']} passed, {gates['failed']} failed, {gates['not_evaluated']} not evaluated")
    if args.strict_gates and not gates["all_decisive_gates_pass"]:
        raise SystemExit(2)
    submission = submission_readiness_gates(pd.read_csv(args.results), args.target_model)
    print(f"Submission gates: {submission['passed']} passed, {submission['failed']} failed")
    if args.strict_submission_gates and not submission["all_submission_gates_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
