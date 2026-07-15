#!/usr/bin/env python
"""Build dense paper figures from existing results; never trains."""
from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401

import argparse
import json

from graphatlas.paper_figures_dense import build_dense_paper_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/results/all.csv")
    parser.add_argument("--output-dir", default="outputs/reports/paper_dense")
    parser.add_argument("--target-model", default="graphatlas_c_oracle")
    parser.add_argument("--stress-csv", "--coordinate-stress", dest="stress_csv", default="outputs/results/coordinate_stress.csv")
    parser.add_argument("--cases-csv", "--case-studies", dest="cases_csv", default=None)
    parser.add_argument("--figures", nargs="*", default=None)
    parser.add_argument("--formal-only", action="store_true")
    parser.add_argument("--include-screening", action="store_true")
    parser.add_argument("--include-confirmatory", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--formats", default="svg,pdf,png")
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--upper-bound-results", default=None)
    parser.add_argument("--include-upper-bound", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = build_dense_paper_figures(args.results, args.output_dir, target_model=args.target_model, stress_csv=args.stress_csv, cases_csv=args.cases_csv, upper_bound_results=args.upper_bound_results, include_upper_bound=args.include_upper_bound, bootstrap_reps=args.bootstrap_reps, bootstrap_seed=args.bootstrap_seed, formats=args.formats.split(","), dpi=args.dpi, figures=args.figures)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
