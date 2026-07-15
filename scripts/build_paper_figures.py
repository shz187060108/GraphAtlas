#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401

import argparse
import json
from pathlib import Path

from graphatlas.paper_figures import build_paper_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Build high-density Atlas Transport Network paper figures (SVG only).")
    parser.add_argument("--results", default="outputs/results/all.csv")
    parser.add_argument("--output-dir", default="outputs/reports/paper_figures")
    parser.add_argument("--target-model", default="graphatlas_c_oracle")
    parser.add_argument("--style", choices=["nature"], default="nature")
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--supplementary-only", action="store_true")
    parser.add_argument("--include-case-studies", action="store_true", help="Reserved: skips gracefully without compact case tensors.")
    parser.add_argument("--include-hetgb", action="store_true")
    parser.add_argument("--include-link", action="store_true")
    parser.add_argument("--include-ogb", action="store_true")
    parser.add_argument("--include-upper-bound", action="store_true", help="Include explicitly test-selected/upper-bound rows in this exploratory report.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.main_only and args.supplementary_only:
        raise SystemExit("--main-only and --supplementary-only cannot be combined")
    report = build_paper_figures(
        Path(args.results), args.output_dir, args.target_model,
        main_only=args.main_only, supplementary_only=args.supplementary_only,
        include_hetgb=args.include_hetgb, include_link=args.include_link, include_ogb=args.include_ogb,
        include_upper_bound=args.include_upper_bound,
    )
    if args.include_case_studies:
        report["case_studies"] = {"generated": False, "reason": "case tensors were not requested from run artifacts"}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
