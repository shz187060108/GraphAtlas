#!/usr/bin/env python
"""One entry point for generated GraphAtlas-C stages; never syncs by default."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "smoke": "graphatlasc_smoke", "screening": "graphatlasc_phase_screening",
    "beta": "graphatlasc_beta_sweep", "counterfactual": "graphatlasc_counterfactual",
    "real": "graphatlasc_real_screening", "confirmatory": "graphatlasc_main_confirmatory",
    "ogb": "graphatlasc_ogb_arxiv",
    "h2gb": "atn_h2gb_screening",
    "h2gb_confirmatory": "atn_h2gb_confirmatory",
    "h2gb_mechanism": "atn_h2gb_mechanism",
    "phase_dense_screening": "graphatlasc_phase_dense_screening",
    "phase_dense_confirmatory": "graphatlasc_phase_dense_confirmatory",
    "routing_causal": "graphatlasc_routing_causal",
    "depth_dynamics": "graphatlasc_depth_dynamics",
    "scaling": "graphatlasc_scaling",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--sync-github", action="store_true")
    parser.add_argument("--build-paper-figures", action="store_true")
    parser.add_argument("--build-dense-paper-figures", action="store_true")
    parser.add_argument("--paper-figures-only", action="store_true")
    args = parser.parse_args()
    if args.paper_figures_only:
        subprocess.run([sys.executable, "-u", "scripts/build_paper_figures.py", "--results", "outputs/best_config_search/search_summary.csv"], cwd=ROOT, check=False)
        return
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for stage in stages:
        preset = STAGES[stage]
        command = [sys.executable, "-u", "scripts/run_all.py", "--preset", preset]
        if args.dry_run:
            command.append("--dry-run")
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        if args.sync_github:
            command.append("--sync-github")
        subprocess.run(command, cwd=ROOT, check=True)
    if args.build_paper_figures:
        subprocess.run([sys.executable, "-u", "scripts/summarize.py", "--results", "outputs/best_config_search/search_summary.csv", "--no-plots"], cwd=ROOT, check=False)
        subprocess.run([sys.executable, "-u", "scripts/visualize.py", "--results", "outputs/best_config_search/search_summary.csv", "--output-dir", "outputs/reports/latest/figures"], cwd=ROOT, check=False)
        subprocess.run([sys.executable, "-u", "scripts/build_paper_figures.py", "--results", "outputs/best_config_search/search_summary.csv"], cwd=ROOT, check=False)
    if args.build_dense_paper_figures:
        subprocess.run([sys.executable, "-u", "scripts/build_dense_paper_figures.py", "--results", "outputs/best_config_search/search_summary.csv"], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
