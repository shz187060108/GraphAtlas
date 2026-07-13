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
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--sync-github", action="store_true")
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
