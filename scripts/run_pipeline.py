#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from tqdm.auto import tqdm


ROOT = Path(PROJECT_ROOT)
OUTPUTS = ROOT / "outputs"


def _command(*parts: str) -> list[str]:
    return [sys.executable, "-u", *parts]


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _runner_args(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    if args.force:
        values.append("--force")
    if args.continue_on_error:
        values.append("--continue-on-error")
    if args.limit is not None:
        values.extend(["--limit", str(args.limit)])
    return values


def _preset_runner(preset: str, common: Sequence[str]) -> list[str]:
    """Select a runnable isolated preset launcher for the current platform."""
    if os.name == "nt":
        return _command("scripts/run_all.py", "--preset", preset, *common)
    return [str(ROOT / "scripts" / "run_preset_isolated.sh"), preset, *common]


def _stage_plan(args: argparse.Namespace) -> list[tuple[str, Sequence[str]]]:
    common = _runner_args(args)
    stages: list[tuple[str, Sequence[str]]] = []

    if args.preset:
        preset = Path(args.preset).stem
        stages.append((f"Validate {preset} manifest", _command("scripts/preflight.py", "--preset", args.preset)))
        if not args.skip_download:
            stages.append((f"Download {preset} datasets", _command("scripts/download_data.py", "--preset", args.preset)))
        stages.append((f"Run {preset} matrix", _preset_runner(args.preset, common)))
        result = f"outputs/results/{preset}.csv"
        report = f"outputs/reports/{preset}"
    elif args.mode == "mechanism":
        stages.append(("Validate mechanism manifest", _command("scripts/preflight.py", "--preset", "mechanism")))
        stages.append(("Run three-seed mechanism matrix", _preset_runner("mechanism", common)))
        result = "outputs/results/mechanism.csv"
        report = "outputs/reports/mechanism"
    elif args.mode == "smoke":
        stages.append(("Validate smoke manifest", _command("scripts/preflight.py", "--preset", "smoke")))
        stages.append(("Run smoke experiment matrix", _preset_runner("smoke", common)))
        result = "outputs/results/smoke.csv"
        report = "outputs/reports/smoke"
    elif args.mode == "kill":
        stages.append(("Validate elimination manifest", _command("scripts/preflight.py", "--preset", "kill")))
        stages.append(("Run three-seed elimination matrix", _preset_runner("kill", common)))
        result = "outputs/results/kill.csv"
        report = "outputs/reports/kill"
    elif args.mode == "paper":
        stages.append(("Validate paper manifest", _command("scripts/preflight.py", "--preset", "paper")))
        if not args.skip_download:
            stages.append(("Download paper datasets", _command("scripts/download_data.py", "--preset", "paper")))
        stages.append(("Run node-classification matrix", _preset_runner("paper", common)))
        result = "outputs/results/paper.csv"
        report = "outputs/reports/paper"
    elif args.mode == "link":
        stages.append(("Validate link manifest", _command("scripts/preflight.py", "--preset", "link")))
        if not args.skip_download:
            stages.append(("Download link datasets", _command("scripts/download_data.py", "--preset", "link")))
        stages.append(("Run link-prediction matrix", _preset_runner("link", common)))
        result = "outputs/results/link.csv"
        report = "outputs/reports/link"
    else:
        stages.extend(
            [
                ("Validate paper manifest", _command("scripts/preflight.py", "--preset", "paper")),
                ("Validate link manifest", _command("scripts/preflight.py", "--preset", "link")),
            ]
        )
        if not args.skip_download:
            stages.append(
                (
                    "Download all configured datasets",
                    _command("scripts/download_data.py", "--preset", "paper", "--preset", "link"),
                )
            )
        stages.extend(
            [
                ("Run node-classification matrix", _preset_runner("paper", common)),
                ("Run link-prediction matrix", _preset_runner("link", common)),
                (
                    "Combine task results",
                    _command(
                        "scripts/combine_results.py",
                        "--input",
                        "outputs/results/paper.csv",
                        "--input",
                        "outputs/results/link.csv",
                        "--output",
                        "outputs/results/complete.csv",
                    ),
                ),
            ]
        )
        result = "outputs/results/complete.csv"
        report = "outputs/reports/complete"

    stages.extend(
        [
            (
                "Compute statistics and claim gates",
                _command("scripts/summarize.py", "--results", result, "--output-dir", report, "--no-plots"),
            ),
            (
                "Generate publication figures",
                _command("scripts/visualize.py", "--results", result, "--output-dir", f"{report}/figures"),
            ),
            (
                "Validate curated published results",
                _command("scripts/validate_published_results.py"),
            ),
            (
                "Build provenance-aware published comparison tables",
                _command(
                    "scripts/build_comparison_tables.py",
                    "--results", result,
                    "--output-dir", f"{report}/published_comparison",
                ),
            ),
        ]
    )
    # Keep tests in a fresh final process. Some OpenMP builds are less reliable
    # when a parent process imports PyTorch and then starts additional workers.
    if not args.skip_tests:
        stages.extend(
            [
                ("Run unit and integration tests", _command("-m", "pytest", "-q")),
                ("Run visualization subprocess test", _command("-m", "pytest", "-q", "tests/test_visualization.py", "-m", "visualization")),
            ]
        )
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete resumable GraphAtlas research pipeline.")
    parser.add_argument("--mode", choices=["mechanism", "smoke", "kill", "paper", "link", "full"], default="full")
    parser.add_argument("--preset", default=None, help="Run any configs/presets/<name>.yaml through the full report pipeline.")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit jobs per experiment preset for debugging.")
    args = parser.parse_args()

    stages = _stage_plan(args)
    status_name = Path(args.preset).stem if args.preset else args.mode
    status_path = OUTPUTS / "status" / f"pipeline_{status_name}.json"
    started = time.time()
    completed: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    progress = tqdm(stages, desc=f"GraphAtlas {status_name} pipeline", unit="stage")
    for index, (name, command) in enumerate(progress, start=1):
        progress.set_postfix(stage=name[:30])
        print(f"\n[{index}/{len(stages)}] {name}\n$ {' '.join(command)}", flush=True)
        stage_start = time.time()
        stage_environment = os.environ.copy()
        stage_environment["GRAPHATLAS_DEFER_SYNC"] = "1"
        process = subprocess.run(command, cwd=ROOT, env=stage_environment, check=False)
        record = {
            "index": index,
            "name": name,
            "command": list(command),
            "returncode": process.returncode,
            "elapsed_seconds": time.time() - stage_start,
        }
        completed.append(record)
        if process.returncode != 0:
            failures.append(record)
        _atomic_json(
            {
                "mode": status_name,
                "total_stages": len(stages),
                "completed_stages": len(completed),
                "failed_stages": len(failures),
                "elapsed_seconds": time.time() - started,
                "current_stage": name,
                "complete": len(completed) == len(stages) and not failures,
                "stages": completed,
            },
            status_path,
        )
        if process.returncode != 0 and not args.continue_on_error:
            raise SystemExit(process.returncode)

    print(f"\nPipeline status: {status_path}")
    if failures:
        print(f"Completed with {len(failures)} failed stage(s).")
        raise SystemExit(2)
    subprocess.run(
        [sys.executable, "-u", "scripts/sync_github.py", "--message", f"Complete {status_name} pipeline"],
        cwd=ROOT,
        env=os.environ.copy(),
        check=True,
    )
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
