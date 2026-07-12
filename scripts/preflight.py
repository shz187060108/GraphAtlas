#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import tempfile

try:
    import fcntl  # type: ignore
except ImportError:  # Windows
    fcntl = None

import json
import platform
import os
import subprocess
import sys
from pathlib import Path


def _acquire_experiment_lock():
    default_lock = Path(tempfile.gettempdir()) / "graphatlas_experiments.lock"
    lock_path = Path(os.environ.get("GRAPHATLAS_LOCK_PATH", str(default_lock)))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            handle.seek(0)
            if lock_path.stat().st_size == 0:
                handle.write("0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except (BlockingIOError, OSError):
        print(
            f"Another GraphAtlas experiment or preflight process is already running (lock: {lock_path}).",
            file=sys.stderr,
        )
        handle.close()
        raise SystemExit(75)
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def main() -> None:
    lock_handle = _acquire_experiment_lock()
    parser = argparse.ArgumentParser(description="Validate environment and GraphAtlas experiment manifests.")
    parser.add_argument("--preset", default="smoke", help="Preset name or YAML path")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    root = Path(PROJECT_ROOT)

    # Run tests before importing PyTorch in this parent process. Forking a
    # process after OpenMP initialization can deadlock JVP tests on some Linux
    # builds, while this ordering is reliable across CPU and CUDA images.
    tests_exit_code = None
    if args.run_tests:
        completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, check=False)
        if completed.returncode:
            raise SystemExit(completed.returncode)
        # Replace this process with a clean preflight process. Importing PyTorch
        # after a completed PyTorch test subprocess can deadlock some OpenMP
        # runtimes even when the tests themselves passed.
        os.execv(
            sys.executable,
            [sys.executable, str(Path(__file__).resolve()), "--preset", args.preset],
        )

    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import torch
    import yaml

    from graphatlas.config import ExperimentConfig
    from graphatlas.experiments import run_preset
    from graphatlas.utils import save_json

    base = ExperimentConfig.from_yaml(root / "configs" / "base.yaml")
    base.validate()
    preset_path = Path(args.preset)
    if not preset_path.suffix:
        preset_path = root / "configs" / "presets" / f"{args.preset}.yaml"
    elif not preset_path.is_absolute():
        preset_path = root / preset_path
    preset_payload = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    manifest = run_preset(preset_path, dry_run=True)
    if manifest.empty and not bool(preset_payload.get("allow_empty", False)):
        raise RuntimeError("Preset produced an empty experiment manifest")
    duplicate_count = int(manifest.duplicated("run_id").sum())
    if duplicate_count:
        raise RuntimeError(f"Experiment manifest contains {duplicate_count} duplicate run IDs")

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
        "git_commit": _git_commit(root),
        "preset": preset_path.stem,
        "manifest_runs": int(len(manifest)),
        "manifest_datasets": sorted(manifest["dataset"].unique().tolist()) if not manifest.empty else [],
        "manifest_tasks": sorted(manifest["task"].unique().tolist()) if not manifest.empty else [],
        "manifest_models": sorted(manifest["model"].unique().tolist()) if not manifest.empty else [],
        "optional_empty_manifest": bool(manifest.empty),
        "tests_exit_code": tests_exit_code,
        "status": "passed",
    }
    save_json(report, root / "outputs" / "status" / f"preflight_{preset_path.stem}.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    lock_handle.close()


if __name__ == "__main__":
    main()
