#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured official external baseline wrapper.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(PROJECT_ROOT)
    registry = yaml.safe_load((root / "external_baselines" / "registry.yaml").read_text(encoding="utf-8"))
    method = registry.get("methods", {}).get(args.method)
    if method is None:
        raise SystemExit(f"Unknown external method: {args.method}")
    if method.get("status") != "configured":
        raise SystemExit(f"External method {args.method} is unconfigured; no result was generated")
    if not method.get("repository_url") or not method.get("pinned_commit"):
        raise SystemExit(f"External method {args.method} lacks a verified repository and pinned commit")
    wrapper = root / "external_baselines" / "wrappers" / f"{args.method}.py"
    if not wrapper.exists():
        raise SystemExit(f"Configured wrapper is missing: {wrapper}")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(wrapper), "--input", str(Path(args.export_dir) / "dataset.npz"),
        "--dataset", args.dataset, "--seed", str(args.seed), "--split", str(args.split),
    ]
    started = time.time()
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(f"Official wrapper failed with exit code {completed.returncode}")
    try:
        parsed = json.loads(completed.stdout.strip().splitlines()[-1])
        test_metric = float(parsed["test_metric"])
        metric_name = str(parsed["metric_name"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not parse official wrapper output: {error}") from error
    result = {
        "dataset": args.dataset, "task": "node_classification", "model": args.method,
        "seed": args.seed, "split": args.split, "metric_name": metric_name,
        "test_metric": test_metric, "source_repository": method["repository_url"],
        "source_commit": method["pinned_commit"], "protocol": "direct_same_split",
        "runtime_seconds": time.time() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
