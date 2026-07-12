#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exactly one job from a materialized GraphAtlas plan.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--index", required=True, type=int, help="One-based plan index")
    args = parser.parse_args()
    root = Path(PROJECT_ROOT)
    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    records = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.index < 1 or args.index > len(records):
        raise SystemExit(f"Index must be between 1 and {len(records)}")
    record = records[args.index - 1]
    run_dir = root / record["run_dir"]
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if all(metrics.get(k) == record[k] for k in ("run_id", "config_hash", "source_hash")):
                print(json.dumps({"status": "skipped", "index": args.index, "run_id": record["run_id"]}))
                return
        except Exception:
            pass
    command = [
        sys.executable, "-u", str(root / "scripts" / "train.py"),
        "--config", str(root / record["job_config"]),
        "--run-dir", str(run_dir),
        "--run-id", record["run_id"],
        "--config-hash", record["config_hash"],
        "--source-hash", record["source_hash"],
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    threads = str(record.get("num_threads", 1))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = threads
    print(json.dumps({"status": "running", "index": args.index, "run_id": record["run_id"]}), flush=True)
    completed = subprocess.run(command, cwd=root, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print(json.dumps({"status": "completed", "index": args.index, "run_id": record["run_id"]}), flush=True)


if __name__ == "__main__":
    main()
