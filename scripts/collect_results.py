#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import json
from pathlib import Path

import pandas as pd

from graphatlas.experiments import _validate_run_artifacts, concise_results_frame
from graphatlas.utils import save_json


def load_plan(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect completed GraphAtlas planned jobs.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--preset-name", required=True)
    args = parser.parse_args()

    root = Path(PROJECT_ROOT)
    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    records = load_plan(plan_path)
    metrics: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for record in records:
        run_dir = root / str(record["run_dir"])
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            missing.append({**record, "reason": "metrics.json missing"})
            continue
        try:
            _validate_run_artifacts(run_dir)
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            for key in ("run_id", "config_hash", "source_hash"):
                if payload.get(key) != record[key]:
                    raise RuntimeError(f"{key} mismatch")
        except Exception as error:  # noqa: BLE001
            missing.append({**record, "reason": repr(error)})
            continue
        metrics.append(payload)

    outputs = root / "outputs"
    results_dir = outputs / "results"
    status_dir = outputs / ".work" / "status"
    results_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(metrics)
    preset_results = results_dir / f"{args.preset_name}.csv"
    if not frame.empty:
        sort_columns = [c for c in ["dataset", "task", "model", "split", "seed"] if c in frame]
        if sort_columns:
            frame = frame.sort_values(sort_columns).reset_index(drop=True)
        concise = concise_results_frame(frame)
        concise.to_csv(preset_results, index=False)
        concise.to_csv(results_dir / "latest.csv", index=False)
    status = {
        "preset": args.preset_name,
        "expected_runs": len(records),
        "completed_runs": len(metrics),
        "failed_runs": len(missing),
        "remaining_runs": len(records) - len(metrics) - len(missing),
        "progress_fraction": len(metrics) / max(len(records), 1),
        "current": None,
        "complete": len(metrics) == len(records) and not missing,
    }
    save_json(status, status_dir / f"experiment_{args.preset_name}.json")
    failures_path = status_dir / f"failures_{args.preset_name}.json"
    if missing:
        save_json({"failures": missing}, failures_path)
    elif failures_path.exists():
        failures_path.unlink()
    print(json.dumps({
        "results": str(preset_results),
        "completed": len(metrics),
        "expected": len(records),
        "missing": len(missing),
    }, ensure_ascii=False))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
