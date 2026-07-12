#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import csv
import itertools
import json
import shutil
from pathlib import Path

from graphatlas.experiments import (
    _job_config,
    _run_directory,
    _run_identity,
    _source_fingerprint,
)
from graphatlas.utils import load_yaml, save_yaml


def resolve_preset(value: str) -> Path:
    root = Path(PROJECT_ROOT)
    path = Path(value)
    if not path.suffix:
        path = root / "configs" / "presets" / f"{value}.yaml"
    elif not path.is_absolute():
        path = root / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a deterministic GraphAtlas experiment plan.")
    parser.add_argument("--preset", default="full")
    parser.add_argument("--plan", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    preset_path = resolve_preset(args.preset)
    preset = load_yaml(preset_path)
    root = Path(PROJECT_ROOT)
    base = load_yaml(root / preset["base_config"])
    source_hash = _source_fingerprint(root)
    jobs = list(itertools.product(preset["datasets"], preset["models"], preset["seeds"]))
    if args.limit is not None:
        jobs = jobs[: args.limit]

    manifests = root / "outputs" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    preset_name = preset_path.stem
    plan_path = Path(args.plan) if args.plan else manifests / f"{preset_name}.jsonl"
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    manifest_path = manifests / f"{preset_name}.csv"

    records: list[dict[str, object]] = []
    for dataset_override, model_override, seed in jobs:
        config = _job_config(base, preset, dataset_override, model_override, int(seed))
        config.validate()
        run_id, config_hash = _run_identity(config, source_hash)
        run_dir = _run_directory(root, config, config_hash)
        if args.force and run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        job_config = run_dir / "job_config.yaml"
        save_yaml(config.to_dict(), job_config)
        records.append(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "source_hash": source_hash,
                "dataset": config.dataset.name,
                "task": config.dataset.task,
                "model": config.model.name,
                "seed": int(seed),
                "split": int(config.dataset.split),
                "num_threads": int(config.train.num_threads),
                "run_dir": str(run_dir.relative_to(root)),
                "job_config": str(job_config.relative_to(root)),
            }
        )

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    fieldnames = list(records[0]) if records else [
        "run_id", "config_hash", "source_hash", "dataset", "task", "model", "seed",
        "split", "num_threads", "run_dir", "job_config",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps({
        "preset": preset_name,
        "preset_path": str(preset_path),
        "plan": str(plan_path),
        "manifest": str(manifest_path),
        "source_hash": source_hash,
        "runs": len(records),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
