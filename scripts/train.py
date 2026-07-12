#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from threadpoolctl import threadpool_limits

from graphatlas.config import ExperimentConfig
from graphatlas.datasets import load_dataset
from graphatlas.trainer import Trainer
from graphatlas.utils import configure_torch_threads, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one GraphAtlas experiment.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config-hash", default=None)
    parser.add_argument("--source-hash", default=None)
    args = parser.parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    config.validate()
    configure_torch_threads(config.train.num_threads)
    run_dir = Path(args.run_dir) if args.run_dir else (
        Path(config.train.output_dir)
        / config.dataset.name
        / config.dataset.task
        / config.model.name
        / f"split_{config.dataset.split}"
        / f"seed_{config.train.seed}"
    )
    with threadpool_limits(limits=config.train.num_threads):
        data = load_dataset(config.dataset, config.train.seed)
        _, metrics = Trainer(config).fit(data, run_dir)
    for key, value in (
        ("run_id", args.run_id),
        ("config_hash", args.config_hash),
        ("source_hash", args.source_hash),
    ):
        if value is not None:
            metrics[key] = value
    save_json(metrics, run_dir / "metrics.json")
    concise_keys = (
        "dataset",
        "task",
        "model",
        "seed",
        "split",
        "test_metric",
        "boundary_accuracy",
        "invariance_error_nonlinear",
        "runtime_seconds",
    )
    print(json.dumps({key: metrics.get(key) for key in concise_keys}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 - ensure worker errors reach the parent immediately
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    # Some PyTorch/OpenMP builds can stall during interpreter shutdown after
    # Jacobian diagnostics. All artifacts are durable at this point.
    os._exit(0)
