#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from graphatlas.config import ExperimentConfig
from graphatlas.datasets import load_dataset
from graphatlas.experiments import _source_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one GraphAtlas job with its exact split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(PROJECT_ROOT)
    config = ExperimentConfig.from_yaml(args.config)
    data = load_dataset(config.dataset, config.train.seed)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        x=data.x.cpu().numpy(), y=data.y.cpu().numpy(),
        edge_index=data.edge_index.cpu().numpy(),
        train_mask=data.train_mask.cpu().numpy(), val_mask=data.val_mask.cpu().numpy(),
        test_mask=data.test_mask.cpu().numpy(), seed=np.asarray(config.train.seed),
        split=np.asarray(config.dataset.split), metric=np.asarray((data.metadata or {}).get("metric", "accuracy")),
        dataset_name=np.asarray(config.dataset.name),
    )
    sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    metadata = {
        "dataset_sha256": sha, "dataset": config.dataset.name, "split": config.dataset.split,
        "seed": config.train.seed, "feature_source": config.dataset.feature_source,
        "directed": config.dataset.directed, "metric": (data.metadata or {}).get("metric", "accuracy"),
        "graphatlas_source_hash": _source_fingerprint(root),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(dataset_path)


if __name__ == "__main__":
    main()
