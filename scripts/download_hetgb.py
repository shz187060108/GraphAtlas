#!/usr/bin/env python
"""Explicitly download and validate official HeTGB NPZ files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

FILES = {
    "texas": "Texas.npz",
    "actor": "Actor.npz",
    "amazon": "Amazon.npz",
    "cornell": "Cornell.npz",
    "wisconsin": "Wisconsin.npz",
}
REQUIRED = {"edges", "node_features", "node_labels", "train_masks", "val_masks", "test_masks"}
BASE = "https://huggingface.co/datasets/0219shujie/HeTGB/resolve/main"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED - set(archive.files)
        if missing:
            raise ValueError(f"missing required arrays: {sorted(missing)}")
        features = archive["node_features"]
        labels = archive["node_labels"]
        edges = archive["edges"]
        if features.ndim != 2 or labels.shape[0] != features.shape[0]:
            raise ValueError("node_features and node_labels have inconsistent shapes")
        if edges.ndim != 2 or 2 not in edges.shape:
            raise ValueError("edges must have shape [2,E] or [E,2]")
        for name in ("train_masks", "val_masks", "test_masks"):
            masks = archive[name]
            if features.shape[0] not in masks.shape:
                raise ValueError(f"{name} does not include the node dimension")
        return {"num_nodes": int(features.shape[0]), "feature_dim": int(features.shape[1])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(FILES), default=sorted(FILES))
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    failures = []
    for name in args.datasets:
        destination = args.root / f"hetgb_{name}" / "raw" / f"hetgb_{name}.npz"
        try:
            if destination.exists():
                details = validate(destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False, suffix=".npz") as handle:
                    temporary = Path(handle.name)
                try:
                    urllib.request.urlretrieve(f"{BASE}/{FILES[name]}", temporary)
                    details = validate(temporary)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            manifest = {
                "dataset": f"hetgb_{name}",
                "source": "0219shujie/HeTGB",
                "source_file": FILES[name],
                "sha256": digest(destination),
                **details,
            }
            manifest_path = destination.parent / "manifest.json"
            temporary_manifest = manifest_path.with_suffix(".json.tmp")
            temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary_manifest, manifest_path)
            print(f"hetgb_{name}: ok")
        except Exception as error:
            failures.append(f"hetgb_{name}: {error}")
            print(f"ERROR hetgb_{name}: {error}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
