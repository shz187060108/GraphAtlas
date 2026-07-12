from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from graphatlas.datasets.local import LocalDatasetCandidate, convert_candidate, discover_local_datasets
from graphatlas.datasets.real import load_real_dataset, validate_dataset_file


def test_discover_and_convert_generic_npz(tmp_path: Path):
    source = tmp_path / "PHSGCL" / "data" / "Cora" / "cora_source.npz"
    source.parent.mkdir(parents=True)
    np.savez_compressed(
        source,
        x=np.eye(8, dtype=np.float32),
        y=np.arange(8, dtype=np.int64) % 2,
        edge_index=np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
        train_mask=np.asarray([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool),
        val_mask=np.asarray([0, 0, 0, 1, 1, 0, 0, 0], dtype=bool),
        test_mask=np.asarray([0, 0, 0, 0, 0, 1, 1, 1], dtype=bool),
    )
    candidates = discover_local_datasets([tmp_path])
    assert any(item.dataset == "cora" for item in candidates)
    candidate = next(item for item in candidates if item.dataset == "cora")
    destination = convert_candidate(candidate, tmp_path / "standardized")
    info = validate_dataset_file(destination)
    assert info["num_nodes"] == 8
    data = load_real_dataset("cora", tmp_path / "standardized", split=0)
    assert data.num_nodes == 8
    assert data.num_classes == 2


def test_unknown_standardized_dataset_can_load(tmp_path: Path):
    root = tmp_path / "data"
    path = root / "new_benchmark" / "raw" / "new_benchmark.npz"
    path.parent.mkdir(parents=True)
    train = np.zeros((6, 1), dtype=bool); train[:2] = True
    val = np.zeros((6, 1), dtype=bool); val[2:4] = True
    test = np.zeros((6, 1), dtype=bool); test[4:] = True
    np.savez_compressed(
        path,
        node_features=np.ones((6, 3), dtype=np.float32),
        node_labels=np.arange(6) % 2,
        edges=np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        train_masks=train, val_masks=val, test_masks=test,
    )
    data = load_real_dataset("new_benchmark", root)
    assert data.num_nodes == 6
