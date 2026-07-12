from __future__ import annotations

from pathlib import Path

import numpy as np

from graphatlas.datasets.real import load_real_dataset, validate_dataset_file


def _write_fixture(root: Path, name: str, masks_split_first: bool) -> None:
    folder = root / name / "raw"
    folder.mkdir(parents=True)
    n = 12
    x = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
    y = np.asarray([0, 1] * 6, dtype=np.int64)
    edges = np.asarray([[i, (i + 1) % n] for i in range(n)], dtype=np.int64)
    train = np.zeros((3, n), dtype=bool)
    val = np.zeros((3, n), dtype=bool)
    test = np.zeros((3, n), dtype=bool)
    for split in range(3):
        train[split, split:split + 6] = True
        val[split, split + 6:split + 9] = True
        test[split] = ~(train[split] | val[split])
    if not masks_split_first:
        train, val, test = train.T, val.T, test.T
    np.savez(
        folder / f"{name}.npz",
        node_features=x,
        node_labels=y,
        edges=edges,
        train_masks=train,
        val_masks=val,
        test_masks=test,
    )


def test_hgb_npz_supports_split_first_masks(tmp_path):
    _write_fixture(tmp_path, "minesweeper", masks_split_first=True)
    info = validate_dataset_file(tmp_path / "minesweeper" / "raw" / "minesweeper.npz")
    data = load_real_dataset("minesweeper", tmp_path, split=2)
    assert info["num_splits"] == 3
    assert data.metadata["metric"] == "roc_auc"
    assert data.metadata["split"] == 2
    assert data.train_mask.shape == (12,)
    assert data.edge_index.shape[0] == 2


def test_hgb_npz_supports_node_first_masks(tmp_path):
    _write_fixture(tmp_path, "actor", masks_split_first=False)
    data = load_real_dataset("actor", tmp_path, split=4)
    assert data.metadata["metric"] == "accuracy"
    assert data.metadata["split"] == 1
    assert data.x.shape == (12, 4)


def test_geom_gcn_text_parsers(tmp_path):
    from graphatlas.datasets.real import _parse_geom_edge_file, _parse_geom_node_file

    node_file = tmp_path / "nodes.txt"
    node_file.write_text(
        "node_id\tfeature\tlabel\n0\t0,2\t1\n1\t1\t0\n2\t0,1,2\t1\n",
        encoding="utf-8",
    )
    edge_file = tmp_path / "edges.txt"
    edge_file.write_text("node_id\tneighbor_id\n0\t1\n1\t2\n", encoding="utf-8")
    features, labels = _parse_geom_node_file(node_file, "actor")
    edges = _parse_geom_edge_file(edge_file)
    assert features.shape == (3, 3)
    assert features[0].tolist() == [1.0, 0.0, 1.0]
    assert labels.tolist() == [1, 0, 1]
    assert edges.tolist() == [[0, 1], [1, 2]]
