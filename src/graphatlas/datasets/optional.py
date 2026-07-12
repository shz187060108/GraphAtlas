from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from graphatlas.data import random_masks
from graphatlas.datasets.local import sha256_file

PYG_DATASETS = {
    "cora": ("Planetoid", {"name": "Cora"}),
    "citeseer": ("Planetoid", {"name": "CiteSeer"}),
    "pubmed": ("Planetoid", {"name": "PubMed"}),
    "wikics": ("WikiCS", {}),
    "cornell": ("WebKB", {"name": "Cornell"}),
    "texas": ("WebKB", {"name": "Texas"}),
    "wisconsin": ("WebKB", {"name": "Wisconsin"}),
    "coauthor_cs": ("Coauthor", {"name": "CS"}),
    "coauthor_physics": ("Coauthor", {"name": "Physics"}),
    "dblp": ("CitationFull", {"name": "DBLP"}),
}
OGB_DATASETS = {"ogbn_arxiv", "ogbn_products", "ogbn_proteins"}


def _mask(value: torch.Tensor | None, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if value is not None:
        raise ValueError("internal helper expects explicit train/val/test masks")
    masks = random_masks(n, seed)
    return tuple(mask.cpu().numpy()[:, None] for mask in masks)  # type: ignore[return-value]


def _standardize_split_masks(
    train: torch.Tensor, val: torch.Tensor, test: torch.Tensor, num_nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orient split masks and broadcast shared validation/test masks.

    WikiCS supplies 20 training splits but a single shared validation and test
    mask. GraphAtlas stores all splits column-wise, so shared masks must be
    repeated to match the number of training splits.
    """
    def orient(mask: torch.Tensor) -> np.ndarray:
        array = mask.cpu().numpy().astype(bool)
        if array.ndim == 1:
            if array.shape[0] != num_nodes:
                raise ValueError(f"Mask length {array.shape[0]} does not match {num_nodes} nodes")
            return array[:, None]
        if array.ndim != 2:
            raise ValueError(f"Masks must be rank 1 or 2, got {array.shape}")
        if array.shape[0] == num_nodes:
            return array
        if array.shape[1] == num_nodes:
            return array.T
        raise ValueError(f"Neither mask axis matches {num_nodes} nodes: {array.shape}")

    masks = [orient(mask) for mask in (train, val, test)]
    num_splits = max(mask.shape[1] for mask in masks)
    standardized = []
    for mask in masks:
        if mask.shape[1] == num_splits:
            standardized.append(mask)
        elif mask.shape[1] == 1:
            standardized.append(np.repeat(mask, num_splits, axis=1))
        else:
            raise ValueError(f"Incompatible split counts: {[item.shape[1] for item in masks]}")
    return tuple(standardized)  # type: ignore[return-value]


def download_pyg_dataset(name: str, root: str | Path) -> Path:
    try:
        import torch_geometric.datasets as datasets
    except ImportError as error:
        raise RuntimeError("This dataset requires the optional graph dependencies: `pip install -e .[graph]`.") from error
    class_name, kwargs = PYG_DATASETS[name]
    dataset_root = Path(root) / "_pyg_cache" / name
    dataset_class = getattr(datasets, class_name)
    dataset = dataset_class(root=str(dataset_root), **kwargs)
    data = dataset[0]
    n = int(data.num_nodes)
    train = getattr(data, "train_mask", None)
    val = getattr(data, "val_mask", None)
    test = getattr(data, "test_mask", None)
    if train is None or val is None or test is None:
        train_np, val_np, test_np = _mask(None, n)
    else:
        train_np, val_np, test_np = _standardize_split_masks(train, val, test, n)
    path = Path(root) / name / "raw" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "node_features": data.x.cpu().numpy().astype(np.float32),
        "node_labels": data.y.cpu().numpy(),
        "edges": data.edge_index.cpu().numpy().T,
        "train_masks": train_np,
        "val_masks": val_np,
        "test_masks": test_np,
    }
    if getattr(data, "edge_attr", None) is not None:
        payload["edge_attr"] = data.edge_attr.cpu().numpy()
    np.savez_compressed(path, **payload)
    (path.parent / "download_manifest.json").write_text(json.dumps({
        "dataset": name,
        "source": f"PyTorch Geometric {class_name}",
        "sha256": sha256_file(path),
        "num_nodes": n,
        "num_edges": int(data.edge_index.shape[1]),
    }, indent=2), encoding="utf-8")
    return path


def download_ogb_dataset(name: str, root: str | Path) -> Path:
    from graphatlas.datasets.local import LocalDatasetCandidate, _convert_ogb, directory_fingerprint
    cache = Path(root) / "_ogb_cache" / name
    payload = _convert_ogb(name, cache)
    path = Path(root) / name / "raw" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    (path.parent / "download_manifest.json").write_text(json.dumps({
        "dataset": name,
        "source": "Open Graph Benchmark",
        "cache": str(cache),
        "cache_fingerprint": directory_fingerprint(cache) if cache.exists() else None,
        "sha256": sha256_file(path),
    }, indent=2), encoding="utf-8")
    return path


def download_optional_dataset(name: str, root: str | Path) -> Path:
    if name in PYG_DATASETS:
        return download_pyg_dataset(name, root)
    if name in OGB_DATASETS:
        return download_ogb_dataset(name, root)
    raise ValueError(name)
