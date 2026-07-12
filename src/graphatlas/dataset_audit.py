from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from graphatlas.data import GraphData


def _tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _edge_set_hash(edge_index: torch.Tensor) -> str:
    edges = edge_index.detach().cpu().numpy().T
    canonical = np.unique(np.sort(edges, axis=1), axis=0)
    digest = hashlib.sha256(canonical.astype(np.int64, copy=False).tobytes())
    return digest.hexdigest()


def _single_label_homophily(data: GraphData) -> dict[str, float]:
    if data.is_multilabel or data.edge_index.numel() == 0:
        return {"edge_homophily": float("nan"), "node_homophily": float("nan"), "adjusted_homophily": float("nan")}
    src, dst = data.edge_index
    same = data.y[src].eq(data.y[dst]).float()
    edge_h = float(same.mean().cpu())
    degree = torch.bincount(dst, minlength=data.num_nodes).to(torch.float32)
    same_count = torch.zeros(data.num_nodes, dtype=torch.float32)
    same_count.index_add_(0, dst.cpu(), same.cpu())
    valid = degree > 0
    node_h = float((same_count[valid] / degree[valid]).mean()) if bool(valid.any()) else float("nan")
    class_degree = torch.zeros(data.num_classes, dtype=torch.float64)
    class_degree.index_add_(0, data.y[dst].cpu(), torch.ones_like(dst, dtype=torch.float64).cpu())
    expected = float((class_degree / class_degree.sum().clamp_min(1)).square().sum())
    adjusted = (edge_h - expected) / max(1.0 - expected, 1e-12)
    return {"edge_homophily": edge_h, "node_homophily": node_h, "adjusted_homophily": float(adjusted)}


def _connected_components(edge_index: torch.Tensor, n: int) -> int:
    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    for u, v in edge_index.detach().cpu().numpy().T:
        union(int(u), int(v))
    return len({find(i) for i in range(n)})


def audit_graph(data: GraphData, name: str | None = None) -> dict[str, Any]:
    data.validate()
    src, dst = data.edge_index
    degree = torch.bincount(dst.cpu(), minlength=data.num_nodes).to(torch.float32)
    x = data.x.detach().cpu()
    feature_sparsity = float(x.eq(0).float().mean()) if x.numel() else float("nan")
    duplicate_rows = 0
    if data.num_nodes <= 200_000:
        unique = torch.unique(x, dim=0)
        duplicate_rows = data.num_nodes - int(unique.shape[0])
    result: dict[str, Any] = {
        "dataset": name or (data.metadata or {}).get("name", "unknown"),
        "num_nodes": data.num_nodes,
        "num_edges_directed": int(data.edge_index.shape[1]),
        "num_features": data.num_features,
        "num_classes": data.num_classes,
        "multilabel": data.is_multilabel,
        "directed": bool((data.metadata or {}).get("directed", False)),
        "train_nodes": int(data.train_mask.sum()),
        "val_nodes": int(data.val_mask.sum()),
        "test_nodes": int(data.test_mask.sum()),
        "average_degree": float(degree.mean()),
        "median_degree": float(degree.median()),
        "max_degree": int(degree.max()),
        "isolated_nodes": int(degree.eq(0).sum()),
        "connected_components": _connected_components(data.edge_index, data.num_nodes) if data.num_nodes <= 200_000 else -1,
        "feature_sparsity": feature_sparsity,
        "duplicate_feature_rows": duplicate_rows,
        "feature_hash": _tensor_hash(data.x),
        "label_hash": _tensor_hash(data.y),
        "edge_set_hash": _edge_set_hash(data.edge_index),
        "primary_metric": (data.metadata or {}).get("metric", "roc_auc" if data.is_multilabel else "accuracy"),
        **_single_label_homophily(data),
    }
    if not data.is_multilabel:
        counts = torch.bincount(data.y.cpu(), minlength=data.num_classes).numpy()
        result["class_imbalance_ratio"] = float(counts.max() / max(counts[counts > 0].min(), 1))
        compatibility = np.zeros((data.num_classes, data.num_classes), dtype=np.int64)
        np.add.at(compatibility, (data.y[src].cpu().numpy(), data.y[dst].cpu().numpy()), 1)
        result["compatibility_matrix"] = compatibility.tolist()
    else:
        result["class_imbalance_ratio"] = float("nan")
        result["compatibility_matrix"] = None
    return result


def write_audit(records: list[dict[str, Any]], output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "dataset_audit.json"
    csv_path = output / "dataset_audit.csv"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    flat = [{k: v for k, v in row.items() if k != "compatibility_matrix"} for row in records]
    pd.DataFrame(flat).to_csv(csv_path, index=False)
    return csv_path, json_path
