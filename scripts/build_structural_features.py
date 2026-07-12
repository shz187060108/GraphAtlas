#!/usr/bin/env python
from __future__ import annotations

try:
    from _bootstrap import PROJECT_ROOT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT_ROOT

import argparse
from pathlib import Path

import numpy as np


def structural_features(num_nodes: int, edges: np.ndarray, iterations: int = 20) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64)
    if edges.shape[0] == 2 and edges.shape[1] != 2:
        edges = edges.T
    src, dst = edges[:, 0], edges[:, 1]
    out_degree = np.bincount(src, minlength=num_nodes).astype(np.float64)
    in_degree = np.bincount(dst, minlength=num_nodes).astype(np.float64)
    total = in_degree + out_degree
    neighbor_degree_sum = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(neighbor_degree_sum, dst, total[src])
    mean_neighbor_degree = neighbor_degree_sum / np.maximum(in_degree, 1.0)
    pagerank = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)
    damping = 0.85
    for _ in range(iterations):
        message = pagerank[src] / np.maximum(out_degree[src], 1.0)
        updated = np.full(num_nodes, (1.0 - damping) / num_nodes, dtype=np.float64)
        np.add.at(updated, dst, damping * message)
        dangling = pagerank[out_degree == 0].sum()
        updated += damping * dangling / num_nodes
        pagerank = updated
    raw = np.stack(
        [
            in_degree,
            out_degree,
            total,
            np.log1p(in_degree),
            np.log1p(out_degree),
            np.log1p(total),
            np.sqrt(total),
            mean_neighbor_degree,
            np.log1p(mean_neighbor_degree),
            pagerank,
            1.0 / np.maximum(total, 1.0),
        ],
        axis=1,
    )
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True)
    return ((raw - mean) / np.maximum(std, 1e-8)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add label-free structural features to standardized NPZ datasets.")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--root", default="data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in args.dataset:
        normalized = name.lower().replace("-", "_")
        path = PROJECT_ROOT / args.root / normalized / "raw" / f"{normalized}.npz"
        with np.load(path, allow_pickle=True) as raw:
            payload = {key: raw[key] for key in raw.files}
        if "structural_features" in payload and not args.overwrite:
            print(f"skip {normalized}: structural_features already exist")
            continue
        payload["structural_features"] = structural_features(payload["node_features"].shape[0], payload["edges"])
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(path)
        print(f"updated {path} with {payload['structural_features'].shape[1]} structural features")

if __name__ == "__main__":
    main()
