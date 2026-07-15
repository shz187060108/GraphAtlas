#!/usr/bin/env python
"""Run inside the isolated H2GB interpreter and export native typed tensors."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import sys
import types

import numpy as np
import torch


def _mask(data, entity: str, name: str) -> torch.Tensor:
    value = getattr(data[entity], name, None)
    if value is None:
        raise RuntimeError(f"Official H2GB loader did not provide {entity}.{name}")
    return value.view(-1).bool().cpu()


def _load(dataset: str, root: Path):
    # H2GB's top-level __init__ imports all training backends.  The bridge only
    # needs the official dataset package, so expose that package without
    # importing unrelated optional model dependencies.
    distribution = importlib.metadata.distribution("H2GB")
    package_root = Path(distribution.locate_file("H2GB"))
    package = types.ModuleType("H2GB")
    package.__path__ = [str(package_root)]
    datasets_package = types.ModuleType("H2GB.datasets")
    datasets_package.__path__ = [str(package_root / "datasets")]
    sys.modules["H2GB"] = package
    sys.modules["H2GB.datasets"] = datasets_package
    torch.manual_seed(42)
    np.random.seed(42)
    if dataset == "h2gb_pdns":
        pdns_module = importlib.import_module("H2GB.datasets.pdns_dataset")
        PDNSDataset = pdns_module.PDNSDataset
        # The upstream timestamp sorter splits paths on '/'. Keep the official
        # loader intact while normalizing its generated path strings on Windows.
        original_join = pdns_module.osp.join
        pdns_module.osp.join = lambda *parts: original_join(*parts).replace("\\", "/")
        try:
            loaded = PDNSDataset(root=str(root).replace("\\", "/"), start=0, end=60, domain_file="domains2.csv")
        finally:
            pdns_module.osp.join = original_join
        return loaded.data, "domain_node", "f1"
    if dataset == "h2gb_mag_year":
        MAGDataset = importlib.import_module("H2GB.datasets.mag_dataset").MAGDataset

        loaded = MAGDataset(root=str(root), name="mag-year", rand_split=False)
        return loaded[0], "paper", "accuracy"
    if dataset == "h2gb_ieee_cis":
        IeeeCisDataset = importlib.import_module("H2GB.datasets.ieee_cis_dataset").IeeeCisDataset

        loaded = IeeeCisDataset(root=str(root))
        data = loaded.data
        count = int(data["transaction"].num_nodes)
        train_end, val_end = int(0.8 * count), int(0.9 * count)
        for name, start, end in (("train_mask", 0, train_end), ("val_mask", train_end, val_end), ("test_mask", val_end, count)):
            mask = torch.zeros(count, dtype=torch.bool)
            mask[start:end] = True
            setattr(data["transaction"], name, mask)
        return data, "transaction", "f1"
    raise ValueError(f"Unsupported H2GB dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data, task_entity, metric = _load(args.dataset, args.root)
    node_types = sorted(str(value) for value in data.node_types)
    edge_types = sorted(tuple(str(part) for part in value) for value in data.edge_types)
    num_nodes_dict = {node_type: int(data[node_type].num_nodes) for node_type in node_types}
    x_dict = {
        node_type: (None if getattr(data[node_type], "x", None) is None else data[node_type].x.detach().cpu())
        for node_type in node_types
    }
    edge_index_dict = {
        edge_type: data[edge_type].edge_index.detach().cpu().long()
        for edge_type in edge_types
    }
    target_y = data[task_entity].y.detach().cpu()
    if target_y.ndim > 1 and target_y.shape[-1] == 1:
        target_y = target_y.squeeze(-1)
    try:
        package_version = importlib.metadata.version("H2GB")
    except importlib.metadata.PackageNotFoundError:
        package_version = "source-checkout"
    payload = {
        "schema_version": 1,
        "dataset_name": args.dataset,
        "benchmark_family": "H2GB",
        "source_repository": "junhongmit/H2GB",
        "source_revision": os.environ.get("H2GB_SOURCE_REVISION", package_version),
        "node_types": node_types,
        "edge_types": edge_types,
        "task_entity": task_entity,
        "x_dict": x_dict,
        "num_nodes_dict": num_nodes_dict,
        "edge_index_dict": edge_index_dict,
        "target_y": target_y,
        "train_mask": _mask(data, task_entity, "train_mask"),
        "val_mask": _mask(data, task_entity, "val_mask"),
        "test_mask": _mask(data, task_entity, "test_mask"),
        "primary_metric": metric,
        "global_node_id_dict": {
            node_type: torch.arange(count, dtype=torch.long) for node_type, count in num_nodes_dict.items()
        },
        "metadata": {
            "native_heterogeneous": True,
            "homogeneous_projection": False,
            "split_source": "official_h2gb_loader",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)


if __name__ == "__main__":
    main()
