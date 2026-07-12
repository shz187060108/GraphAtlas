from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy import sparse

from graphatlas.data import random_masks


KNOWN_ALIASES = {
    "cora": "cora",
    "citeseer": "citeseer",
    "cite_seer": "citeseer",
    "pubmed": "pubmed",
    "wikics": "wikics",
    "wiki_cs": "wikics",
    "dblp": "dblp",
    "coauthor_cs": "coauthor_cs",
    "coauthor_physics": "coauthor_physics",
    "cornell": "cornell",
    "texas": "texas",
    "wisconsin": "wisconsin",
    "ogbn_arxiv": "ogbn_arxiv",
    "ogbn_products": "ogbn_products",
    "ogbn_proteins": "ogbn_proteins",
    "ogbn_mag": "ogbn_mag",
    "mag_year": "mag_year",
    "oag_cs": "oag_cs",
    "oag_eng": "oag_eng",
    "oag_chem": "oag_chem",
    "rcdd": "rcdd",
    "ieee_cis_g": "ieee_cis_g",
    "h_pokec": "h_pokec",
    "pdns": "pdns",
    "hetgb_cornell": "hetgb_cornell",
    "hetgb_texas": "hetgb_texas",
    "hetgb_wisconsin": "hetgb_wisconsin",
    "hetgb_actor": "hetgb_actor",
    "hetgb_amazon": "hetgb_amazon",
    "penn94": "penn94",
    "genius": "genius",
    "twitch_gamer": "twitch_gamer",
    "pokec": "pokec",
    "wiki": "wiki",
    "arxiv_year": "arxiv_year",
    "snap_patents": "snap_patents",
    "cite_catalysis": "cite_catalysis",
}

HETGB_BASE_NAMES = {"cornell", "texas", "wisconsin", "actor", "amazon"}


@dataclass
class LocalDatasetCandidate:
    dataset: str
    path: str
    format: str
    confidence: str
    size_bytes: int
    fingerprint: str
    notes: str = ""


def normalize_name(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return KNOWN_ALIASES.get(value, value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(path: Path, limit: int = 64) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())[:limit]
    for file in files:
        relative = str(file.relative_to(path)).replace("\\", "/").encode()
        digest.update(relative)
        digest.update(str(file.stat().st_size).encode())
        if file.stat().st_size <= 64 * 1024 * 1024:
            digest.update(bytes.fromhex(sha256_file(file)))
    return digest.hexdigest()


def _candidate_name(path: Path) -> str | None:
    parts = [normalize_name(part) for part in path.parts]
    # HeTGB reconstructs classic datasets with raw text and slightly different
    # graph statistics. Keep them separate from the non-text benchmark copies.
    if any("hetgb" in part for part in parts):
        for base in HETGB_BASE_NAMES:
            if base in parts:
                return f"hetgb_{base}"
    for alias in KNOWN_ALIASES:
        normalized = normalize_name(alias)
        if normalized in parts:
            return KNOWN_ALIASES[alias]
    return None


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def discover_local_datasets(search_roots: Iterable[str | Path]) -> list[LocalDatasetCandidate]:
    candidates: dict[tuple[str, str], LocalDatasetCandidate] = {}
    for root_value in search_roots:
        root = Path(root_value).expanduser()
        if not root.exists():
            continue
        for file in root.rglob("*"):
            if not file.is_file():
                continue
            dataset = _candidate_name(file)
            if dataset is None:
                continue
            fmt: str | None = None
            base = file.parent
            notes = ""
            if file.name == "data.pt" and file.parent.name == "processed":
                fmt, base = "pyg_processed", file.parent.parent
            elif file.suffix == ".npz":
                fmt = "graphatlas_npz" if file.name == f"{dataset}.npz" else "npz"
                base = file
            elif file.name.startswith("ind.") and file.name.endswith((".x", ".tx", ".allx", ".graph")):
                fmt, base = "planetoid_raw", file.parent
            elif file.name == "out1_node_feature_label.txt":
                fmt, base = "geom_gcn_raw", file.parent
            elif dataset.startswith("ogbn_") and file.name in {"RELEASE_v1.txt", "master.csv", "meta_info.py"}:
                fmt, base = "ogb_cache", next((parent for parent in file.parents if normalize_name(parent.name) == dataset), file.parent)
                notes = "Requires the optional ogb and torch-geometric packages for conversion."
            if fmt is None:
                continue
            key = (dataset, str(base.resolve()))
            if key in candidates:
                continue
            fingerprint = sha256_file(base) if base.is_file() else directory_fingerprint(base)
            candidates[key] = LocalDatasetCandidate(
                dataset=dataset,
                path=str(base.resolve()),
                format=fmt,
                confidence="high" if fmt in {"pyg_processed", "planetoid_raw", "geom_gcn_raw", "ogb_cache"} else "medium",
                size_bytes=base.stat().st_size if base.is_file() else _directory_size(base),
                fingerprint=fingerprint,
                notes=notes,
            )
    return sorted(candidates.values(), key=lambda row: (row.dataset, row.format, row.path))


def write_discovery_manifest(candidates: list[LocalDatasetCandidate], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps([asdict(item) for item in candidates], indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


def _mask_from_index(index: torch.Tensor | np.ndarray, num_nodes: int) -> np.ndarray:
    mask = np.zeros(num_nodes, dtype=bool)
    mask[np.asarray(index, dtype=np.int64).reshape(-1)] = True
    return mask


def _orient_mask_array(value: Any, num_nodes: int) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.ndim == 1:
        return array[:, None]
    if array.shape[0] == num_nodes:
        return array
    if array.shape[1] == num_nodes:
        return array.T
    raise ValueError(f"Cannot orient split masks with shape {array.shape} for {num_nodes} nodes")


def _extract_pyg_data(path: Path) -> dict[str, np.ndarray]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Loading PyG processed/data.pt requires torch-geometric because the file contains PyG classes. "
            "Install with `pip install -e .[graph]`."
        ) from error
    data = payload[0] if isinstance(payload, tuple) else payload
    if isinstance(data, (list, tuple)) and data:
        data = data[0]
    if isinstance(data, dict):
        try:
            from torch_geometric.data import Data, HeteroData
            if any(isinstance(key, tuple) for key in data):
                data = HeteroData.from_dict(data)
            else:
                data = Data.from_dict(data)
        except ImportError as error:
            raise RuntimeError("Deserializing this PyG data.pt requires torch-geometric.") from error
    if hasattr(data, "node_types") and hasattr(data, "edge_types"):
        return _project_heterodata(data)
    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    y = getattr(data, "y", None)
    if x is None or edge_index is None or y is None:
        raise ValueError(f"{path} does not expose x, edge_index, and y")
    num_nodes = int(x.shape[0])
    split_arrays: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        mask = getattr(data, f"{name}_mask", None)
        if mask is not None:
            split_arrays[name] = _orient_mask_array(mask.cpu().numpy(), num_nodes)
    if len(split_arrays) != 3:
        generated = random_masks(num_nodes, seed=0)
        split_arrays = {name: mask.numpy()[:, None] for name, mask in zip(("train", "val", "test"), generated, strict=True)}
    result = {
        "node_features": x.cpu().numpy(),
        "node_labels": y.cpu().numpy(),
        "edges": edge_index.cpu().numpy().T,
        "train_masks": split_arrays["train"],
        "val_masks": split_arrays["val"],
        "test_masks": split_arrays["test"],
    }
    for attr in ("edge_attr", "node_type", "edge_type"):
        value = getattr(data, attr, None)
        if value is not None:
            result[attr] = value.cpu().numpy()
    return result


def _project_heterodata(data: Any) -> dict[str, np.ndarray]:
    """Project a PyG HeteroData object to one typed homogeneous graph.

    The projection preserves node_type and edge_type arrays. Only the label-bearing
    target node type participates in train/validation/test masks; all other node
    types remain available as context nodes. Feature matrices with different
    dimensions are zero-padded to a common width.
    """
    node_types = list(data.node_types)
    target_candidates = []
    max_dim = 1
    counts: dict[str, int] = {}
    for node_type in node_types:
        store = data[node_type]
        count = int(getattr(store, "num_nodes", 0) or (store.x.shape[0] if getattr(store, "x", None) is not None else 0))
        counts[node_type] = count
        x = getattr(store, "x", None)
        if x is not None:
            max_dim = max(max_dim, int(x.shape[-1]))
        if getattr(store, "y", None) is not None:
            target_candidates.append(node_type)
    if not target_candidates:
        raise ValueError("HeteroData has no label-bearing node type")
    target = next(
        (node_type for node_type in target_candidates if any(getattr(data[node_type], f"{name}_mask", None) is not None for name in ("train", "val", "test"))),
        target_candidates[0],
    )
    offsets: dict[str, int] = {}
    cursor = 0
    feature_blocks = []
    node_type_blocks = []
    for type_id, node_type in enumerate(node_types):
        offsets[node_type] = cursor
        count = counts[node_type]
        cursor += count
        store = data[node_type]
        x = getattr(store, "x", None)
        block = torch.zeros(count, max_dim, dtype=torch.float32)
        if x is not None:
            x = x.to(torch.float32)
            block[:, : x.shape[1]] = x
        feature_blocks.append(block)
        node_type_blocks.append(torch.full((count,), type_id, dtype=torch.long))
    num_nodes = cursor
    target_store = data[target]
    raw_y = target_store.y.cpu()
    if raw_y.ndim == 2 and raw_y.shape[1] == 1:
        raw_y = raw_y.reshape(-1)
    if raw_y.ndim == 1:
        labels = torch.zeros(num_nodes, dtype=torch.long)
    else:
        labels = torch.zeros(num_nodes, raw_y.shape[1], dtype=torch.float32)
    target_slice = slice(offsets[target], offsets[target] + counts[target])
    labels[target_slice] = raw_y.to(labels.dtype)

    split_arrays: dict[str, np.ndarray] = {}
    for split_name in ("train", "val", "test"):
        mask = getattr(target_store, f"{split_name}_mask", None)
        index = getattr(target_store, f"{split_name}_idx", None)
        global_mask = np.zeros(num_nodes, dtype=bool)
        if mask is not None:
            local_mask = np.asarray(mask.cpu(), dtype=bool).reshape(-1)
            global_mask[offsets[target] : offsets[target] + counts[target]] = local_mask
        elif index is not None:
            local_index = np.asarray(index.cpu(), dtype=np.int64).reshape(-1)
            global_mask[offsets[target] + local_index] = True
        split_arrays[split_name] = global_mask[:, None]
    if any(array.sum() == 0 for array in split_arrays.values()):
        generated = random_masks(counts[target], seed=0)
        for split_name, local in zip(("train", "val", "test"), generated, strict=True):
            global_mask = np.zeros(num_nodes, dtype=bool)
            global_mask[offsets[target] : offsets[target] + counts[target]] = local.numpy()
            split_arrays[split_name] = global_mask[:, None]

    edge_blocks = []
    edge_type_blocks = []
    for type_id, edge_type in enumerate(data.edge_types):
        source_type, _, target_type = edge_type
        edge_index = data[edge_type].edge_index.cpu().clone()
        edge_index[0] += offsets[source_type]
        edge_index[1] += offsets[target_type]
        edge_blocks.append(edge_index)
        edge_type_blocks.append(torch.full((edge_index.shape[1],), type_id, dtype=torch.long))
    if not edge_blocks:
        raise ValueError("HeteroData has no edges")
    return {
        "node_features": torch.cat(feature_blocks, dim=0).numpy(),
        "node_labels": labels.numpy(),
        "edges": torch.cat(edge_blocks, dim=1).T.numpy(),
        "train_masks": split_arrays["train"],
        "val_masks": split_arrays["val"],
        "test_masks": split_arrays["test"],
        "node_type": torch.cat(node_type_blocks).numpy(),
        "edge_type": torch.cat(edge_type_blocks).numpy(),
        "target_node_type": np.asarray([node_types.index(target)], dtype=np.int64),
    }


def _parse_planetoid(directory: Path, dataset: str) -> dict[str, np.ndarray]:
    prefix = dataset.lower()
    objects: list[Any] = []
    for suffix in ("x", "tx", "allx", "y", "ty", "ally", "graph"):
        path = directory / f"ind.{prefix}.{suffix}"
        if not path.exists():
            matches = list(directory.glob(f"ind.*.{suffix}"))
            if len(matches) != 1:
                raise FileNotFoundError(path)
            path = matches[0]
        with path.open("rb") as handle:
            objects.append(pickle.load(handle, encoding="latin1"))
    x, tx, allx, y, ty, ally, graph = objects
    test_path = next(iter(directory.glob("ind.*.test.index")), None)
    if test_path is None:
        raise FileNotFoundError("Planetoid test index file")
    test_index = np.asarray([int(line.strip()) for line in test_path.read_text().splitlines() if line.strip()])
    sorted_test = np.sort(test_index)
    if dataset == "citeseer":
        full_range = np.arange(sorted_test.min(), sorted_test.max() + 1)
        tx_extended = sparse.lil_matrix((len(full_range), x.shape[1]))
        tx_extended[sorted_test - full_range.min(), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(full_range), y.shape[1]))
        ty_extended[sorted_test - full_range.min(), :] = ty
        ty = ty_extended
    features = sparse.vstack((allx, tx)).tolil()
    features[test_index, :] = features[sorted_test, :]
    labels = np.vstack((ally, ty))
    labels[test_index, :] = labels[sorted_test, :]
    labels = labels.argmax(axis=1).astype(np.int64)
    edges = np.asarray([(int(source), int(target)) for source, targets in graph.items() for target in targets], dtype=np.int64)
    train = np.arange(len(y))
    val = np.arange(len(y), min(len(y) + 500, labels.shape[0]))
    test = test_index
    return {
        "node_features": np.asarray(features.toarray(), dtype=np.float32),
        "node_labels": labels,
        "edges": edges,
        "train_masks": _mask_from_index(train, labels.shape[0])[:, None],
        "val_masks": _mask_from_index(val, labels.shape[0])[:, None],
        "test_masks": _mask_from_index(test, labels.shape[0])[:, None],
    }


def _load_npz_flexible(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as raw:
        keys = set(raw.files)
        if {"node_features", "node_labels", "edges"}.issubset(keys):
            return {key: np.asarray(raw[key]) for key in raw.files}
        aliases = {
            "node_features": ["x", "features", "feat"],
            "node_labels": ["y", "labels", "label"],
            "edges": ["edge_index", "edges", "edge_list"],
            "train_masks": ["train_mask", "train_masks"],
            "val_masks": ["val_mask", "valid_mask", "val_masks"],
            "test_masks": ["test_mask", "test_masks"],
        }
        result: dict[str, np.ndarray] = {}
        for target, options in aliases.items():
            source = next((option for option in options if option in keys), None)
            if source is not None:
                result[target] = np.asarray(raw[source])
        for optional in (
            "text_features",
            "structural_features",
            "edge_attr",
            "node_type",
            "edge_type",
            "texts",
            "raw_text",
            "text",
        ):
            if optional in keys:
                result[optional] = np.asarray(raw[optional])
    if not {"node_features", "node_labels", "edges"}.issubset(result):
        raise ValueError(f"Could not map NPZ keys in {path}")
    n = result["node_features"].shape[0]
    if not {"train_masks", "val_masks", "test_masks"}.issubset(result):
        masks = random_masks(n, 0)
        for name, mask in zip(("train_masks", "val_masks", "test_masks"), masks, strict=True):
            result[name] = mask.numpy()[:, None]
    return result


def convert_candidate(candidate: LocalDatasetCandidate | dict[str, Any], destination_root: str | Path) -> Path:
    if isinstance(candidate, dict):
        candidate = LocalDatasetCandidate(**candidate)
    source = Path(candidate.path)
    if candidate.format == "pyg_processed":
        payload = _extract_pyg_data(source / "processed" / "data.pt" if source.is_dir() else source)
    elif candidate.format == "planetoid_raw":
        payload = _parse_planetoid(source, candidate.dataset)
    elif candidate.format in {"npz", "graphatlas_npz"}:
        payload = _load_npz_flexible(source)
    elif candidate.format == "geom_gcn_raw":
        from graphatlas.datasets.real import _parse_geom_edge_file, _parse_geom_node_file
        payload = {
            "node_features": _parse_geom_node_file(source / "out1_node_feature_label.txt", candidate.dataset)[0],
            "node_labels": _parse_geom_node_file(source / "out1_node_feature_label.txt", candidate.dataset)[1],
            "edges": _parse_geom_edge_file(source / "out1_graph_edges.txt"),
        }
        n = payload["node_features"].shape[0]
        masks = random_masks(n, 0)
        for name, mask in zip(("train_masks", "val_masks", "test_masks"), masks, strict=True):
            payload[name] = mask.numpy()[:, None]
    elif candidate.format == "ogb_cache":
        payload = _convert_ogb(candidate.dataset, source)
    else:
        raise ValueError(f"Unsupported local format: {candidate.format}")
    edges = np.asarray(payload["edges"])
    if edges.ndim == 2 and edges.shape[0] == 2 and edges.shape[1] != 2:
        payload["edges"] = edges.T
    destination = Path(destination_root) / candidate.dataset / "raw" / f"{candidate.dataset}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    manifest = {
        "dataset": candidate.dataset,
        "source_path": candidate.path,
        "source_format": candidate.format,
        "source_fingerprint": candidate.fingerprint,
        "converted_sha256": sha256_file(destination),
    }
    (destination.parent / "local_import_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def _convert_ogb(dataset: str, source: Path) -> dict[str, np.ndarray]:
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as error:
        raise RuntimeError("OGB conversion requires `pip install -e .[graph,ogb]`.") from error
    root = source.parent if normalize_name(source.name) == dataset else source
    pyg = PygNodePropPredDataset(name=dataset.replace("_", "-"), root=str(root))
    data = pyg[0]
    split = pyg.get_idx_split()
    x = data.x
    edge_attr = getattr(data, "edge_attr", None)
    if x is None and edge_attr is not None:
        x = torch.zeros(data.num_nodes, edge_attr.shape[-1], dtype=edge_attr.dtype)
        x.index_add_(0, data.edge_index[1], edge_attr)
        degree = torch.bincount(data.edge_index[1], minlength=data.num_nodes).clamp_min(1).to(x.dtype)
        x = x / degree[:, None]
    if x is None:
        raise ValueError(f"{dataset} contains no node features and no usable edge features")
    result = {
        "node_features": x.cpu().numpy(),
        "node_labels": data.y.cpu().numpy(),
        "edges": data.edge_index.cpu().numpy().T,
        "train_masks": _mask_from_index(split["train"].cpu().numpy(), data.num_nodes)[:, None],
        "val_masks": _mask_from_index(split["valid"].cpu().numpy(), data.num_nodes)[:, None],
        "test_masks": _mask_from_index(split["test"].cpu().numpy(), data.num_nodes)[:, None],
    }
    if edge_attr is not None:
        result["edge_attr"] = edge_attr.cpu().numpy()
    return result
