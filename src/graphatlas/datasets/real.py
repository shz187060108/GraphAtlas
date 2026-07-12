from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from graphatlas.data import GraphData, coalesce_undirected


SUPPORTED_NAMES = {
    "roman_empire",
    "amazon_ratings",
    "minesweeper",
    "tolokers",
    "questions",
    "actor",
    "chameleon",
    "squirrel",
    "chameleon_filtered",
    "squirrel_filtered",
}
ROC_AUC_NAMES = {"minesweeper", "tolokers", "questions"}
HGB_NAMES = {
    "roman_empire",
    "amazon_ratings",
    "minesweeper",
    "tolokers",
    "questions",
    "chameleon_filtered",
    "squirrel_filtered",
}
GEOM_GCN_NAMES = {"actor", "chameleon", "squirrel"}
OPTIONAL_NAMES = {"cora", "citeseer", "pubmed", "wikics", "dblp", "coauthor_cs", "coauthor_physics", "cornell", "texas", "wisconsin", "ogbn_arxiv", "ogbn_products", "ogbn_proteins"}
SUPPORTED_NAMES |= OPTIONAL_NAMES
HGB_BASE = "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data"
GEOM_GCN_BASE = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master"
GEOM_GCN_FOLDER = {"actor": "film", "chameleon": "chameleon", "squirrel": "squirrel"}
REQUIRED_KEYS = {"node_features", "node_labels", "edges", "train_masks", "val_masks", "test_masks"}


def _normalize_name(name: str) -> str:
    return name.lower().replace("-", "_").strip()


def _download(url: str, destination: Path, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "GraphAtlas/0.3"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, open(temporary, "wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if temporary.stat().st_size == 0:
                raise OSError(f"Downloaded empty file from {url}")
            temporary.replace(destination)
            return
        except Exception as error:  # noqa: BLE001 - retain source network details
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_error


def _dataset_path(name: str, root: str | Path) -> Path:
    normalized = _normalize_name(name)
    return Path(root) / normalized / "raw" / f"{normalized}.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _dense_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object and array.size == 1:
        item = array.item()
        if sparse.issparse(item):
            return item.toarray()
        array = np.asarray(item)
    if sparse.issparse(array):
        return array.toarray()
    return array


def _orient_masks(masks: np.ndarray, num_nodes: int) -> np.ndarray:
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim == 1:
        if masks.shape[0] != num_nodes:
            raise ValueError(f"Mask length {masks.shape[0]} does not match {num_nodes} nodes")
        return masks[:, None]
    if masks.ndim != 2:
        raise ValueError(f"Masks must be rank 1 or 2, got {masks.shape}")
    if masks.shape[0] == num_nodes:
        return masks
    if masks.shape[1] == num_nodes:
        return masks.T
    raise ValueError(f"Neither mask axis matches {num_nodes} nodes: {masks.shape}")


def validate_dataset_file(path: str | Path) -> dict[str, int | str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as raw:
        missing = REQUIRED_KEYS.difference(raw.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        features = _dense_array(raw["node_features"])
        labels = np.asarray(raw["node_labels"])
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.reshape(-1)
        edges = np.asarray(raw["edges"])
        if features.ndim != 2:
            raise ValueError(f"node_features must be rank 2, got {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError("node_features contains NaN or infinite values")
        if labels.shape[0] != features.shape[0]:
            raise ValueError("node_labels and node_features have inconsistent node counts")
        if labels.size == 0:
            raise ValueError("node_labels must be non-empty")
        if labels.ndim == 1 and labels.min() < 0:
            raise ValueError("single-label node_labels must be non-negative integers")
        if labels.ndim not in {1, 2}:
            raise ValueError(f"node_labels must be rank 1 or 2, got {labels.shape}")
        if edges.ndim != 2 or 2 not in edges.shape:
            raise ValueError(f"edges must be [E,2] or [2,E], got {edges.shape}")
        edges_by_two = edges if edges.shape[1] == 2 else edges.T
        if edges_by_two.size and (edges_by_two.min() < 0 or edges_by_two.max() >= features.shape[0]):
            raise ValueError("edges contain out-of-range node IDs")
        train = _orient_masks(np.asarray(raw["train_masks"]), features.shape[0])
        val = _orient_masks(np.asarray(raw["val_masks"]), features.shape[0])
        test = _orient_masks(np.asarray(raw["test_masks"]), features.shape[0])
        if train.shape != val.shape or train.shape != test.shape:
            raise ValueError("train, validation, and test masks have inconsistent shapes")
        if np.any(train.astype(np.int8) + val.astype(np.int8) + test.astype(np.int8) > 1):
            raise ValueError("train, validation, and test masks overlap")
        if np.any(train.sum(axis=0) == 0) or np.any(val.sum(axis=0) == 0) or np.any(test.sum(axis=0) == 0):
            raise ValueError("every split must contain train, validation, and test nodes")
        return {
            "num_nodes": int(features.shape[0]),
            "num_features": int(features.shape[1]),
            "num_classes": int(labels.shape[1] if labels.ndim == 2 else np.unique(labels).size),
            "multilabel": bool(labels.ndim == 2),
            "num_edges_raw": int(edges_by_two.shape[0]),
            "num_splits": int(train.shape[1]),
            "sha256": _sha256(path),
        }


def _parse_geom_node_file(path: Path, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[int, list[float], int]] = []
    with path.open("r", encoding="utf-8") as handle:
        header = next(handle, None)
        if header is None:
            raise ValueError(f"Empty Geom-GCN node file: {path}")
        for line_number, line in enumerate(handle, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Malformed node row at {path}:{line_number}")
            node_id = int(parts[0])
            values = [float(value) for value in parts[1].split(",") if value != ""]
            label = int(parts[2])
            rows.append((node_id, values, label))
    if not rows:
        raise ValueError(f"No nodes parsed from {path}")
    rows.sort(key=lambda row: row[0])
    node_ids = [row[0] for row in rows]
    if node_ids != list(range(len(rows))):
        raise ValueError(f"Geom-GCN node IDs must be contiguous from zero in {path}")
    labels = np.asarray([row[2] for row in rows], dtype=np.int64)
    feature_rows = [row[1] for row in rows]
    lengths = {len(row) for row in feature_rows}
    if dataset_name == "actor" or len(lengths) != 1:
        indices = [[int(value) for value in row] for row in feature_rows]
        feature_dim = max((max(row) if row else -1) for row in indices) + 1
        features = np.zeros((len(rows), feature_dim), dtype=np.float32)
        for node_id, active in enumerate(indices):
            if active:
                features[node_id, np.asarray(active, dtype=np.int64)] = 1.0
    else:
        features = np.asarray(feature_rows, dtype=np.float32)
    return features, labels


def _parse_geom_edge_file(path: Path) -> np.ndarray:
    edges: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        header = next(handle, None)
        if header is None:
            raise ValueError(f"Empty Geom-GCN edge file: {path}")
        for line_number, line in enumerate(handle, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Malformed edge row at {path}:{line_number}")
            edges.append((int(parts[0]), int(parts[1])))
    return np.asarray(edges, dtype=np.int64)


def _download_hgb_dataset(name: str, root: Path) -> tuple[Path, list[str]]:
    path = _dataset_path(name, root)
    url = f"{HGB_BASE}/{name}.npz"
    _download(url, path)
    return path, [url]


def _download_geom_gcn_dataset(name: str, root: Path) -> tuple[Path, list[str]]:
    folder = GEOM_GCN_FOLDER[name]
    dataset_root = root / name / "raw"
    source_root = dataset_root / "source"
    node_url = f"{GEOM_GCN_BASE}/new_data/{folder}/out1_node_feature_label.txt"
    edge_url = f"{GEOM_GCN_BASE}/new_data/{folder}/out1_graph_edges.txt"
    node_path = source_root / "out1_node_feature_label.txt"
    edge_path = source_root / "out1_graph_edges.txt"
    _download(node_url, node_path)
    _download(edge_url, edge_path)
    split_urls: list[str] = []
    split_paths: list[Path] = []
    for split in range(10):
        split_url = f"{GEOM_GCN_BASE}/splits/{folder}_split_0.6_0.2_{split}.npz"
        split_path = source_root / f"split_{split}.npz"
        _download(split_url, split_path)
        split_urls.append(split_url)
        split_paths.append(split_path)

    features, labels = _parse_geom_node_file(node_path, name)
    edges = _parse_geom_edge_file(edge_path)
    train_masks: list[np.ndarray] = []
    val_masks: list[np.ndarray] = []
    test_masks: list[np.ndarray] = []
    for split_path in split_paths:
        with np.load(split_path, allow_pickle=False) as raw:
            train_masks.append(np.asarray(raw["train_mask"], dtype=bool).reshape(-1))
            val_masks.append(np.asarray(raw["val_mask"], dtype=bool).reshape(-1))
            test_masks.append(np.asarray(raw["test_mask"], dtype=bool).reshape(-1))
    path = _dataset_path(name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        node_features=features,
        node_labels=labels,
        edges=edges,
        train_masks=np.stack(train_masks, axis=1),
        val_masks=np.stack(val_masks, axis=1),
        test_masks=np.stack(test_masks, axis=1),
    )
    return path, [node_url, edge_url, *split_urls]


def download_dataset(name: str, root: str | Path) -> Path:
    normalized = _normalize_name(name)
    if normalized not in SUPPORTED_NAMES:
        raise ValueError(f"Unsupported real dataset: {name}")
    root_path = Path(root)
    path = _dataset_path(normalized, root_path)
    if path.exists():
        validate_dataset_file(path)
        return path.parent
    if normalized in HGB_NAMES:
        path, urls = _download_hgb_dataset(normalized, root_path)
        source = "Heterophilous Graph Benchmark"
    elif normalized in GEOM_GCN_NAMES:
        path, urls = _download_geom_gcn_dataset(normalized, root_path)
        source = "Geom-GCN"
    elif normalized in OPTIONAL_NAMES:
        from graphatlas.datasets.optional import download_optional_dataset
        path = download_optional_dataset(normalized, root_path)
        urls = []
        source = "PyTorch Geometric / Open Graph Benchmark"
    else:
        raise ValueError(f"No download strategy for {normalized}")
    info = validate_dataset_file(path)
    manifest = {"dataset": normalized, "source": source, "urls": urls, **info}
    (path.parent / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path.parent


def _load_npz(name: str, root: Path, split: int, feature_source: str = "native") -> GraphData:
    path = _dataset_path(name, root)
    if not path.exists():
        download_dataset(name, root)
    info = validate_dataset_file(path)
    with np.load(path, allow_pickle=True) as raw:
        if feature_source == "text":
            if "text_features" not in raw.files:
                raise ValueError(f"{name} does not provide text_features")
            x_np = _dense_array(raw["text_features"]).astype(np.float32, copy=False)
        elif feature_source == "structure":
            if "structural_features" not in raw.files:
                raise ValueError(f"{name} does not provide structural_features")
            x_np = _dense_array(raw["structural_features"]).astype(np.float32, copy=False)
        elif feature_source == "text_plus_structure":
            if "text_features" not in raw.files or "structural_features" not in raw.files:
                raise ValueError(f"{name} requires text_features and structural_features")
            x_np = np.concatenate([
                _dense_array(raw["text_features"]).astype(np.float32, copy=False),
                _dense_array(raw["structural_features"]).astype(np.float32, copy=False),
            ], axis=1)
        else:
            x_np = _dense_array(raw["node_features"]).astype(np.float32, copy=False)
        y_raw = np.asarray(raw["node_labels"])
        if y_raw.ndim == 2 and y_raw.shape[1] == 1:
            y_raw = y_raw.reshape(-1)
        y_np = y_raw.astype(np.float32 if y_raw.ndim == 2 else np.int64, copy=False)
        edges_np = np.asarray(raw["edges"]).astype(np.int64, copy=False)
        if edges_np.shape[0] == 2 and edges_np.shape[1] != 2:
            edges_np = edges_np.T
        train_masks = _orient_masks(raw["train_masks"], x_np.shape[0])
        val_masks = _orient_masks(raw["val_masks"], x_np.shape[0])
        test_masks = _orient_masks(raw["test_masks"], x_np.shape[0])
        edge_attr_np = np.asarray(raw["edge_attr"]) if "edge_attr" in raw.files else None
        node_type_np = np.asarray(raw["node_type"]) if "node_type" in raw.files else None
        edge_type_np = np.asarray(raw["edge_type"]) if "edge_type" in raw.files else None

    split_index = int(split) % train_masks.shape[1]
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    edge_index = torch.from_numpy(edges_np).to(torch.long).t().contiguous()
    preserve_edge_metadata = edge_attr_np is not None or edge_type_np is not None
    if not preserve_edge_metadata:
        edge_index = coalesce_undirected(edge_index, x.shape[0])
    metric = "roc_auc" if name in ROC_AUC_NAMES else "accuracy"
    if name in HGB_NAMES:
        source = "Heterophilous Graph Benchmark"
    elif name in GEOM_GCN_NAMES:
        source = "Geom-GCN"
    else:
        source = "PyTorch Geometric / Open Graph Benchmark / local import"
    data = GraphData(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=torch.from_numpy(train_masks[:, split_index]),
        val_mask=torch.from_numpy(val_masks[:, split_index]),
        test_mask=torch.from_numpy(test_masks[:, split_index]),
        edge_attr=torch.from_numpy(edge_attr_np.astype(np.float32, copy=False)) if edge_attr_np is not None else None,
        node_type=torch.from_numpy(node_type_np.astype(np.int64, copy=False)) if node_type_np is not None else None,
        edge_type=torch.from_numpy(edge_type_np.astype(np.int64, copy=False)) if edge_type_np is not None else None,
        metadata={
            "name": name,
            "split": split_index,
            "num_splits": int(train_masks.shape[1]),
            "metric": "roc_auc" if y_np.ndim == 2 else metric,
            "multilabel": bool(y_np.ndim == 2),
            "directed": bool(preserve_edge_metadata),
            "source": source,
            "sha256": info["sha256"],
        },
    )
    data.validate()
    return data


def load_real_dataset(name: str, root: str | Path, split: int = 0, feature_source: str = "native") -> GraphData:
    normalized = _normalize_name(name)
    path = _dataset_path(normalized, root)
    if normalized not in SUPPORTED_NAMES and not path.exists():
        raise ValueError(
            f"Unsupported or not-yet-imported dataset: {name}. "
            "Use scripts/import_local_data.py to create a standardized NPZ first."
        )
    return _load_npz(normalized, Path(root), split, feature_source=feature_source)
