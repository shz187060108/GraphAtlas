from __future__ import annotations

from graphatlas.config import DatasetConfig
from graphatlas.data import GraphData, HeteroGraphData, prepare_link_prediction_data
from .h2gb import is_h2gb_name, load_h2gb_dataset
from .real import download_dataset, load_real_dataset
from .synthetic import generate_atlas_het
from .names import is_atlas_het_name


def _append_type_relation_features(data: GraphData, include_relation: bool) -> GraphData:
    import torch

    blocks = [data.x]
    if data.node_type is not None:
        num_types = int(data.node_type.max().item()) + 1 if data.node_type.numel() else 0
        if num_types:
            blocks.append(torch.nn.functional.one_hot(data.node_type, num_classes=num_types).to(data.x.dtype))
    if include_relation and data.edge_type is not None and data.edge_type.numel():
        num_relations = int(data.edge_type.max().item()) + 1
        source, target = data.edge_index
        relation_signature = torch.zeros(
            data.num_nodes,
            2 * num_relations,
            dtype=data.x.dtype,
            device=data.x.device,
        )
        for relation in range(num_relations):
            mask = data.edge_type.eq(relation)
            if not bool(mask.any()):
                continue
            relation_signature[:, 2 * relation].index_add_(
                0, source[mask], torch.ones_like(source[mask], dtype=data.x.dtype)
            )
            relation_signature[:, 2 * relation + 1].index_add_(
                0, target[mask], torch.ones_like(target[mask], dtype=data.x.dtype)
            )
        # log1p avoids a few high-degree relations dominating chart discovery.
        blocks.append(torch.log1p(relation_signature))
    data.x = torch.cat(blocks, dim=-1)
    return data


def load_dataset(config: DatasetConfig, seed: int) -> GraphData | HeteroGraphData:
    name = config.name.lower().replace("-", "_")
    if is_h2gb_name(name):
        data = load_h2gb_dataset(name, config.root)
        if config.task != "node_classification":
            raise ValueError("Native H2GB integration currently supports node classification only")
        return data
    if is_atlas_het_name(name):
        data = generate_atlas_het(config, seed)
    else:
        if config.local_path:
            from graphatlas.datasets.local import LocalDatasetCandidate, convert_candidate, directory_fingerprint, sha256_file
            from pathlib import Path
            source = Path(config.local_path)
            fingerprint = sha256_file(source) if source.is_file() else directory_fingerprint(source)
            candidate = LocalDatasetCandidate(
                dataset=name, path=str(source), format=config.format, confidence="user",
                size_bytes=source.stat().st_size if source.is_file() else 0, fingerprint=fingerprint
            )
            convert_candidate(candidate, config.root)
        data = load_real_dataset(name, config.root, config.split, feature_source=config.feature_source)
        if config.metric:
            data.metadata = dict(data.metadata or {})
            data.metadata["metric"] = config.metric
        if config.feature_source in {"native_plus_type", "native_plus_type_relation"}:
            data = _append_type_relation_features(
                data,
                include_relation=config.feature_source == "native_plus_type_relation",
            )
        if config.max_nodes is not None and data.num_nodes > config.max_nodes:
            raise ValueError(
                f"{name} has {data.num_nodes} nodes, above dataset.max_nodes={config.max_nodes}. "
                "Use a sampling-capable preset for large graphs."
            )
    if config.task == "link_prediction":
        data = prepare_link_prediction_data(
            data,
            seed=seed + 100_003 * int(config.split),
            val_ratio=config.link_val_ratio,
            test_ratio=config.link_test_ratio,
            negative_ratio=config.negative_ratio,
        )
    elif config.task != "node_classification":
        raise ValueError(f"Unsupported task: {config.task}")
    else:
        metadata = dict(data.metadata or {})
        metadata["task"] = "node_classification"
        data.metadata = metadata
    data.validate()
    return data


__all__ = ["GraphData", "HeteroGraphData", "load_dataset", "download_dataset", "generate_atlas_het", "is_atlas_het_name", "is_h2gb_name"]
