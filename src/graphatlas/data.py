from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import torch


@dataclass
class LinkPredictionSplit:
    train_pos: torch.Tensor
    train_neg: torch.Tensor
    val_pos: torch.Tensor
    val_neg: torch.Tensor
    test_pos: torch.Tensor
    test_neg: torch.Tensor

    def to(self, device: torch.device | str) -> "LinkPredictionSplit":
        return LinkPredictionSplit(**{field.name: getattr(self, field.name).to(device) for field in fields(self)})


@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    boundary_mask: torch.Tensor | None = None
    chart_membership: torch.Tensor | None = None
    true_chart_coordinates: torch.Tensor | None = None
    latent_positions: torch.Tensor | None = None
    link_split: LinkPredictionSplit | None = None
    metadata: dict[str, Any] | None = None
    edge_attr: torch.Tensor | None = None
    node_type: torch.Tensor | None = None
    edge_type: torch.Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_classes(self) -> int:
        if self.y.ndim == 2:
            return int(self.y.shape[1])
        return int(self.y.max().item()) + 1

    @property
    def is_multilabel(self) -> bool:
        return self.y.ndim == 2

    def validate(self) -> None:
        errors: list[str] = []
        if self.x.ndim != 2:
            errors.append(f"x must be [N,F], got {tuple(self.x.shape)}")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            errors.append(f"edge_index must be [2,E], got {tuple(self.edge_index.shape)}")
        if self.y.ndim not in {1, 2} or self.y.shape[0] != self.x.shape[0]:
            errors.append("y must have shape [N] or [N,C]")
        for name in ("train_mask", "val_mask", "test_mask"):
            mask = getattr(self, name)
            if mask.dtype != torch.bool or mask.ndim != 1 or mask.shape[0] != self.x.shape[0]:
                errors.append(f"{name} must be a boolean length-N vector")
        if self.edge_index.numel():
            if int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= self.x.shape[0]:
                errors.append("edge_index contains an out-of-range node ID")
        if self.y.numel() and self.y.ndim == 1 and int(self.y.min()) < 0:
            errors.append("single-label targets must be non-negative integers")
        if self.y.ndim == 2 and not torch.isfinite(self.y[~torch.isnan(self.y)]).all():
            errors.append("multi-label targets contain infinite values")
        if self.edge_attr is not None and self.edge_attr.shape[0] != self.edge_index.shape[1]:
            errors.append("edge_attr first dimension must equal the number of directed edges")
        if self.node_type is not None and self.node_type.shape[0] != self.x.shape[0]:
            errors.append("node_type must have length N")
        if self.edge_type is not None and self.edge_type.shape[0] != self.edge_index.shape[1]:
            errors.append("edge_type must have length E")
        overlap = self.train_mask.int() + self.val_mask.int() + self.test_mask.int()
        if bool((overlap > 1).any()):
            errors.append("train, validation, and test masks must be disjoint")
        if int(self.train_mask.sum()) == 0 or int(self.val_mask.sum()) == 0 or int(self.test_mask.sum()) == 0:
            errors.append("train, validation, and test masks must all be non-empty")
        if self.boundary_mask is not None and self.boundary_mask.shape != self.y.shape:
            errors.append("boundary_mask must have shape [N]")
        if self.chart_membership is not None:
            if self.chart_membership.ndim != 2 or self.chart_membership.shape[0] != self.x.shape[0]:
                errors.append("chart_membership must have shape [N,K]")
            elif not torch.allclose(
                self.chart_membership.sum(dim=-1),
                torch.ones(self.x.shape[0], dtype=self.chart_membership.dtype, device=self.chart_membership.device),
                atol=1e-4,
            ):
                errors.append("chart_membership rows must sum to one")
        if self.link_split is not None:
            for field in fields(self.link_split):
                edges = getattr(self.link_split, field.name)
                if edges.ndim != 2 or edges.shape[0] != 2:
                    errors.append(f"link_split.{field.name} must have shape [2,E]")
                elif edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= self.num_nodes):
                    errors.append(f"link_split.{field.name} contains an out-of-range node ID")
        if errors:
            raise ValueError("Invalid GraphData:\n- " + "\n- ".join(errors))

    def to(self, device: torch.device | str) -> "GraphData":
        payload: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                payload[field.name] = value.to(device)
            elif isinstance(value, LinkPredictionSplit):
                payload[field.name] = value.to(device)
            else:
                payload[field.name] = value
        return GraphData(**payload)


def coalesce_undirected(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index = edge_index.to(torch.long)
    reverse = edge_index.flip(0)
    merged = torch.cat([edge_index, reverse], dim=1)
    linear = merged[0] * num_nodes + merged[1]
    linear = torch.unique(linear)
    src = torch.div(linear, num_nodes, rounding_mode="floor")
    dst = linear.remainder(num_nodes)
    return torch.stack([src, dst], dim=0)


def random_masks(num_nodes: int, seed: int, train_ratio: float = 0.6, val_ratio: float = 0.2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_nodes, generator=generator)
    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[order[:train_end]] = True
    val_mask[order[train_end:val_end]] = True
    test_mask[order[val_end:]] = True
    return train_mask, val_mask, test_mask


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> bool:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def _unique_undirected_edges(edge_index: torch.Tensor) -> np.ndarray:
    edges = edge_index.detach().cpu().numpy().T
    unique = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges if int(u) != int(v)}
    return np.asarray(sorted(unique), dtype=np.int64)


def _sample_negative_edges(
    num_nodes: int,
    forbidden: set[tuple[int, int]],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    negatives: set[tuple[int, int]] = set()
    max_possible = num_nodes * (num_nodes - 1) // 2 - len(forbidden)
    if count > max_possible:
        raise ValueError(f"Requested {count} negative edges but only {max_possible} are available")
    while len(negatives) < count:
        batch = max(1024, 3 * (count - len(negatives)))
        left = rng.integers(0, num_nodes, size=batch)
        right = rng.integers(0, num_nodes, size=batch)
        for u, v in zip(left, right, strict=True):
            if u == v:
                continue
            edge = (int(min(u, v)), int(max(u, v)))
            if edge not in forbidden and edge not in negatives:
                negatives.add(edge)
                if len(negatives) == count:
                    break
    return np.asarray(sorted(negatives), dtype=np.int64)


def prepare_link_prediction_data(
    data: GraphData,
    seed: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    negative_ratio: float = 1.0,
) -> GraphData:
    if not 0.0 < val_ratio < 1.0 or not 0.0 < test_ratio < 1.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("link validation and test ratios must be positive and sum to less than one")
    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive")
    rng = np.random.default_rng(seed)
    positives = _unique_undirected_edges(data.edge_index)
    if len(positives) < 10:
        raise ValueError("At least ten unique edges are required for link prediction")
    order = rng.permutation(len(positives))
    shuffled = positives[order]

    # Preserve a spanning forest in the training graph. Remaining edges can be
    # held out without introducing avoidable isolated components.
    dsu = _DisjointSet(data.num_nodes)
    required_indices: list[int] = []
    optional_indices: list[int] = []
    for index, (u, v) in enumerate(shuffled):
        if dsu.union(int(u), int(v)):
            required_indices.append(index)
        else:
            optional_indices.append(index)
    val_count = max(1, int(round(val_ratio * len(positives))))
    test_count = max(1, int(round(test_ratio * len(positives))))
    if val_count + test_count > len(optional_indices):
        overflow = val_count + test_count - len(optional_indices)
        test_count = max(1, test_count - overflow)
    if val_count + test_count > len(optional_indices):
        raise ValueError("Graph has too few non-tree edges for leakage-free validation and test splits")
    val_indices = optional_indices[:val_count]
    test_indices = optional_indices[val_count : val_count + test_count]
    held_out = set(val_indices + test_indices)
    train_indices = [index for index in range(len(shuffled)) if index not in held_out]

    train_pos_np = shuffled[train_indices]
    val_pos_np = shuffled[val_indices]
    test_pos_np = shuffled[test_indices]
    forbidden = {tuple(map(int, edge)) for edge in positives}
    total_neg = int(round(negative_ratio * (len(train_pos_np) + len(val_pos_np) + len(test_pos_np))))
    negatives = _sample_negative_edges(data.num_nodes, forbidden, total_neg, rng)
    cursor = 0

    def take_negative(positive_count: int) -> np.ndarray:
        nonlocal cursor
        count = int(round(negative_ratio * positive_count))
        result = negatives[cursor : cursor + count]
        cursor += count
        return result

    train_neg_np = take_negative(len(train_pos_np))
    val_neg_np = take_negative(len(val_pos_np))
    test_neg_np = take_negative(len(test_pos_np))

    def as_edge_index(edges: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(edges.T.copy()).to(torch.long)

    train_pos = as_edge_index(train_pos_np)
    train_graph = coalesce_undirected(train_pos, data.num_nodes)
    split = LinkPredictionSplit(
        train_pos=train_pos,
        train_neg=as_edge_index(train_neg_np),
        val_pos=as_edge_index(val_pos_np),
        val_neg=as_edge_index(val_neg_np),
        test_pos=as_edge_index(test_pos_np),
        test_neg=as_edge_index(test_neg_np),
    )
    metadata = dict(data.metadata or {})
    metadata.update(
        {
            "task": "link_prediction",
            "metric": "roc_auc",
            "link_train_positive_edges": int(train_pos.shape[1]),
            "link_val_positive_edges": int(split.val_pos.shape[1]),
            "link_test_positive_edges": int(split.test_pos.shape[1]),
            "link_negative_ratio": float(negative_ratio),
        }
    )
    result = replace(data, edge_index=train_graph, link_split=split, metadata=metadata)
    result.validate()
    return result
