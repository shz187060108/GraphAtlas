from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from graphatlas.config import DatasetConfig
from graphatlas.data import GraphData, coalesce_undirected


@dataclass
class GroundTruthReparameterization:
    orthogonal: np.ndarray
    scale: np.ndarray
    shift: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (np.arcsinh(x) * self.scale) @ self.orthogonal.T + self.shift


def _stratified_masks(y: np.ndarray, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    train = np.zeros(len(y), dtype=bool)
    val = np.zeros(len(y), dtype=bool)
    test = np.zeros(len(y), dtype=bool)
    for label in np.unique(y):
        ids = np.flatnonzero(y == label)
        rng.shuffle(ids)
        n_train = max(1, int(0.6 * len(ids)))
        n_val = max(1, int(0.2 * len(ids)))
        train[ids[:n_train]] = True
        val[ids[n_train:n_train + n_val]] = True
        test[ids[n_train + n_val:]] = True
    return torch.from_numpy(train), torch.from_numpy(val), torch.from_numpy(test)


def _chart_memberships(t: np.ndarray, num_charts: int, overlap: float) -> np.ndarray:
    centers = np.linspace(-1.0, 1.0, num_charts)
    spacing = 2.0 / max(num_charts - 1, 1)
    half_width = spacing * (0.60 + overlap)
    weights = np.maximum(0.0, 1.0 - np.abs(t[:, None] - centers[None, :]) / half_width)
    uncovered = weights.sum(axis=1) == 0
    if uncovered.any():
        nearest = np.argmin(np.abs(t[uncovered, None] - centers[None, :]), axis=1)
        weights[uncovered] = 0.0
        weights[np.flatnonzero(uncovered), nearest] = 1.0
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def generate_atlas_het(config: DatasetConfig, seed: int) -> GraphData:
    rng = np.random.default_rng(seed)
    n = config.num_nodes
    d = config.latent_dim
    if d != 2:
        raise ValueError("The current Atlas-Het generator uses latent_dim=2.")

    t = rng.uniform(-1.0, 1.0, size=n)
    radial = rng.normal(0.0, 0.32, size=n)
    z = np.stack([t, radial + 0.22 * np.sin(2.5 * math.pi * t)], axis=1)

    phase = np.floor((t + 1.0) * config.num_classes * 1.5).astype(int)
    y = np.mod(phase + (radial > 0).astype(int), config.num_classes).astype(np.int64)

    q_true = _chart_memberships(t, config.num_charts, config.overlap)
    boundary = (q_true > 0.08).sum(axis=1) > 1

    true_coords = []
    reparams: list[GroundTruthReparameterization] = []
    for _ in range(config.num_charts):
        matrix = rng.normal(size=(d, d))
        q, _ = np.linalg.qr(matrix)
        scale = rng.uniform(0.65, 1.55, size=d)
        shift = rng.normal(0.0, 0.25, size=d)
        rep = GroundTruthReparameterization(q, scale, shift)
        reparams.append(rep)
        true_coords.append(rep.forward(z))
    true_coords_np = np.stack(true_coords, axis=1)

    nbrs = NearestNeighbors(n_neighbors=min(config.knn + 1, n), metric="euclidean").fit(z)
    _, indices = nbrs.kneighbors(z)
    src_geo = np.repeat(np.arange(n), indices.shape[1] - 1)
    dst_geo = indices[:, 1:].reshape(-1)

    src_rel: list[int] = []
    dst_rel: list[int] = []
    by_class = {label: np.flatnonzero(y == label) for label in range(config.num_classes)}
    for i in range(n):
        for _ in range(config.relation_edges_per_node):
            choose_different = rng.random() < config.heterophily
            if choose_different:
                labels = [label for label in range(config.num_classes) if label != y[i] and len(by_class[label])]
            else:
                labels = [int(y[i])]
            label = int(rng.choice(labels))
            candidates = by_class[label]
            j = int(rng.choice(candidates))
            if j == i and len(candidates) > 1:
                j = int(rng.choice(candidates[candidates != i]))
            if j != i:
                src_rel.append(i)
                dst_rel.append(j)

    edge_index = torch.tensor(
        np.stack([
            np.concatenate([src_geo, np.asarray(src_rel, dtype=np.int64)]),
            np.concatenate([dst_geo, np.asarray(dst_rel, dtype=np.int64)]),
        ]),
        dtype=torch.long,
    )
    edge_index = coalesce_undirected(edge_index, n)

    def feature_basis(points: np.ndarray) -> np.ndarray:
        first = points[:, 0]
        second = points[:, 1]
        return np.stack(
            [
                first,
                second,
                np.sin(math.pi * first),
                np.cos(math.pi * first),
                first * second,
                first ** 2,
                second ** 2,
                np.sin(2.0 * math.pi * second),
            ],
            axis=1,
        )

    # A small invariant component keeps the task identifiable, while most of
    # the observation is chart-specific.  Each chart uses an independent
    # feature map of its own coordinates.  Nodes in overlaps blend compatible
    # local observations.  This creates genuine coordinate mismatch without
    # changing the underlying latent object or labels.
    global_basis = feature_basis(z)
    global_projection = rng.normal(
        0.0,
        1.0 / math.sqrt(global_basis.shape[1]),
        size=(global_basis.shape[1], config.feature_dim),
    )
    global_features = global_basis @ global_projection

    chart_features = []
    for chart in range(config.num_charts):
        local_basis = feature_basis(true_coords_np[:, chart])
        local_projection = rng.normal(
            0.0,
            1.0 / math.sqrt(local_basis.shape[1]),
            size=(local_basis.shape[1], config.feature_dim),
        )
        chart_features.append(local_basis @ local_projection)
    chart_features_np = np.stack(chart_features, axis=1)
    local_observation = (chart_features_np * q_true[:, :, None]).sum(axis=1)
    strength = float(config.coordinate_shift_strength)
    x = (1.0 - strength) * global_features + strength * local_observation
    x = x + rng.normal(0.0, config.noise, size=(n, config.feature_dim))
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)

    train_mask, val_mask, test_mask = _stratified_masks(y, seed + 17)
    return GraphData(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        boundary_mask=torch.from_numpy(boundary),
        chart_membership=torch.tensor(q_true, dtype=torch.float32),
        true_chart_coordinates=torch.tensor(true_coords_np, dtype=torch.float32),
        latent_positions=torch.tensor(z, dtype=torch.float32),
        metadata={
            "name": "atlas_het",
            "heterophily": float(config.heterophily),
            "overlap": float(config.overlap),
            "coordinate_shift_strength": float(config.coordinate_shift_strength),
        },
    )
