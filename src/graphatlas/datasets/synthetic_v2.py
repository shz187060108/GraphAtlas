from __future__ import annotations

import math

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

from graphatlas.config import DatasetConfig
from graphatlas.data import GraphData, coalesce_undirected


def _surface_geometry(config: DatasetConfig, canonical: torch.Tensor):
    points = canonical.detach().to(torch.float64).requires_grad_(True)
    x, y = points[:, 0], points[:, 1]
    width = config.surface_region_width
    positive_window = torch.exp(-((x + 2.0) / width).pow(4))
    negative_window = torch.exp(-((x - 2.0) / width).pow(4))
    positive_bowl = 0.5 * ((x + 2.0).square() + y.square())
    negative_saddle = 0.5 * ((x - 2.0).square() - y.square())
    height = (
        config.surface_positive_amplitude * positive_window * positive_bowl
        + config.surface_negative_amplitude * negative_window * negative_saddle
    )
    gradient = torch.autograd.grad(height.sum(), points, create_graph=True)[0]
    hessian_columns = [
        torch.autograd.grad(gradient[:, column].sum(), points, retain_graph=True)[0]
        for column in range(2)
    ]
    hessian = torch.stack(hessian_columns, dim=1)
    f_x, f_y = gradient[:, 0], gradient[:, 1]
    curvature = (hessian[:, 0, 0] * hessian[:, 1, 1] - hessian[:, 0, 1].square()) / (
        1.0 + f_x.square() + f_y.square()
    ).square()
    metric = torch.eye(2, dtype=points.dtype)[None] + gradient[:, :, None] * gradient[:, None, :]
    surface = torch.stack([x, y, height], dim=-1)
    normal = torch.stack([-f_x, -f_y, torch.ones_like(f_x)], dim=-1)
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-12)
    return tuple(value.detach().to(torch.float32) for value in (surface, normal, metric, curvature))


def _memberships(x: torch.Tensor, config: DatasetConfig, boundary_stress: bool):
    centers = torch.linspace(-2.6, 2.6, config.num_charts, dtype=x.dtype)
    spacing = float(centers[1] - centers[0]) if config.num_charts > 1 else 6.0
    overlap = max(config.overlap, 0.45) if boundary_stress else config.overlap
    half_width = spacing * ((0.65 + overlap) if boundary_stress else (0.52 + 0.5 * overlap))
    raw = torch.relu(1.0 - (x[:, None] - centers[None]).abs() / half_width)
    uncovered = raw.sum(dim=-1) == 0
    if bool(uncovered.any()):
        nearest = (x[uncovered, None] - centers[None]).abs().argmin(dim=-1)
        raw[uncovered] = 0
        raw[uncovered, nearest] = 1
    return raw / raw.sum(dim=-1, keepdim=True), centers, half_width


def _stratified_masks(labels: np.ndarray, boundary: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    strata = labels.astype(np.int64) * 2 + boundary.astype(np.int64)
    train = np.zeros(len(labels), dtype=bool)
    val = np.zeros(len(labels), dtype=bool)
    test = np.zeros(len(labels), dtype=bool)
    for stratum in np.unique(strata):
        ids = np.flatnonzero(strata == stratum)
        rng.shuffle(ids)
        n_train = max(1, int(0.6 * len(ids)))
        n_val = max(1, int(0.2 * len(ids))) if len(ids) >= 3 else 0
        train[ids[:n_train]] = True
        val[ids[n_train:n_train + n_val]] = True
        test[ids[n_train + n_val:]] = True
    # Very small strata can leave a split empty; deterministic fallback fixes it.
    for mask in (train, val, test):
        if not mask.any():
            mask[rng.integers(0, len(labels))] = True
    train[val | test] = False
    val[test] = False
    return torch.from_numpy(train), torch.from_numpy(val), torch.from_numpy(test)


def _intrinsic_edges(canonical: torch.Tensor, surface: torch.Tensor, metric: torch.Tensor, config: DatasetConfig):
    n = canonical.shape[0]
    candidate_count = min(max(config.intrinsic_candidate_neighbors, config.knn) + 1, n)
    neighbors = NearestNeighbors(n_neighbors=candidate_count).fit(surface.numpy())
    _, candidates = neighbors.kneighbors(surface.numpy())
    edges: list[tuple[int, int]] = []
    for i in range(n):
        js = torch.as_tensor(candidates[i, 1:], dtype=torch.long)
        delta = canonical[js] - canonical[i]
        midpoint_metric = 0.5 * (metric[i][None] + metric[js])
        length_squared = torch.einsum("ni,nij,nj->n", delta, midpoint_metric, delta)
        chosen = js[torch.argsort(length_squared)[: min(config.knn, len(js))]]
        edges.extend((i, int(j)) for j in chosen)
    return edges


def _relation_edges(labels: torch.Tensor, dominant: torch.Tensor, config: DatasetConfig, seed: int, boundary_stress: bool):
    rng = np.random.default_rng(seed)
    labels_np = labels.numpy()
    dominant_np = dominant.numpy()
    n = len(labels_np)
    target_cross = max(config.cross_chart_edge_fraction, 0.50) if boundary_stress else config.cross_chart_edge_fraction
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for _ in range(max(1, config.relation_edges_per_node)):
            force_cross = rng.random() < target_cross
            candidates = np.arange(n)
            if force_cross:
                candidates = candidates[dominant_np != dominant_np[i]]
            else:
                candidates = candidates[dominant_np == dominant_np[i]]
                different_label = rng.random() < config.heterophily
                label_mask = (labels_np != labels_np[i]) if different_label else (labels_np == labels_np[i])
                matching = candidates[label_mask[candidates]]
                if len(matching):
                    candidates = matching
            candidates = candidates[candidates != i]
            if len(candidates):
                edges.append((i, int(rng.choice(candidates))))
    cross = sum(dominant_np[i] != dominant_np[j] for i, j in edges)
    return edges, cross / max(len(edges), 1)


def _edge_lengths(edge_index: torch.Tensor, canonical: torch.Tensor, metric: torch.Tensor):
    source, target = edge_index
    delta = canonical[target] - canonical[source]
    midpoint_metric = 0.5 * (metric[source] + metric[target])
    return torch.sqrt(torch.einsum("ni,nij,nj->n", delta, midpoint_metric, delta).clamp_min(1e-12))


def _geodesic_references(
    edge_index: torch.Tensor,
    lengths: torch.Tensor,
    dominant: torch.Tensor,
    pair_count: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    n = int(dominant.shape[0])
    same: list[tuple[int, int]] = []
    cross: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    target_same = pair_count // 2
    target_cross = pair_count - target_same
    attempts = 0
    while (len(same) < target_same or len(cross) < target_cross) and attempts < pair_count * 100:
        i, j = map(int, rng.integers(0, n, size=2))
        attempts += 1
        if i == j:
            continue
        pair = (min(i, j), max(i, j))
        if pair in seen:
            continue
        is_same = int(dominant[i]) == int(dominant[j])
        if (is_same and len(same) < target_same) or (not is_same and len(cross) < target_cross):
            seen.add(pair)
            (same if is_same else cross).append(pair)
    pairs = same + cross
    source, target = edge_index.numpy()
    graph = csr_matrix((lengths.numpy(), (source, target)), shape=(n, n))
    unique_sources = sorted({i for i, _ in pairs})
    distances = dijkstra(graph, directed=False, indices=unique_sources)
    lookup = {node: row for row, node in enumerate(unique_sources)}
    valid_pairs, valid_distances = [], []
    for i, j in pairs:
        distance = float(distances[lookup[i], j])
        if np.isfinite(distance) and distance > 0:
            valid_pairs.append((i, j))
            valid_distances.append(distance)
    return (
        torch.tensor(valid_pairs, dtype=torch.long).T.contiguous(),
        torch.tensor(valid_distances, dtype=torch.float32),
    )


def generate_mixed_metric_atlas_het(
    config: DatasetConfig,
    seed: int,
    boundary_stress: bool = False,
) -> GraphData:
    if config.latent_dim != 2:
        raise ValueError("Mixed-metric Atlas-Het requires latent_dim=2")
    generator = torch.Generator().manual_seed(seed)
    canonical = torch.stack(
        [
            torch.empty(config.num_nodes).uniform_(-3.0, 3.0, generator=generator),
            torch.empty(config.num_nodes).uniform_(-1.0, 1.0, generator=generator),
        ],
        dim=-1,
    )
    surface, normal, metric, curvature = _surface_geometry(config, canonical)
    x_coordinate = canonical[:, 0]
    curvature_tolerance, flat_tolerance = 2e-2, 2e-2
    geometry_region = torch.zeros(config.num_nodes, dtype=torch.long)
    geometry_region[curvature > curvature_tolerance] = 1
    geometry_region[curvature < -curvature_tolerance] = 2

    membership, centers, half_width = _memberships(x_coordinate, config, boundary_stress)
    active = membership > 0
    boundary = active.sum(dim=-1) > 1
    dominant = membership.argmax(dim=-1)

    true_coordinates = torch.full((config.num_nodes, config.num_charts, 2), torch.nan)
    true_jacobians = torch.full((config.num_nodes, config.num_charts, 2, 2), torch.nan)
    true_chart_metrics = torch.full_like(true_jacobians, torch.nan)
    rng = np.random.default_rng(seed + 31)
    for chart, center in enumerate(centers):
        domain = active[:, chart]
        base = torch.stack([(x_coordinate - center) / half_width, canonical[:, 1]], dim=-1)
        matrix = torch.tensor(rng.normal(size=(2, 2)), dtype=torch.float32)
        orthogonal, _ = torch.linalg.qr(matrix)
        scale = torch.tensor(rng.uniform(0.65, 1.55, size=2), dtype=torch.float32)
        shift = torch.tensor(rng.normal(0.0, 0.25, size=2), dtype=torch.float32)
        transformed = (torch.asinh(base) * scale) @ orthogonal.T + shift
        true_coordinates[domain, chart] = transformed[domain]
        base_scale = torch.tensor([1.0 / half_width, 1.0])
        diagonal = scale[None] * torch.rsqrt(1.0 + base.square()) * base_scale[None]
        jacobian = orthogonal[None] @ torch.diag_embed(diagonal)
        true_jacobians[domain, chart] = jacobian[domain]
        inverse = torch.linalg.inv(jacobian[domain])
        true_chart_metrics[domain, chart] = inverse.transpose(-1, -2) @ metric[domain] @ inverse

    phase = torch.floor((canonical[:, 1] + 1.0) * config.num_classes).long()
    periodic = (torch.sin(0.75 * math.pi * canonical[:, 0]) > 0).long()
    labels = (phase + geometry_region + periodic) % config.num_classes

    intrinsic = _intrinsic_edges(canonical, surface, metric, config)
    relation, cross_fraction = _relation_edges(labels, dominant, config, seed + 47, boundary_stress)
    all_edges = torch.tensor(intrinsic + relation, dtype=torch.long).T.contiguous()
    edge_index = coalesce_undirected(all_edges, config.num_nodes)
    true_edge_lengths = _edge_lengths(edge_index, canonical, metric)

    invariant_basis = torch.cat(
        [
            surface,
            normal,
            torch.sin(canonical),
            torch.cos(canonical),
            torch.sin(0.5 * canonical),
        ],
        dim=-1,
    )
    invariant_projection = torch.randn(
        invariant_basis.shape[1], config.feature_dim, generator=generator
    ) / math.sqrt(invariant_basis.shape[1])
    invariant_features = invariant_basis @ invariant_projection

    local_features = []
    for chart in range(config.num_charts):
        coordinate = torch.nan_to_num(true_coordinates[:, chart])
        first, second = coordinate[:, 0], coordinate[:, 1]
        basis = torch.stack(
            [first, second, torch.sin(math.pi * first), torch.cos(math.pi * first), first * second,
             first.square(), second.square(), torch.sin(2 * math.pi * second)],
            dim=-1,
        )
        projection = torch.randn(basis.shape[1], config.feature_dim, generator=generator) / math.sqrt(basis.shape[1])
        local_features.append(basis @ projection)
    local_features_tensor = torch.stack(local_features, dim=1)
    mixed_local = (local_features_tensor * membership[:, :, None]).sum(dim=1)
    strength = config.coordinate_shift_strength
    features = (1.0 - strength) * invariant_features + strength * mixed_local
    features = features + config.noise * torch.randn(features.shape, generator=generator)
    features = (features - features.mean(dim=0, keepdim=True)) / features.std(dim=0, keepdim=True).clamp_min(1e-6)

    train, val, test = _stratified_masks(labels.numpy(), boundary.numpy(), seed + 17)
    pairs, distances = _geodesic_references(
        edge_index, true_edge_lengths, dominant, config.geodesic_pair_count, seed + 71
    )
    region_counts = {str(region): int((geometry_region == region).sum()) for region in range(3)}
    region_means = {
        str(region): float(curvature[geometry_region == region].mean())
        for region in range(3)
        if bool((geometry_region == region).any())
    }
    return GraphData(
        x=features.to(torch.float32),
        edge_index=edge_index,
        y=labels,
        train_mask=train,
        val_mask=val,
        test_mask=test,
        boundary_mask=boundary,
        chart_membership=membership,
        true_chart_coordinates=true_coordinates,
        latent_positions=canonical,
        geometry_region=geometry_region,
        true_metric_tensors=metric,
        true_curvature=curvature,
        true_chart_jacobians=true_jacobians,
        true_chart_metrics=true_chart_metrics,
        true_edge_lengths=true_edge_lengths,
        geodesic_pairs=pairs,
        geodesic_distances=distances,
        metadata={
            "name": config.name,
            "atlas_variant": "boundary_stress" if boundary_stress else "mixed_metric",
            "has_mixed_curvature": True,
            "geometry_region_counts": region_counts,
            "curvature_mean": float(curvature.mean()),
            "curvature_min": float(curvature.min()),
            "curvature_max": float(curvature.max()),
            "geometry_region_curvature_mean": region_means,
            "overlap_ratio": float(boundary.float().mean()),
            "cross_chart_edge_fraction": float(cross_fraction),
            "metric": "accuracy",
        },
    )
