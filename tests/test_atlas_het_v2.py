from __future__ import annotations

import torch

from graphatlas.config import DatasetConfig
from graphatlas.datasets.synthetic import generate_atlas_het


def _config(variant: str, nodes: int = 240) -> DatasetConfig:
    return DatasetConfig(
        name=f"atlas_het_{variant}", atlas_variant=variant, num_nodes=nodes,
        num_charts=5, feature_dim=10, knn=5, intrinsic_candidate_neighbors=12,
        relation_edges_per_node=1, geodesic_pair_count=32,
        overlap=0.45 if variant == "boundary_stress" else 0.25,
        cross_chart_edge_fraction=0.50 if variant == "boundary_stress" else 0.20,
    )


def test_mixed_metric_is_deterministic_and_structurally_valid():
    first = generate_atlas_het(_config("mixed_metric"), 7)
    second = generate_atlas_het(_config("mixed_metric"), 7)
    for name in ("x", "y", "edge_index", "chart_membership", "true_curvature",
                 "true_metric_tensors", "true_edge_lengths", "geodesic_pairs",
                 "geodesic_distances"):
        left, right = getattr(first, name), getattr(second, name)
        assert torch.equal(torch.nan_to_num(left), torch.nan_to_num(right))
        assert torch.equal(torch.isnan(left), torch.isnan(right))
    first.validate()
    assert torch.allclose(first.chart_membership.sum(-1), torch.ones(first.num_nodes))
    active = first.chart_membership > 0
    assert torch.isnan(first.true_chart_coordinates[~active]).all()
    assert first.true_edge_lengths.numel() == first.edge_index.shape[1]
    assert torch.isfinite(first.geodesic_distances).all()
    assert (first.geodesic_distances > 0).all()


def test_boundary_stress_has_required_overlap_and_cross_chart_edges():
    data = generate_atlas_het(_config("boundary_stress"), 11)
    assert data.metadata["overlap_ratio"] >= 0.35
    assert data.metadata["cross_chart_edge_fraction"] >= 0.45


def test_legacy_name_still_selects_coordinate_only():
    data = generate_atlas_het(DatasetConfig(num_nodes=32, num_charts=3, knn=4), 3)
    assert data.metadata["atlas_variant"] == "coordinate_only"
    assert data.metadata["has_mixed_curvature"] is False
