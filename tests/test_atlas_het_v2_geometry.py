from __future__ import annotations

import torch

from graphatlas.config import DatasetConfig
from graphatlas.datasets.synthetic import generate_atlas_het


def _data():
    return generate_atlas_het(DatasetConfig(
        name="atlas_het_mixed_metric", atlas_variant="mixed_metric", num_nodes=320,
        num_charts=5, feature_dim=8, knn=5, intrinsic_candidate_neighbors=12,
        relation_edges_per_node=1, geodesic_pair_count=24,
    ), 19)


def test_curvature_regions_have_the_expected_geometry():
    data = _data()
    fractions = torch.bincount(data.geometry_region, minlength=3).float() / data.num_nodes
    assert (fractions >= 0.15).all()
    flat = data.true_curvature[data.geometry_region == 0]
    positive = data.true_curvature[data.geometry_region == 1]
    negative = data.true_curvature[data.geometry_region == 2]
    assert positive.mean() > 0
    assert negative.mean() < 0
    assert flat.abs().mean() < min(positive.abs().mean(), negative.abs().mean())


def test_true_metrics_are_symmetric_positive_definite():
    data = _data()
    assert torch.allclose(data.true_metric_tensors, data.true_metric_tensors.transpose(-1, -2))
    assert (torch.linalg.eigvalsh(data.true_metric_tensors).min(-1).values > 0).all()
    active = data.chart_membership > 0
    metrics = data.true_chart_metrics[active]
    assert torch.isfinite(metrics).all()
    assert torch.allclose(metrics, metrics.transpose(-1, -2), atol=1e-5)
    assert (torch.linalg.eigvalsh(metrics).min(-1).values > 0).all()
