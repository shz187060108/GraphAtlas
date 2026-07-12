from __future__ import annotations

import torch

from graphatlas.config import DatasetConfig, LossConfig, ModelConfig
from graphatlas.datasets.synthetic import generate_atlas_het
from graphatlas.losses import atlas_regularization_terms
from graphatlas.nn.model import GraphAtlas


def build_model(chart_dim: int = 2):
    data = generate_atlas_het(
        DatasetConfig(num_nodes=30, num_charts=3, feature_dim=8, knn=4, relation_edges_per_node=1),
        seed=21,
    )
    model = GraphAtlas(
        data.num_features,
        data.num_classes,
        ModelConfig(
            name="graphatlas", hidden_dim=14, observation_dim=6, chart_dim=chart_dim,
            vector_channels=2, num_charts=3, num_layers=1, dropout=0.0,
            membership_topk=2, jacobian_chunk_size=32,
        ),
    )
    return data, model


def rank_config() -> LossConfig:
    return LossConfig(
        cocycle=0.0, inverse_cycle=0.0, path_consistency=0.0,
        metric=0.0, chart_rank=1.0, rank_margin=1.0,
        rank_max_condition=1.01, rank_condition_weight=0.1,
        geometry=0.0, sample_nodes=16,
    )


def test_chart_rank_loss_backpropagates_to_decoder():
    data, model = build_model()
    terms = atlas_regularization_terms(model, model(data), data, rank_config())
    terms["chart_rank"].backward()
    gradients = [
        parameter.grad
        for chart in model.charts
        for parameter in chart.decoder.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert any(torch.isfinite(gradient).all() and bool((gradient != 0).any()) for gradient in gradients)


def test_nondegenerate_decoder_has_larger_relative_minimum_than_degenerate_decoder():
    data, model = build_model()
    healthy = atlas_regularization_terms(model, model(data), data, rank_config())[
        "chart_min_relative_singular_value"
    ]
    with torch.no_grad():
        for chart in model.charts:
            chart.decoder.layers[-1].weight.zero_()
    degenerate = atlas_regularization_terms(model, model(data), data, rank_config())[
        "chart_min_relative_singular_value"
    ]
    assert torch.isfinite(healthy) and torch.isfinite(degenerate)
    assert float(healthy) > float(degenerate) + 1e-4


def test_chart_rank_supports_one_dimensional_charts():
    data, model = build_model(chart_dim=1)
    terms = atlas_regularization_terms(model, model(data), data, rank_config())
    assert torch.isfinite(terms["chart_rank"])
    assert torch.isfinite(terms["chart_condition_number"])
