from __future__ import annotations

import torch

from graphatlas.config import DatasetConfig, ModelConfig
from graphatlas.datasets.synthetic import generate_atlas_het
from graphatlas.nn.charts import SmoothDiffeomorphism
from graphatlas.nn.model import GraphAtlas


def _small_data():
    return generate_atlas_het(
        DatasetConfig(num_nodes=28, num_charts=3, feature_dim=8, knn=4, relation_edges_per_node=1),
        seed=3,
    )


def _config(name: str = "graphatlas") -> ModelConfig:
    return ModelConfig(
        name=name,
        hidden_dim=16,
        observation_dim=6,
        chart_dim=2,
        vector_channels=3,
        num_charts=3,
        num_layers=2,
        dropout=0.0,
        jacobian_chunk_size=16,
    )


def test_nonlinear_reparameterization_invariance():
    torch.manual_seed(4)
    data = _small_data()
    model = GraphAtlas(data.num_features, data.num_classes, _config()).eval()
    original = model(data)
    reps = [SmoothDiffeomorphism.random(2, data.x.device, data.x.dtype, 100 + k, nonlinear=True) for k in range(3)]
    transformed = model(data, reparameterizations=reps)
    assert torch.allclose(original["observation_vectors"], transformed["observation_vectors"], atol=2e-6, rtol=2e-5)
    assert torch.allclose(original["logits"], transformed["logits"], atol=2e-6, rtol=2e-5)


def test_source_only_no_transport_is_not_invariant():
    torch.manual_seed(5)
    data = _small_data()
    model = GraphAtlas(data.num_features, data.num_classes, _config("graphatlas_no_transport")).eval()
    original = model(data)
    reps = [SmoothDiffeomorphism.random(2, data.x.device, data.x.dtype, 200 + k, nonlinear=True) for k in range(3)]
    transformed = model(data, reparameterizations=reps)
    difference = (original["logits"] - transformed["logits"]).abs().max().item()
    assert difference > 1e-6


def test_training_mode_dropout_remains_equivariant_with_matched_randomness():
    torch.manual_seed(31)
    data = _small_data()
    config = _config()
    config.dropout = 0.25
    model = GraphAtlas(data.num_features, data.num_classes, config).train()
    reps = [SmoothDiffeomorphism.random(2, data.x.device, data.x.dtype, 310 + k, nonlinear=True) for k in range(3)]
    torch.manual_seed(99)
    original = model(data)
    torch.manual_seed(99)
    transformed = model(data, reparameterizations=reps)
    assert torch.allclose(original["logits"], transformed["logits"], atol=4e-6, rtol=4e-5)
