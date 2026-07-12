from __future__ import annotations

import pytest

from graphatlas.config import DatasetConfig, ModelConfig
from graphatlas.datasets.synthetic import generate_atlas_het
from graphatlas.nn.model import build_model


@pytest.mark.parametrize(
    "name",
    [
        "linear", "mlp", "link", "gcn", "gat", "gcnii", "fagcn", "acmgcn",
        "resgcn", "sgc", "appnp", "gprgnn", "mixhop", "h2gcn",
        "linkx", "geometry_moe", "graphatlas", "graphatlas_no_transport",
        "graphatlas_free_transition", "graphatlas_no_metric", "graphatlas_no_cocycle",
        "graphatlas_no_overlap",
    ],
)
def test_every_configured_model_builds_and_runs(name: str):
    data = generate_atlas_het(
        DatasetConfig(num_nodes=24, num_charts=3, feature_dim=8, knn=3, relation_edges_per_node=1),
        seed=4,
    )
    config = ModelConfig(
        name=name,
        hidden_dim=12,
        observation_dim=6,
        chart_dim=2,
        vector_channels=2,
        num_charts=3,
        num_layers=1,
        dropout=0.0,
        membership_topk=2,
        jacobian_chunk_size=16,
        propagation_steps=2,
    )
    model = build_model(data.num_features, data.num_classes, config, data.num_nodes).eval()
    output = model(data)
    assert output["logits"].shape == (data.num_nodes, data.num_classes)
