from __future__ import annotations

import torch

from graphatlas.config import DatasetConfig, LossConfig, ModelConfig
from graphatlas.datasets.synthetic import generate_atlas_het
from graphatlas.losses import atlas_regularization_terms
from graphatlas.nn.model import GraphAtlas


def test_route_consistency_is_non_vacuous_with_topk_two():
    torch.manual_seed(17)
    data = generate_atlas_het(
        DatasetConfig(num_nodes=36, num_charts=3, feature_dim=8, knn=4, relation_edges_per_node=1),
        seed=17,
    )
    config = ModelConfig(
        name="graphatlas",
        hidden_dim=14,
        observation_dim=6,
        chart_dim=2,
        vector_channels=2,
        num_charts=3,
        num_layers=1,
        dropout=0.0,
        membership_topk=2,
        jacobian_chunk_size=32,
    )
    model = GraphAtlas(data.num_features, data.num_classes, config).eval()
    output = model(data)
    terms = atlas_regularization_terms(
        model,
        output,
        data,
        LossConfig(cocycle=1.0, metric=0.0, geometry=0.0, sample_nodes=36),
    )
    assert torch.isfinite(terms["cocycle"])
    assert float(terms["inverse_cycle"].detach()) > 0.0
    assert float(terms["path_consistency"].detach()) > 0.0
    assert float(terms["triple_cocycle"].detach()) == 0.0
    assert float(terms["cocycle"].detach()) > float(terms["inverse_cycle"].detach())
