from __future__ import annotations

import pytest
import torch

from graphatlas.config import DatasetConfig, ModelConfig
from graphatlas.datasets.synthetic import generate_atlas_het
from graphatlas.nn.model import build_model


@pytest.mark.parametrize("name", ["linear", "mlp", "link", "gcn", "gat", "gcnii", "fagcn", "acmgcn", "sgc", "appnp", "gprgnn", "mixhop", "linkx", "geometry_moe"])
def test_internal_baseline_output_shape(name: str):
    torch.manual_seed(2)
    data = generate_atlas_het(DatasetConfig(num_nodes=36, feature_dim=7, num_charts=3, knn=3), seed=2)
    config = ModelConfig(name=name, hidden_dim=12, num_layers=2, num_charts=3, propagation_steps=3, dropout=0.0)
    model = build_model(data.num_features, data.num_classes, config, data.num_nodes)
    output = model(data)
    assert output["logits"].shape == (data.num_nodes, data.num_classes)
    assert torch.isfinite(output["logits"]).all()
