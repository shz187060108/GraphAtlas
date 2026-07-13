from __future__ import annotations

import torch

from graphatlas.config import DatasetConfig, ModelConfig
from graphatlas.datasets.synthetic import generate_atlas_het
from graphatlas.nn.model import GraphAtlas


def _data():
    return generate_atlas_het(DatasetConfig(num_nodes=32, num_charts=3, feature_dim=8, knn=3), 7)


def _model(**updates: object) -> GraphAtlas:
    config = ModelConfig(name="graphatlas_certified", hidden_dim=12, observation_dim=6, chart_dim=2,
                         vector_channels=2, num_charts=3, num_layers=1, dropout=0.0, **updates)
    return GraphAtlas(8, 3, config).eval()


def test_controls_are_finite_and_emit_diagnostics():
    data = _data()
    for control in ("certificate", "membership_only", "q_shuffle_source", "q_shuffle_edge", "random", "oracle_max_q"):
        torch.manual_seed(4)
        output = _model(certified_routing_mode=control, routing_control_max_edges=10_000)(data, transport_diagnostics=True)
        assert torch.isfinite(output["logits"]).all()
        diagnostic = output["layer_diagnostics"][0]
        assert torch.isfinite(diagnostic["q_delta_closure_error"])
        assert diagnostic["node_q_spread"].shape[0] == data.num_nodes


def test_beta_zero_matches_minimum_distortion():
    data = _data()
    torch.manual_seed(15)
    certified = _model(transportability_beta=0.0)
    torch.manual_seed(15)
    config = ModelConfig(name="graphatlas_min_distortion", hidden_dim=12, observation_dim=6, chart_dim=2,
                         vector_channels=2, num_charts=3, num_layers=1, dropout=0.0)
    minimum = GraphAtlas(data.num_features, data.num_classes, config).eval()
    assert torch.allclose(certified(data)["logits"], minimum(data)["logits"], atol=2e-6, rtol=2e-5)
