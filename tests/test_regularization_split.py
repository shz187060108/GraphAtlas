from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from graphatlas.config import DatasetConfig, LossConfig, ModelConfig
from graphatlas.datasets.synthetic import generate_atlas_het
from graphatlas.experiments import _job_config
from graphatlas.losses import atlas_regularization_terms
from graphatlas.nn.model import GraphAtlas
from graphatlas.utils import load_yaml


def graphatlas_fixture():
    data = generate_atlas_het(
        DatasetConfig(num_nodes=28, num_charts=3, feature_dim=8, knn=4, relation_edges_per_node=1),
        seed=13,
    )
    config = ModelConfig(
        name="graphatlas", hidden_dim=12, observation_dim=6, chart_dim=2,
        vector_channels=2, num_charts=3, num_layers=1, dropout=0.0,
        membership_topk=2, jacobian_chunk_size=32,
    )
    model = GraphAtlas(data.num_features, data.num_classes, config).eval()
    return data, model, model(data)


def test_legacy_cocycle_resolves_both_weights():
    config = LossConfig(cocycle=0.03)
    assert config.resolved_inverse_cycle() == pytest.approx(0.03)
    assert config.resolved_path_consistency() == pytest.approx(0.03)


def test_explicit_cycle_weights_override_legacy_weight():
    config = LossConfig(cocycle=0.03, inverse_cycle=0.01, path_consistency=0.02)
    assert config.resolved_inverse_cycle() == pytest.approx(0.01)
    assert config.resolved_path_consistency() == pytest.approx(0.02)


@pytest.mark.parametrize("probes", [1, 4, 8])
def test_metric_multiple_probes_is_finite(probes: int):
    data, model, output = graphatlas_fixture()
    terms = atlas_regularization_terms(
        model,
        output,
        data,
        LossConfig(
            cocycle=0.0, inverse_cycle=0.0, path_consistency=0.0,
            metric=1.0, metric_probes=probes, chart_rank=0.0, geometry=0.0,
            sample_nodes=12,
        ),
    )
    assert terms["metric"].ndim == 0
    assert torch.isfinite(terms["metric"])
    assert torch.isfinite(terms["metric_direction_error"])
    assert torch.isfinite(terms["metric_scale_error"])


def test_no_cocycle_ablation_disables_both_resolved_terms():
    base = load_yaml("configs/base.yaml")
    preset = load_yaml("configs/presets/roman_p0.yaml")
    config = _job_config(
        base,
        preset,
        preset["datasets"][0],
        {"name": "graphatlas_no_cocycle"},
        0,
    )
    assert asdict(config.loss)["cocycle"] == 0.0
    assert config.loss.resolved_inverse_cycle() == 0.0
    assert config.loss.resolved_path_consistency() == 0.0
