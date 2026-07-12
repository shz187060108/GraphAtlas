from __future__ import annotations

from graphatlas.config import ExperimentConfig, LossConfig, TrainConfig
from graphatlas.trainer import _loss_config_for_epoch


def test_light_regularization_preserves_structural_terms_and_reduces_cost():
    config = ExperimentConfig(
        loss=LossConfig(
            cocycle=0.03,
            inverse_cycle=0.01,
            path_consistency=0.02,
            metric=0.04,
            metric_probes=8,
            chart_rank=0.05,
            geometry=0.06,
            sample_nodes=128,
        ),
        train=TrainConfig(
            regularization_every=10,
            light_regularization_nodes=24,
            light_metric_probes=2,
        ),
    )
    light = _loss_config_for_epoch(config, 1)
    assert light.resolved_inverse_cycle() == 0.01
    assert light.resolved_path_consistency() == 0.02
    assert light.metric == 0.04
    assert light.chart_rank == 0.05
    assert light.sample_nodes == 24
    assert light.metric_probes == 2
    assert light.geometry == 0.0

    full = _loss_config_for_epoch(config, 10)
    assert full == config.loss
