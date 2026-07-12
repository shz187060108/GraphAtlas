from __future__ import annotations

import pytest

from graphatlas.config import ExperimentConfig, ModelConfig


def test_default_config_validates():
    ExperimentConfig().validate()


def test_invalid_membership_topk_is_rejected():
    config = ExperimentConfig(model=ModelConfig(num_charts=2, membership_topk=3))
    config.dataset.num_charts = 2
    with pytest.raises(ValueError, match="membership_topk"):
        config.validate()
