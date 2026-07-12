from __future__ import annotations

from graphatlas.config import DatasetConfig, ExperimentConfig, LossConfig, ModelConfig, TrainConfig
from graphatlas.data import prepare_link_prediction_data
from graphatlas.datasets.synthetic import generate_atlas_het
from graphatlas.trainer import Trainer


def _edge_set(edge_index):
    return {tuple(sorted((int(u), int(v)))) for u, v in edge_index.t().tolist() if int(u) != int(v)}


def test_link_split_is_deterministic_and_leakage_free():
    original = generate_atlas_het(DatasetConfig(num_nodes=64, feature_dim=8, knn=4), seed=4)
    first = prepare_link_prediction_data(original, seed=9)
    second = prepare_link_prediction_data(original, seed=9)
    assert first.link_split is not None and second.link_split is not None
    assert first.link_split.test_pos.equal(second.link_split.test_pos)
    full_positive = _edge_set(original.edge_index)
    train_positive = _edge_set(first.edge_index)
    val_positive = _edge_set(first.link_split.val_pos)
    test_positive = _edge_set(first.link_split.test_pos)
    negative = _edge_set(first.link_split.train_neg) | _edge_set(first.link_split.val_neg) | _edge_set(first.link_split.test_neg)
    assert not train_positive.intersection(val_positive | test_positive)
    assert not full_positive.intersection(negative)


def test_link_prediction_training_smoke(tmp_path):
    dataset = DatasetConfig(
        num_nodes=56,
        num_charts=3,
        feature_dim=8,
        knn=4,
        relation_edges_per_node=1,
        task="link_prediction",
    )
    data = prepare_link_prediction_data(generate_atlas_het(dataset, seed=5), seed=5)
    config = ExperimentConfig(
        dataset=dataset,
        model=ModelConfig(name="graphatlas", hidden_dim=12, observation_dim=5, vector_channels=2, num_charts=3, num_layers=1, dropout=0.0),
        loss=LossConfig(reconstruction=0.05, cocycle=0.0, metric=0.0, geometry=0.0, sample_nodes=8),
        train=TrainConfig(seed=5, epochs=2, patience=2, eval_every=1, progress=False, resume=False),
    )
    _, metrics = Trainer(config).fit(data, tmp_path / "link")
    assert metrics["task"] == "link_prediction"
    assert 0.0 <= metrics["test_link_roc_auc"] <= 1.0
    assert 0.0 <= metrics["test_link_average_precision"] <= 1.0
    assert metrics["invariance_error_nonlinear"] < 1e-6
