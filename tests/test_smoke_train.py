from __future__ import annotations

from graphatlas.config import DatasetConfig, ExperimentConfig, LossConfig, ModelConfig, TrainConfig
from graphatlas.datasets.synthetic import generate_atlas_het
from graphatlas.trainer import Trainer


def test_two_epoch_smoke_train(tmp_path):
    config = ExperimentConfig(
        dataset=DatasetConfig(num_nodes=48, num_charts=3, feature_dim=8, knn=4, relation_edges_per_node=1),
        model=ModelConfig(
            name="graphatlas",
            hidden_dim=12,
            observation_dim=5,
            chart_dim=2,
            vector_channels=2,
            num_charts=3,
            num_layers=1,
            dropout=0.0,
            jacobian_chunk_size=32,
        ),
        loss=LossConfig(reconstruction=0.1, cocycle=0.0, metric=0.0, geometry=0.0, sample_nodes=8),
        train=TrainConfig(epochs=2, patience=2, eval_every=1, progress=False, output_dir=str(tmp_path), seed=7),
    )
    data = generate_atlas_het(config.dataset, seed=7)
    _, metrics = Trainer(config).fit(data, tmp_path / "run")
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert metrics["invariance_error_nonlinear"] < 1e-5
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "metrics.json").exists()
