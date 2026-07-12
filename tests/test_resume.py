from __future__ import annotations

from dataclasses import replace

import pandas as pd

from graphatlas.config import DatasetConfig, ExperimentConfig, ModelConfig, TrainConfig
from graphatlas.datasets.synthetic import generate_atlas_het
from graphatlas.trainer import Trainer


def test_epoch_checkpoint_resume(tmp_path):
    data = generate_atlas_het(DatasetConfig(num_nodes=36, feature_dim=8, knn=3), seed=8)
    config = ExperimentConfig(
        dataset=DatasetConfig(num_nodes=36, feature_dim=8, knn=3),
        model=ModelConfig(name="gcn", hidden_dim=12, num_layers=2, dropout=0.0),
        train=TrainConfig(seed=8, epochs=2, patience=10, eval_every=1, progress=False, num_threads=1),
    )
    run_dir = tmp_path / "run"
    Trainer(config).fit(data, run_dir)
    resumed = replace(config, train=replace(config.train, epochs=4))
    _, metrics = Trainer(resumed).fit(data, run_dir)
    history = pd.read_csv(run_dir / "history.csv")
    assert history["epoch"].tolist() == [1, 2, 3, 4]
    assert metrics["last_epoch"] == 4
    assert (run_dir / "last.pt").exists()
