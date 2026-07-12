from __future__ import annotations

import torch

from graphatlas.config import LossConfig, ModelConfig
from graphatlas.data import GraphData
from graphatlas.losses import compute_loss
from graphatlas.metrics import evaluate_output
from graphatlas.nn.model import build_model


def test_multilabel_training_and_metrics():
    n = 12
    x = torch.randn(n, 5)
    edge_index = torch.stack([torch.arange(n), torch.roll(torch.arange(n), -1)])
    y = (torch.rand(n, 3) > 0.5).float()
    train = torch.zeros(n, dtype=torch.bool); train[:6] = True
    val = torch.zeros(n, dtype=torch.bool); val[6:9] = True
    test = torch.zeros(n, dtype=torch.bool); test[9:] = True
    data = GraphData(x, edge_index, y, train, val, test, metadata={"metric": "roc_auc", "task": "node_classification"})
    data.validate()
    model = build_model(5, 3, ModelConfig(name="mlp", hidden_dim=8, dropout=0.0), n)
    output = model(data)
    loss, _ = compute_loss(model, output, data, LossConfig(), "node_classification")
    assert torch.isfinite(loss)
    metrics = evaluate_output(model, output, data, seed=0, include_intervention=False)
    assert "test_macro_f1" in metrics
    assert "test_roc_auc" in metrics
