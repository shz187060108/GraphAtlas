from __future__ import annotations

import subprocess
import sys

import torch

from graphatlas.config import DatasetConfig, ModelConfig
from graphatlas.datasets.synthetic import generate_atlas_het
from graphatlas.metrics import evaluate_output, predicted_edge_lengths, transition_error
from graphatlas.nn.charts import SmoothDiffeomorphism
from graphatlas.nn.model import GraphAtlas


def _case():
    data = generate_atlas_het(DatasetConfig(
        name="atlas_het_mixed_metric", atlas_variant="mixed_metric", num_nodes=80,
        num_charts=3, feature_dim=8, knn=4, intrinsic_candidate_neighbors=9,
        relation_edges_per_node=1, geodesic_pair_count=18,
    ), 5)
    config = ModelConfig(hidden_dim=12, observation_dim=6, chart_dim=2,
                         vector_channels=2, num_charts=3, num_layers=1,
                         dropout=0.0, jacobian_chunk_size=32)
    torch.manual_seed(2)
    return data, GraphAtlas(data.num_features, data.num_classes, config).eval()


def test_geometry_diagnostics_are_finite():
    data, model = _case()
    output = model(data)
    metrics = evaluate_output(model, output, data, seed=0, include_intervention=False)
    for key in ("metric_recovery_error", "transition_error", "local_distortion",
                "cross_chart_distortion"):
        assert torch.isfinite(torch.tensor(metrics[key]))


def test_diagnostics_are_chart_reparameterization_invariant():
    data, model = _case()
    original = model(data)
    reps = [SmoothDiffeomorphism.random(2, data.x.device, data.x.dtype, 80 + k, nonlinear=True)
            for k in range(3)]
    transformed = model(data, reparameterizations=reps)
    assert torch.allclose(original["logits"], transformed["logits"], atol=3e-6, rtol=3e-5)
    assert torch.allclose(predicted_edge_lengths(model, original, data),
                          predicted_edge_lengths(model, transformed, data), atol=2e-5, rtol=2e-4)
    torch.manual_seed(99)
    first = transition_error(model, original, data)
    torch.manual_seed(99)
    second = transition_error(model, transformed, data)
    assert torch.allclose(first, second, atol=2e-5, rtol=2e-4)


def test_unconfigured_external_runner_fails_without_result(tmp_path):
    process = subprocess.run([
        sys.executable, "scripts/run_external_baseline.py", "--method", "graphmore_official",
        "--dataset", "roman_empire", "--seed", "0", "--split", "0",
        "--export-dir", str(tmp_path / "export"), "--output-dir", str(tmp_path / "result"),
    ], capture_output=True, text=True, check=False)
    assert process.returncode != 0
    assert "unconfigured" in (process.stdout + process.stderr).lower()
    assert not (tmp_path / "result" / "metrics.json").exists()
