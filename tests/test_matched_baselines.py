from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from graphatlas.config import DatasetConfig, ExperimentConfig, ModelConfig
from graphatlas.data import prepare_link_prediction_data
from graphatlas.datasets.synthetic import generate_atlas_het
from graphatlas.experiments import run_preset
from graphatlas.nn.charts import LocalChart
from graphatlas.nn.functional import edge_dot_scores
from graphatlas.nn.model import build_model


ROOT = Path(__file__).resolve().parents[1]


def small_data():
    config = DatasetConfig(
        num_nodes=40,
        num_charts=3,
        feature_dim=8,
        knn=4,
        relation_edges_per_node=1,
    )
    return generate_atlas_het(config, seed=7)


def model_config(name: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        hidden_dim=16,
        observation_dim=6,
        chart_dim=2,
        vector_channels=3,
        num_charts=3,
        num_layers=2,
        dropout=0.0,
        membership_topk=2,
        jacobian_chunk_size=32,
    )


def test_matched_baseline_output_shapes():
    data = small_data()
    for name in ("ambient_vector_gnn", "signature_gnn"):
        output = build_model(data.num_features, data.num_classes, model_config(name), data.num_nodes)(data)
        assert output["logits"].shape == (data.num_nodes, data.num_classes)
        assert output["embedding"].shape[0] == data.num_nodes


def test_matched_baselines_support_link_prediction_embeddings():
    data = prepare_link_prediction_data(small_data(), seed=5)
    assert data.link_split is not None
    for name in ("ambient_vector_gnn", "signature_gnn"):
        embedding = build_model(data.num_features, data.num_classes, model_config(name), data.num_nodes)(data)["embedding"]
        assert embedding.ndim == 2
        scores = edge_dot_scores(embedding, data.link_split.val_pos)
        assert scores.shape == (data.link_split.val_pos.shape[1],)


def test_ambient_model_contains_no_chart_components():
    data = small_data()
    model = build_model(data.num_features, data.num_classes, model_config("ambient_vector_gnn"), data.num_nodes)
    assert not any(isinstance(module, LocalChart) for module in model.modules())
    assert not hasattr(model, "charts")
    assert not hasattr(model, "pull_vectors")
    assert not hasattr(model, "push_vectors")


def test_signature_model_returns_real_graph_signature():
    data = small_data()
    output = build_model(data.num_features, data.num_classes, model_config("signature_gnn"), data.num_nodes)(data)
    assert output["signature"].shape[-1] == data.num_features + 2


def test_matched_baselines_are_deterministic_in_eval_mode():
    data = small_data()
    for name in ("ambient_vector_gnn", "signature_gnn"):
        torch.manual_seed(11)
        model = build_model(data.num_features, data.num_classes, model_config(name), data.num_nodes).eval()
        with torch.no_grad():
            first = model(data)["logits"]
            second = model(data)["logits"]
        assert torch.equal(first, second)


def test_ambient_compact_forward_has_only_required_outputs():
    data = small_data()
    model = build_model(data.num_features, data.num_classes, model_config("ambient_vector_gnn"), data.num_nodes)
    assert set(model(data, compact=True)) == {"logits", "embedding"}


def test_ambient_capacity_matches_graphatlas_for_roman_default():
    base = ExperimentConfig.from_yaml(ROOT / "configs" / "base.yaml")
    roman_model = replace(base.model, num_charts=4)
    graphatlas = build_model(300, 18, replace(roman_model, name="graphatlas"))
    ambient = build_model(300, 18, replace(roman_model, name="ambient_vector_gnn"))
    graphatlas_parameters = sum(parameter.numel() for parameter in graphatlas.parameters())
    ambient_parameters = sum(parameter.numel() for parameter in ambient.parameters())
    ratio = ambient_parameters / graphatlas_parameters
    assert 0.80 <= ratio <= 1.20


def test_roman_p0_dry_run_manifest_has_100_unique_jobs():
    manifest = run_preset(ROOT / "configs" / "presets" / "roman_p0.yaml", dry_run=True)
    assert len(manifest) == 100
    assert manifest.groupby("model").size().eq(10).all()
    assert set(manifest["split"]) == set(range(10))
    assert set(manifest["seed"]) == set(range(10))
    assert not manifest["run_id"].duplicated().any()
