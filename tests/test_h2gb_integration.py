from __future__ import annotations

import json

import torch

from graphatlas.config import ModelConfig
from graphatlas.config import ExperimentConfig
from graphatlas.data import HeteroGraphData
from graphatlas.datasets.h2gb import load_h2gb_dataset
from graphatlas.nn.heterogeneous import build_hetero_model
from graphatlas.trainer import Trainer


def tiny_hetero() -> HeteroGraphData:
    data = HeteroGraphData(
        x_dict={"paper": torch.randn(6, 3), "author": None},
        num_nodes_dict={"paper": 6, "author": 4},
        edge_index_dict={
            ("author", "writes", "paper"): torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
            ("paper", "cites", "paper"): torch.tensor([[0, 1, 2], [1, 2, 3]]),
        },
        task_entity="paper",
        y=torch.tensor([0, 1, 0, 1, -1, -1]),
        train_mask=torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool),
        val_mask=torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.bool),
        test_mask=torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.bool),
        metadata={"primary_metric": "accuracy"},
        global_node_id_dict={"paper": torch.arange(6), "author": torch.arange(4)},
    )
    data.validate()
    return data


def test_heterogeneous_ids_are_reversible() -> None:
    data = tiny_hetero()
    packed = data.pack_ids("author", torch.tensor([0, 3]))
    type_ids, local = data.unpack_ids(packed)
    assert data.node_types[type_ids[0]] == "author"
    assert torch.equal(local, torch.tensor([0, 3]))


def test_native_atn_forward_is_finite() -> None:
    data = tiny_hetero()
    config = ModelConfig(name="atn", heterogeneous=True, hidden_dim=12, observation_dim=8,
                         chart_dim=2, vector_channels=2, num_charts=2, membership_topk=2)
    output = build_hetero_model(data, config)(data)
    assert output["logits"].shape == (6, 2)
    assert torch.isfinite(output["logits"]).all()
    assert output["layer_diagnostics"]


def test_canonical_cache_round_trip(tmp_path) -> None:
    data = tiny_hetero()
    processed = tmp_path / "h2gb" / "h2gb_mag_year" / "processed"
    processed.mkdir(parents=True)
    payload = {
        "schema_version": 1, "dataset_name": "h2gb_mag_year", "benchmark_family": "H2GB",
        "source_repository": "junhongmit/H2GB", "source_revision": "test",
        "node_types": data.node_types, "edge_types": data.edge_types, "task_entity": data.task_entity,
        "x_dict": data.x_dict, "num_nodes_dict": data.num_nodes_dict, "edge_index_dict": data.edge_index_dict,
        "target_y": data.y, "train_mask": data.train_mask, "val_mask": data.val_mask,
        "test_mask": data.test_mask, "primary_metric": "accuracy", "metadata": data.metadata,
        "global_node_id_dict": data.global_node_id_dict,
    }
    torch.save(payload, processed / "graphatlas_h2gb.pt")
    (processed / "manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    loaded = load_h2gb_dataset("h2gb_mag_year", tmp_path)
    assert loaded.edge_types == data.edge_types
    assert loaded.x_dict["author"] is None


def test_heterogeneous_neighbor_cpu_training(tmp_path) -> None:
    data = tiny_hetero()
    config = ExperimentConfig.from_dict({
        "dataset": {"name": "h2gb_mag_year", "num_charts": 2, "heterogeneous": True},
        "model": {"name": "mlp_typed", "heterogeneous": True, "hidden_dim": 8,
                  "num_charts": 2, "membership_topk": 1},
        "train": {"device": "cpu", "mode": "hetero_neighbor", "epochs": 1,
                  "patience": 1, "eval_every": 1, "batch_size": 2,
                  "neighbor_sizes": [2], "progress": False},
    })
    _, metrics = Trainer(config).fit(data, tmp_path / "run")
    assert metrics["neighbor_sampling"] is True
    assert metrics["sampled_batches"] > 0
