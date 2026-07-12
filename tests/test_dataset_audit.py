from __future__ import annotations

from graphatlas.config import DatasetConfig
from graphatlas.dataset_audit import audit_graph
from graphatlas.datasets.synthetic import generate_atlas_het


def test_dataset_audit_contains_reproducibility_hashes():
    data = generate_atlas_het(DatasetConfig(num_nodes=30, feature_dim=6, num_charts=3, knn=3), seed=1)
    record = audit_graph(data, "toy")
    assert record["dataset"] == "toy"
    assert len(record["feature_hash"]) == 64
    assert len(record["edge_set_hash"]) == 64
    assert "adjusted_homophily" in record
