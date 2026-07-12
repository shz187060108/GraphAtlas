from graphatlas.config import DatasetConfig
from graphatlas.datasets.synthetic import generate_atlas_het


def test_atlas_het_contains_overlap_and_ground_truth():
    data = generate_atlas_het(DatasetConfig(num_nodes=100, num_charts=3), seed=1)
    assert data.chart_membership is not None
    assert data.true_chart_coordinates is not None
    assert data.boundary_mask is not None
    assert data.boundary_mask.any()
    assert data.edge_index.shape[0] == 2
