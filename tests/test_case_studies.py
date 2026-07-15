from __future__ import annotations

import numpy as np

from graphatlas.case_studies import select_case_nodes, two_hop_ego


def test_case_selection_is_deterministic():
    probabilities = np.asarray([[.9, .1], [.2, .8], [.6, .4], [.1, .9]])
    labels = np.asarray([0, 1, 1, 0])
    diagnostics = {"node_transport_risk": np.asarray([.1, .9, .2, .8]), "node_cross_chart_mass": np.ones(4), "node_q_spread": np.ones(4)}
    first = select_case_nodes(probabilities, labels, np.ones(4, dtype=bool), diagnostics, cases_per_category=2)
    second = select_case_nodes(probabilities, labels, np.ones(4, dtype=bool), diagnostics, cases_per_category=2)
    assert all(np.array_equal(first[key], second[key]) for key in first)


def test_two_hop_ego_returns_local_edges():
    nodes, edges, truncated = two_hop_ego(np.asarray([[0, 1, 2], [1, 2, 3]]), 1)
    assert set(nodes.tolist()) == {0, 1, 2, 3}
    assert edges.shape[0] == 2
    assert not truncated
