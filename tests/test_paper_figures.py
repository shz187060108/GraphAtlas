from __future__ import annotations

import pandas as pd

from graphatlas.figure_style import configure_nature_style
from graphatlas.paper_figures import build_paper_figures


def _rows() -> pd.DataFrame:
    rows = []
    for overlap in (.1, .4):
        for cross in (0.0, .5):
            condition = f"ov{overlap:.2f}_cross{cross:.2f}"
            for seed in range(2):
                for model, score in (("graphatlas_c", .72), ("graphatlas_no_transport", .64), ("graphatlas_free_transition", .66), ("graphatlas_original", .69)):
                    rows.append({"dataset": "atlas_het", "task": "node_classification", "condition_id": condition,
                                 "overlap": overlap, "cross_chart_edge_fraction": cross, "model": model, "seed": seed, "split": seed,
                                 "test_metric": score + seed * .005, "boundary_accuracy": score + .02,
                                 "transportability_beta": 1.0 if model == "graphatlas_c" else 0.0,
                                 "certified_routing_mode": "certificate", "runtime_seconds": 2.0, "parameters": 1000,
                                 "routing_opportunity": .2, "irreducible_transport_risk": .1,
                                 "transportability_mean": .8, "distortion_mean": .2, "q_route_spread_mean": .1,
                                 "routing_entropy_mean": .4, "membership_entropy_mean": .5,
                                 "invariance_error_affine": 1e-8, "invariance_error_nonlinear": 1e-7,
                                 "intervention_flip_rate_nonlinear": .01})
    return pd.DataFrame(rows)


def test_paper_figure_builder_exports_svg_and_data(tmp_path):
    result = build_paper_figures(_rows(), tmp_path, "graphatlas_c")
    assert result["svg_only"]
    assert (tmp_path / "main" / "figure1_mechanism_overview.svg").exists()
    assert (tmp_path / "main" / "figure2_phase_diagram_gain.svg").exists()
    assert (tmp_path / "data" / "figure2_phase_diagram_gain.csv").exists()
    assert not result["supplementary"]["supplementary_hetgb_features"]["generated"]


def test_nature_style_is_reentrant():
    configure_nature_style()
    configure_nature_style()
