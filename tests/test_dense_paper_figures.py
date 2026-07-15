from __future__ import annotations

import pandas as pd

from graphatlas.paper_figures_dense import build_dense_paper_figures


def test_dense_builder_emits_bundles_and_data(tmp_path):
    rows = []
    for model in ("graphatlas_c", "graphatlas_original", "graphatlas_no_transport", "graphatlas_free_transition"):
        rows.append({"dataset": "atlas_het", "task": "node_classification", "model": model, "seed": 0, "split": 0, "condition_id": "ov0.10_cross0.20", "test_metric": .7 if model == "graphatlas_c" else .6, "runtime_seconds": 1.0, "parameters": 10, "num_nodes": 20, "num_edges": 40, "boundary_accuracy": .5, "transportability_mean": .8, "distortion_mean": .2, "routing_opportunity": .3, "irreducible_transport_risk": .1})
    result = build_dense_paper_figures(pd.DataFrame(rows), tmp_path)
    assert len(result["generated"]) == 18
    assert {
        "coordinate_metric_drop_atlas", "mechanism_gain_phase_atlas",
        "certificate_calibration_atlas", "routing_intervention_atlas",
        "real_benchmark_slope_atlas", "opportunity_response_atlas",
        "rescued_failed_ego_graph_atlas", "runtime_scaling_atlas",
    }.issubset(result["generated"])
    supplementary = {"coordinate_logit_stability_atlas", "coordinate_probability_stability_atlas", "coordinate_flip_atlas", "transport_risk_phase_atlas", "routing_opportunity_phase_atlas", "routing_agreement_phase_atlas", "model_rank_atlas", "depth_performance_atlas", "memory_scaling_atlas", "transport_diagnostics_atlas"}
    for name in result["generated"]:
        assert (tmp_path / "data" / f"{name}.csv").exists()
        section = "supplementary" if name in supplementary else "main"
        assert (tmp_path / section / f"{name}.svg").exists()
        assert (tmp_path / section / f"{name}.pdf").exists()
        assert (tmp_path / section / f"{name}.png").exists()
    assert (tmp_path / "manifests" / "figure_manifest.json").exists()
