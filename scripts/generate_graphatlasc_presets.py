#!/usr/bin/env python
"""Generate deterministic GraphAtlas-C preset grids; use --check in CI."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "presets"


def model(label: str, **extra: object) -> dict[str, object]:
    return {"name": "graphatlas_certified", "label": label, **extra}


COMMON = {"base_config": "configs/base.yaml", "reporting": {"target_model": "graphatlas_c_oracle"}}


def documents() -> dict[str, dict[str, object]]:
    smoke_models = [
        {"name": "graphatlas", "label": "graphatlas_original"},
        {"name": "graphatlas_min_distortion", "label": "graphatlas_min_distortion"},
        model("graphatlas_c"), model("graphatlas_c_beta_minus_1", transportability_beta=-1.0),
        model("graphatlas_c_q_shuffle_source", certified_routing_mode="q_shuffle_source"),
        model("graphatlas_c_q_reference", certified_routing_mode="q_reference_max"),
        model("graphatlas_c_coordinate_relu", coordinate_activation="relu"),
    ]
    grid = [
        {"name": "atlas_het", "num_nodes": 480, "num_charts": 3, "overlap": overlap,
         "cross_chart_edge_fraction": cross, "condition_id": f"ov{overlap:.2f}_cross{cross:.2f}"}
        for overlap in (.10, .25, .40, .55) for cross in (0.0, .15, .30, .50)
    ]
    beta_conditions = grid[:3]
    beta_models = [model(f"graphatlas_c_beta_{value:g}", transportability_beta=value) for value in (-2, -1, 0, .5, 1, 2, 4)]
    controls = [
        model("graphatlas_c"), model("graphatlas_c_membership", certified_routing_mode="membership_only"),
        model("graphatlas_c_q_shuffle_source", certified_routing_mode="q_shuffle_source"),
        model("graphatlas_c_q_shuffle_edge", certified_routing_mode="q_shuffle_edge"),
        model("graphatlas_c_random", certified_routing_mode="random"),
        model("graphatlas_c_q_reference", certified_routing_mode="q_reference_max"),
    ]
    real = [
        {"name": name, "num_charts": 4, **({"metric": "roc_auc"} if name == "questions" else {})}
        for name in (
            "actor", "questions", "dblp", "coauthor_cs", "coauthor_physics",
            "hetgb_texas", "hetgb_actor", "hetgb_amazon",
        )
    ]
    dense_grid = [
        {"name": "atlas_het", "num_nodes": 480, "num_charts": 3, "overlap": overlap,
         "cross_chart_edge_fraction": cross, "condition_id": f"overlap={overlap:.3f}_cross_chart_edge_fraction={cross:.3f}_stage=screening"}
        for overlap in (.05, .10, .15, .225, .30, .40, .50, .60, .70)
        for cross in (0.0, .05, .10, .175, .25, .35, .45, .60, .75)
    ]
    dense_models = [
        model("graphatlas_c_oracle", certified_routing_mode="oracle_max_q"),
        model("graphatlas_c"),
        {"name": "graphatlas", "label": "graphatlas_original"},
        {"name": "graphatlas_free_transition", "label": "graphatlas_free_transition"},
        {"name": "graphatlas_no_transport", "label": "graphatlas_no_transport"},
    ]
    representative_pairs = ((.05, 0.0), (.10, .05), (.15, .175), (.225, .25), (.30, .10), (.30, .35), (.40, .45), (.50, .25), (.50, .60), (.60, .05), (.60, .60), (.70, .10), (.70, .75), (.05, .60), (.10, .75), (.15, .45))
    representative = [{"name": "atlas_het", "num_nodes": 480, "num_charts": 3, "overlap": overlap, "cross_chart_edge_fraction": cross, "condition_id": f"ov{overlap:.2f}_cross{cross:.3f}_stage=confirmatory"} for overlap, cross in representative_pairs]
    depth_models = [model("graphatlas_c_oracle", certified_routing_mode="oracle_max_q", num_layers=depth) for depth in (1, 2, 4, 8, 12)] + [{"name": "graphatlas", "label": f"graphatlas_original_depth_{depth}", "num_layers": depth} for depth in (1, 2, 4, 8, 12)]
    scaling_conditions = [
        {"name": "atlas_het", "num_nodes": nodes, "num_charts": charts, "overlap": .225, "cross_chart_edge_fraction": .25,
         "condition_id": f"num_nodes={nodes}_num_charts={charts}_membership_topk={topk}_sweep_type={sweep}"}
        for nodes, charts, topk, sweep in (
            (2000, 4, 2, "size"), (5000, 4, 2, "size"), (10000, 4, 2, "size"), (20000, 4, 2, "size"), (50000, 4, 2, "size"), (100000, 4, 2, "size"),
            (20000, 2, 1, "chart"), (20000, 3, 2, "chart"), (20000, 4, 2, "chart"), (20000, 6, 2, "chart"), (20000, 8, 2, "chart"),
            (20000, 4, 1, "topk"), (20000, 4, 2, "topk"), (20000, 4, 4, "topk"),
            (5000, 2, 1, "interaction"), (5000, 8, 4, "interaction"), (100000, 2, 1, "interaction"), (100000, 8, 4, "interaction"),
        )
    ]
    routing_synthetic = [dense_grid[index] for index in (10, 40, 70, 80)]
    routing_real = [{"name": name, "num_charts": 4, **({"metric": "roc_auc"} if name == "questions" else {})} for name in ("actor", "questions", "dblp", "coauthor_cs")]
    routing_models = [
        model("graphatlas_c_oracle", certified_routing_mode="oracle_max_q"), model("graphatlas_c"),
        {"name": "graphatlas", "label": "graphatlas_original"}, {"name": "graphatlas_min_distortion", "label": "graphatlas_min_distortion"},
        {"name": "graphatlas_no_transport", "label": "graphatlas_no_transport"}, {"name": "graphatlas_free_transition", "label": "graphatlas_free_transition"},
        model("graphatlas_membership_only", certified_routing_mode="membership_only"), model("graphatlas_q_shuffle_source", certified_routing_mode="q_shuffle_source"),
        model("graphatlas_q_shuffle_edge", certified_routing_mode="q_shuffle_edge"), model("graphatlas_random_routing", certified_routing_mode="random"),
        model("graphatlas_beta_zero", transportability_beta=0.0),
    ]
    return {
        "graphatlasc_smoke": {**COMMON, "seeds": [0], "train": {"device": "cpu", "epochs": 3, "patience": 3, "eval_every": 1, "progress": False}, "datasets": [{"name": "atlas_het", "num_nodes": 96, "num_charts": 3, "condition_id": "smoke"}], "models": smoke_models},
        "graphatlasc_phase_screening": {**COMMON, "seeds": [0, 1, 2], "train": {"epochs": 120, "patience": 25}, "datasets": grid, "models": smoke_models},
        "graphatlasc_beta_sweep": {**COMMON, "seeds": [0, 1, 2, 3, 4], "train": {"epochs": 120, "patience": 25}, "datasets": beta_conditions, "models": beta_models},
        "graphatlasc_counterfactual": {**COMMON, "seeds": [0, 1, 2, 3, 4], "train": {"epochs": 120, "patience": 25}, "datasets": beta_conditions, "models": controls},
        "graphatlasc_real_screening": {**COMMON, "seeds": [0, 1, 2], "train": {"device": "cuda", "execution_tag": "gpu_20260714", "epochs": 200, "patience": 40}, "datasets": real, "models": [{"name": "graphatlas", "label": "graphatlas_original"}, {"name": "graphatlas_min_distortion", "label": "graphatlas_min_distortion"}, model("graphatlas_c"), model("graphatlas_c_beta_minus_1", transportability_beta=-1.0)]},
        "graphatlasc_main_confirmatory": {**COMMON, "seeds": list(range(10)), "train": {"device": "cuda", "execution_tag": "gpu_20260714"}, "datasets": real, "models": [{"name": "graphatlas", "label": "graphatlas_original"}, {"name": "graphatlas_min_distortion", "label": "graphatlas_min_distortion"}, model("graphatlas_c"), model("graphatlas_c_beta_minus_1", transportability_beta=-1.0), model("graphatlas_c_q_shuffle_source", certified_routing_mode="q_shuffle_source"), model("graphatlas_c_q_reference", certified_routing_mode="q_reference_max")]},
        "graphatlasc_ogb_arxiv": {**COMMON, "seeds": [0], "train": {"device": "cuda", "execution_tag": "gpu_20260714"}, "datasets": [{"name": "ogbn_arxiv", "num_charts": 4}], "models": [model("graphatlas_c")]},
        "atn_h2gb_smoke": {**COMMON, "seeds": [0], "train": {"device": "cpu", "mode": "full_batch", "epochs": 2, "patience": 2, "eval_every": 1, "progress": False}, "datasets": [{"name": "h2gb_pdns", "num_charts": 3, "heterogeneous": True, "target_node_type": "domain_node", "metric": "f1"}], "models": [{"name": "atn", "label": "atn", "heterogeneous": True}]},
        "atn_h2gb_screening": {**COMMON, "seeds": [0, 1, 2], "train": {"mode": "hetero_neighbor", "batch_size": 256, "neighbor_sizes": [20, 15], "target_batching": True}, "datasets": [{"name": "h2gb_pdns", "num_charts": 4, "heterogeneous": True, "target_node_type": "domain_node", "metric": "f1"}, {"name": "h2gb_mag_year", "num_charts": 4, "heterogeneous": True, "target_node_type": "paper"}, {"name": "h2gb_ieee_cis", "num_charts": 4, "heterogeneous": True, "target_node_type": "transaction", "metric": "f1"}], "models": [{"name": name, "label": name, "heterogeneous": True} for name in ("mlp_typed", "rgcn", "hgt", "atn")]},
        "atn_h2gb_confirmatory": {**COMMON, "seeds": list(range(5)), "train": {"mode": "hetero_neighbor", "batch_size": 256, "neighbor_sizes": [20, 15], "target_batching": True}, "datasets": [{"name": "h2gb_pdns", "num_charts": 4, "heterogeneous": True, "target_node_type": "domain_node", "metric": "f1"}, {"name": "h2gb_mag_year", "num_charts": 4, "heterogeneous": True, "target_node_type": "paper"}, {"name": "h2gb_ieee_cis", "num_charts": 4, "heterogeneous": True, "target_node_type": "transaction", "metric": "f1"}], "models": [{"name": name, "label": name, "heterogeneous": True} for name in ("mlp_typed", "gcn_homogeneous_projection", "rgcn", "hgt", "graphatlas_typed", "atn")]},
        "atn_h2gb_mechanism": {**COMMON, "seeds": [0, 1, 2], "train": {"mode": "hetero_neighbor", "batch_size": 256, "neighbor_sizes": [20, 15], "target_batching": True}, "datasets": [{"name": "h2gb_pdns", "num_charts": 4, "heterogeneous": True, "target_node_type": "domain_node", "metric": "f1"}, {"name": "h2gb_ieee_cis", "num_charts": 4, "heterogeneous": True, "target_node_type": "transaction", "metric": "f1"}], "models": [{"name": "atn", "label": label, "heterogeneous": True, **extra} for label, extra in (("atn", {}), ("atn_beta_zero", {"transportability_beta": 0.0}), ("atn_no_stop_gradient", {"transportability_stop_gradient": False}), ("atn_membership", {"certified_routing_mode": "membership_only"}), ("atn_min_distortion", {"transport_mode": "min_distortion"}), ("atn_original", {"transport_mode": "original"}))]},
        "graphatlasc_phase_dense_screening": {**COMMON, "seeds": [0, 1, 2], "train": {"epochs": 120, "patience": 25}, "datasets": dense_grid, "models": dense_models},
        "graphatlasc_phase_dense_confirmatory": {**COMMON, "seeds": list(range(10)), "train": {"epochs": 200, "patience": 40}, "datasets": representative, "models": dense_models},
        "graphatlasc_routing_causal": {**COMMON, "seeds": [0, 1, 2, 3, 4], "train": {"epochs": 120, "patience": 25}, "datasets": [*routing_synthetic, *routing_real], "models": routing_models},
        "graphatlasc_depth_dynamics": {**COMMON, "seeds": [0, 1, 2], "train": {"epochs": 120, "patience": 25}, "datasets": [{"name": "atlas_het", "num_nodes": 480, "num_charts": 3, "overlap": .225, "cross_chart_edge_fraction": .25, "condition_id": "depth_overlap=0.225_cross=0.25"}, *real[:2]], "models": depth_models},
        "graphatlasc_scaling": {**COMMON, "seeds": [0, 1, 2], "train": {"epochs": 80, "patience": 15}, "datasets": scaling_conditions, "models": [model("graphatlas_c_oracle", certified_routing_mode="oracle_max_q"), {"name": "graphatlas", "label": "graphatlas_original"}]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    differences = []
    for name, payload in documents().items():
        path = OUT / f"{name}.yaml"
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                differences.append(str(path))
        else:
            path.write_text(text, encoding="utf-8")
    if differences:
        raise SystemExit("Generated presets are stale: " + ", ".join(differences))


if __name__ == "__main__":
    main()
