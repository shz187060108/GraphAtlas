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


COMMON = {"base_config": "configs/base.yaml", "reporting": {"target_model": "graphatlas_c"}}


def documents() -> dict[str, dict[str, object]]:
    smoke_models = [
        {"name": "graphatlas", "label": "graphatlas_original"},
        {"name": "graphatlas_min_distortion", "label": "graphatlas_min_distortion"},
        model("graphatlas_c"), model("graphatlas_c_beta_minus_1", transportability_beta=-1.0),
        model("graphatlas_c_q_shuffle_source", certified_routing_mode="q_shuffle_source"),
        model("graphatlas_c_oracle", certified_routing_mode="oracle_max_q"),
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
        model("graphatlas_c_oracle", certified_routing_mode="oracle_max_q"),
    ]
    real = [
        {"name": name, "num_charts": charts, **({"metric": "roc_auc"} if name in {"tolokers"} else {})}
        for name, charts in [("roman_empire", 4), ("tolokers", 4), ("chameleon_filtered", 4), ("squirrel_filtered", 4), ("cornell", 3), ("texas", 3), ("wisconsin", 3)]
    ]
    return {
        "graphatlasc_smoke": {**COMMON, "seeds": [0], "train": {"device": "cpu", "epochs": 3, "patience": 3, "eval_every": 1, "progress": False}, "datasets": [{"name": "atlas_het", "num_nodes": 96, "num_charts": 3, "condition_id": "smoke"}], "models": smoke_models},
        "graphatlasc_phase_screening": {**COMMON, "seeds": [0, 1, 2], "train": {"epochs": 120, "patience": 25}, "datasets": grid, "models": smoke_models},
        "graphatlasc_beta_sweep": {**COMMON, "seeds": [0, 1, 2, 3, 4], "train": {"epochs": 120, "patience": 25}, "datasets": beta_conditions, "models": beta_models},
        "graphatlasc_counterfactual": {**COMMON, "seeds": [0, 1, 2, 3, 4], "train": {"epochs": 120, "patience": 25}, "datasets": beta_conditions, "models": controls},
        "graphatlasc_real_screening": {**COMMON, "seeds": [0, 1, 2], "train": {"epochs": 200, "patience": 40}, "datasets": real, "models": [{"name": "graphatlas", "label": "graphatlas_original"}, {"name": "graphatlas_min_distortion", "label": "graphatlas_min_distortion"}, model("graphatlas_c"), model("graphatlas_c_beta_minus_1", transportability_beta=-1.0)]},
        "graphatlasc_main_confirmatory": {**COMMON, "seeds": list(range(10)), "datasets": real + [{"name": "actor", "num_charts": 4}, {"name": "questions", "num_charts": 4, "metric": "roc_auc"}, {"name": "hetgb", "num_charts": 4, "enabled": False}], "models": [{"name": "graphatlas", "label": "graphatlas_original"}, {"name": "graphatlas_min_distortion", "label": "graphatlas_min_distortion"}, model("graphatlas_c"), model("graphatlas_c_beta_minus_1", transportability_beta=-1.0), model("graphatlas_c_q_shuffle_source", certified_routing_mode="q_shuffle_source"), model("graphatlas_c_oracle", certified_routing_mode="oracle_max_q")]},
        "graphatlasc_ogb_arxiv": {**COMMON, "seeds": [0], "datasets": [{"name": "ogbn_arxiv", "num_charts": 4}], "models": [model("graphatlas_c")]},
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
