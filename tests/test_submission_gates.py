from __future__ import annotations

import pandas as pd

from graphatlas.reporting import submission_readiness_gates, summarize_results


def passing_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    synthetic_datasets = ("atlas_het_v2_boundary", "atlas_het_v2_overlap")
    synthetic_models = (
        "graphatlas", "ambient_vector_gnn", "geometry_moe",
        "graphatlas_no_cocycle", "graphatlas_no_metric", "graphatlas_free_transition",
    )
    for dataset in synthetic_datasets:
        for seed in range(10):
            for model in synthetic_models:
                full = model == "graphatlas"
                rows.append({
                    "run_id": f"{dataset}-{model}-{seed}",
                    "dataset": dataset,
                    "task": "node_classification",
                    "model": model,
                    "seed": seed,
                    "split": seed,
                    "test_metric": 0.90 if full else 0.80,
                    "boundary_accuracy": 0.82 if full else 0.76,
                    "interior_accuracy": 0.80 if full else (0.805 if model == "ambient_vector_gnn" else 0.79),
                    "path_consistency_error": 0.01 if full else 0.05,
                    "metric_compatibility_error": 0.01 if full else 0.05,
                    "chart_ari": 0.60 if full else 0.10,
                })
    for dataset_index in range(4):
        dataset = f"real_{dataset_index}"
        for seed in range(10):
            for model, score in (("graphatlas", 0.90), ("graphatlas_no_metric", 0.80), ("graphatlas_no_cocycle", 0.79)):
                rows.append({
                    "run_id": f"{dataset}-{model}-{seed}", "dataset": dataset,
                    "task": "node_classification", "model": model, "seed": seed,
                    "split": seed, "test_metric": score,
                })
    for model in ("graphmore_official", "geomoe_official", "argnn_official"):
        rows.append({
            "run_id": f"external-{model}", "dataset": "real_0", "task": "node_classification",
            "model": model, "seed": 0, "split": 0, "test_metric": 0.75,
        })
    return rows


def test_submission_readiness_all_seven_groups_pass(tmp_path):
    frame = pd.DataFrame(passing_rows())
    readiness = submission_readiness_gates(frame)
    assert readiness["all_submission_gates_pass"]
    assert readiness["passed"] == 7

    path = tmp_path / "results.csv"
    frame.to_csv(path, index=False)
    summarize_results(path, plots=False)
    assert (tmp_path / "submission_readiness.json").exists()
    assert (tmp_path / "submission_readiness.md").exists()


def test_submission_readiness_missing_evidence_fails_all_groups():
    frame = pd.DataFrame([
        {"run_id": "one", "dataset": "atlas_het_v2_one", "task": "node_classification",
         "model": "graphatlas", "seed": 0, "split": 0, "test_metric": 0.5}
    ])
    readiness = submission_readiness_gates(frame)
    assert not readiness["all_submission_gates_pass"]
    assert readiness["failed"] == 7
