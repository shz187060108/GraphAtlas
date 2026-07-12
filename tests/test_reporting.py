from __future__ import annotations

import pandas as pd

from graphatlas.reporting import scientific_gates, summarize_results


def test_reporting_writes_claim_gates(tmp_path):
    rows = []
    for seed in [0, 1]:
        rows.extend(
            [
                {"dataset": "atlas_het", "model": "graphatlas", "seed": seed, "split": 0, "test_metric": 0.7, "boundary_accuracy": 0.75, "invariance_error_nonlinear": 1e-10},
                {"dataset": "atlas_het", "model": "graphatlas_no_transport", "seed": seed, "split": 0, "test_metric": 0.65, "boundary_accuracy": 0.70, "invariance_error_nonlinear": 1e-4},
                {"dataset": "atlas_het", "model": "graphatlas_free_transition", "seed": seed, "split": 0, "test_metric": 0.66, "boundary_accuracy": 0.71, "invariance_error_nonlinear": 1e-4},
            ]
        )
    path = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    _, _, gates = summarize_results(path, plots=False)
    assert gates["all_decisive_gates_pass"]
    assert (tmp_path / "scientific_gates.json").exists()
    assert scientific_gates(pd.DataFrame(rows))["passed"] >= 4
