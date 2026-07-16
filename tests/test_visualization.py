from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.visualization
def test_visualization_writes_svg_figures(tmp_path):
    rows: list[dict[str, object]] = []
    for seed in [0, 1]:
        rows.extend(
            [
                {
                    "dataset": "atlas_het",
                    "task": "node_classification",
                    "model": "graphatlas",
                    "seed": seed,
                    "split": 0,
                    "test_metric": 0.72 + 0.01 * seed,
                    "boundary_accuracy": 0.74 + 0.01 * seed,
                    "invariance_error_nonlinear": 1e-10,
                },
                {
                    "dataset": "atlas_het",
                    "task": "node_classification",
                    "model": "graphatlas_no_transport",
                    "seed": seed,
                    "split": 0,
                    "test_metric": 0.66,
                    "boundary_accuracy": 0.68,
                    "invariance_error_nonlinear": 1e-4,
                },
                {
                    "dataset": "atlas_het",
                    "task": "node_classification",
                    "model": "graphatlas_free_transition",
                    "seed": seed,
                    "split": 0,
                    "test_metric": 0.67,
                    "boundary_accuracy": 0.69,
                    "invariance_error_nonlinear": 1e-4,
                },
            ]
        )
    results = tmp_path / "results.csv"
    figures = tmp_path / "figures"
    with results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(root / "scripts" / "visualize.py"),
            "--results",
            str(results),
            "--output-dir",
            str(figures),
        ],
        cwd=root,
        env=environment,
        check=False,
        timeout=90,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (figures / "mechanism_diagnostics.svg").exists()
    assert (figures / "boundary_node_pairs.svg").exists()
    assert (figures / "transition_pairs.svg").exists()
    assert (figures / "graphatlas_mechanism_group.svg").exists()
    parsed = json.loads((figures / "figure_manifest.json").read_text())
    assert parsed["mechanism_diagnostics"]["generated"]
    assert parsed["mechanism_diagnostics"]["no_transport_to_full_ratio"] >= 10
