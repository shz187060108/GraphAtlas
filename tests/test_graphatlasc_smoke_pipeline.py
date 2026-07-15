from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_build_paper_figures_cli_gracefully_handles_smoke_results(tmp_path):
    root = Path(__file__).resolve().parents[1]
    results = tmp_path / "smoke.csv"
    pd.DataFrame([{
        "dataset": "atlas_het", "task": "node_classification", "model": "graphatlas_c",
        "seed": 0, "split": 0, "test_metric": .7, "runtime_seconds": 1., "parameters": 10,
        "transportability_mean": .8, "distortion_mean": .2,
    }]).to_csv(results, index=False)
    output = tmp_path / "figures"
    completed = subprocess.run([sys.executable, "scripts/build_paper_figures.py", "--results", str(results), "--output-dir", str(output)], cwd=root, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert (output / "manifests" / "figure_manifest.json").exists()
    payload = json.loads((output / "manifests" / "figure_manifest.json").read_text(encoding="utf-8"))
    assert payload["svg_only"]
