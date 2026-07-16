from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def test_generated_grid_counts():
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "scripts/generate_graphatlasc_presets.py", "--check"], cwd=root, check=True)
    for name, expected in (("graphatlasc_phase_screening", 336), ("graphatlasc_beta_sweep", 105), ("graphatlasc_counterfactual", 90)):
        payload = yaml.safe_load((root / "configs" / "presets" / f"{name}.yaml").read_text(encoding="utf-8"))
        assert len(payload["datasets"]) * len(payload["models"]) * len(payload["seeds"]) == expected
