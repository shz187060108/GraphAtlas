from __future__ import annotations

from pathlib import Path

from scripts.download_data import _datasets_from_preset


def test_preset_dataset_discovery_excludes_synthetic():
    root = Path(__file__).resolve().parents[1]
    datasets = _datasets_from_preset("smoke", root)
    assert datasets == []
    paper = _datasets_from_preset("paper", root)
    assert "roman_empire" in paper
    assert "chameleon_filtered" in paper
