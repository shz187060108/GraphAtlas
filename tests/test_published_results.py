from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest
from graphatlas.published_results import load_published_results, validate_published_results, build_comparison_bundle

ROOT=Path(__file__).resolve().parents[1]

def test_curated_published_results_validate():
    frame=load_published_results([ROOT/"published_results/data"])
    assert len(frame) >= 150
    assert {"exact","contextual"}.issubset(set(frame.comparison_level))
    assert frame.mean_fraction.between(0,1).all()

def test_invalid_unit_is_rejected():
    row={c:"x" for c in ["source_id","paper_title","venue","table_id","source_url","dataset_version","task","metric","feature_condition","split_protocol","model","comparison_level","notes"]}
    row.update(year=2025,dataset="cora",mean=.5,std=.1,unit="bad",n_runs=3,comparison_level="exact")
    with pytest.raises(ValueError): validate_published_results(pd.DataFrame([row]))

def test_comparison_bundle_keeps_contextual_separate(tmp_path):
    results=tmp_path/"results.csv"
    pd.DataFrame([{"dataset":"roman_empire","task":"node_classification","model":"graphatlas","metric_name":"accuracy","test_metric":.8,"seed":0}]).to_csv(results,index=False)
    outputs=build_comparison_bundle(results,[ROOT/"published_results/data"],ROOT/"configs/comparison_protocols.yaml",tmp_path/"out")
    exact=pd.read_csv(outputs["exact"]); contextual=pd.read_csv(outputs["contextual"])
    assert "delta_vs_published_mean" in exact
    assert not contextual.empty
    audit=Path(outputs["audit"]).read_text()
    assert "never used for paired significance" in audit
