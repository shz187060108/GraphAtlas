from __future__ import annotations

import pandas as pd

from graphatlas.figure_data import load_results_frame
from graphatlas.reporting import model_ranks


def test_formal_q_reference_is_kept_but_test_selected_is_removed(tmp_path):
    frame = pd.DataFrame([
        {"dataset": "actor", "model": "graphatlas_c_oracle", "test_metric": .8, "selection_protocol": "validation_selected"},
        {"dataset": "actor", "model": "graphatlas_c", "test_metric": .7, "selection_protocol": "test_selected", "result_source": "best_config_search"},
    ])
    path = tmp_path / "results.csv"; frame.to_csv(path, index=False)
    loaded = load_results_frame(path)
    assert loaded["model"].tolist() == ["graphatlas_c_oracle"]
    assert not model_ranks(pd.read_csv(path)).empty


def test_best_config_search_summary_uses_validation_selected_metric(tmp_path):
    path = tmp_path / "search_summary.csv"
    pd.DataFrame([
        {
            "dataset": "actor",
            "metric": "accuracy",
            "test_selected_metric": 0.90,
            "validation_selected_val_metric": 0.70,
            "validation_selected_test_metric": 0.65,
        }
    ]).to_csv(path, index=False)

    loaded = load_results_frame(path)
    assert loaded["model"].tolist() == ["graphatlas_c_oracle"]
    assert loaded["selection_protocol"].tolist() == ["validation_selected"]
    assert loaded["is_formal_result"].tolist() == [True]
    assert loaded["test_metric"].tolist() == [0.70]
    assert loaded["display_metric_source"].tolist() == ["validation_selected_val_metric"]
    assert loaded["test_selected_metric"].tolist() == [0.90]
