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
