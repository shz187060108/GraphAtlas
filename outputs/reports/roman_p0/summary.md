# GraphAtlas experiment summary

| dataset | task | model | test_metric_count | test_metric_mean | test_metric_std | test_metric_ci95 | boundary_accuracy_mean | invariance_error_nonlinear_mean | path_consistency_error_mean | metric_compatibility_error_mean | runtime_seconds_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roman_empire | node_classification | ambient_vector_gnn | 1 | 0.7677 |  |  |  |  |  |  | 210.7964 |

# Mean model ranks

| task | model | mean_rank | median_rank | evaluated_runs |
| --- | --- | --- | --- | --- |
| node_classification | ambient_vector_gnn | 1.0000 | 1.0000 | 1 |

# Scientific claim gates

| name | passed | value | criterion |
| --- | --- | --- | --- |
| full_model_present | False | None | At least one Atlas-Het GraphAtlas run exists with intervention metrics |
| invariance_separates_graphatlas_no_transport | None | None | both models must be present |
| invariance_separates_graphatlas_free_transition | None | None | both models must be present |
| boundary_node_gain | None | None | full and no-transport boundary metrics must exist |
| induced_transition_gain | None | None | full and free-transition runs must exist |
