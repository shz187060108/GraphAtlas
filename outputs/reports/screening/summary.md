# GraphAtlas experiment summary

| dataset | task | model | test_metric_count | test_metric_mean | test_metric_std | test_metric_ci95 | boundary_accuracy_mean | invariance_error_nonlinear_mean | path_consistency_error_mean | metric_compatibility_error_mean | runtime_seconds_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roman_empire | node_classification | graphatlas_free_transition | 3 | 0.7593 | 0.0106 | 0.0120 |  | 0.5214 | 0.6534 | 1.1750 | 58.7003 |
| roman_empire | node_classification | graphatlas_no_cocycle | 3 | 0.7657 | 0.0105 | 0.0119 |  | 0.0000 | 1.0732 | 1.3289 | 71.9204 |
| roman_empire | node_classification | graphatlas_no_metric | 3 | 0.7645 | 0.0116 | 0.0132 |  | 0.0000 | 0.6867 | 3.5125 | 75.7281 |
| roman_empire | node_classification | graphatlas_no_transport | 3 | 0.7526 | 0.0060 | 0.0068 |  | 0.3923 | 0.5000 | 0.9959 | 51.1556 |

# Mean model ranks

| task | model | mean_rank | median_rank | evaluated_runs |
| --- | --- | --- | --- | --- |
| node_classification | graphatlas_no_cocycle | 1.3333 | 1.0000 | 3 |
| node_classification | graphatlas_no_metric | 1.6667 | 2.0000 | 3 |
| node_classification | graphatlas_free_transition | 3.5000 | 3.5000 | 3 |
| node_classification | graphatlas_no_transport | 3.5000 | 3.5000 | 3 |

# Scientific claim gates

| name | passed | value | criterion |
| --- | --- | --- | --- |
| full_model_present | False | None | At least one Atlas-Het GraphAtlas run exists with intervention metrics |
| invariance_separates_graphatlas_no_transport | None | None | both models must be present |
| invariance_separates_graphatlas_free_transition | None | None | both models must be present |
| boundary_node_gain | None | None | full and no-transport boundary metrics must exist |
| induced_transition_gain | None | None | full and free-transition runs must exist |
