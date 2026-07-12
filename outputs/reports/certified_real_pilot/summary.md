# GraphAtlas experiment summary

| dataset | task | model | test_metric_count | test_metric_mean | test_metric_std | test_metric_ci95 | boundary_accuracy_mean | invariance_error_nonlinear_mean | path_consistency_error_mean | metric_compatibility_error_mean | runtime_seconds_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chameleon_filtered | node_classification | certified_beta_1 | 1 | 0.4175 |  |  |  | 0.0000 | 0.1789 | 0.2963 | 29.1485 |
| chameleon_filtered | node_classification | min_distortion | 1 | 0.4175 |  |  |  | 0.0000 | 0.1779 | 0.2940 | 76.7519 |
| chameleon_filtered | node_classification | original | 1 | 0.3866 |  |  |  | 0.0000 | 0.0707 | 0.2479 | 19.2257 |
| cornell | node_classification | certified_beta_1 | 1 | 0.6757 |  |  |  | 0.0000 | 0.0492 | 0.1077 | 9.7358 |
| cornell | node_classification | min_distortion | 1 | 0.6216 |  |  |  | 0.0000 | 0.0562 | 0.1212 | 9.6976 |
| cornell | node_classification | original | 1 | 0.7297 |  |  |  | 0.0000 | 0.0142 | 0.1720 | 13.2993 |
| texas | node_classification | certified_beta_1 | 1 | 0.5405 |  |  |  | 0.0000 | 0.0418 | 0.0614 | 31.9001 |
| texas | node_classification | min_distortion | 1 | 0.5135 |  |  |  | 0.0000 | 0.0408 | 0.0630 | 60.3029 |
| texas | node_classification | original | 1 | 0.5676 |  |  |  | 0.0000 | 0.0000 | 0.0011 | 14.8325 |
| wisconsin | node_classification | certified_beta_1 | 1 | 0.7059 |  |  |  | 0.0000 | 0.1473 | 0.1743 | 30.8299 |
| wisconsin | node_classification | min_distortion | 1 | 0.7255 |  |  |  | 0.0000 | 0.1397 | 0.1294 | 61.6350 |
| wisconsin | node_classification | original | 1 | 0.7843 |  |  |  | 0.0000 | 0.0032 | 0.0797 | 85.9591 |

# Mean model ranks

| task | model | mean_rank | median_rank | evaluated_runs |
| --- | --- | --- | --- | --- |
| node_classification | original | 1.5000 | 1.0000 | 4 |
| node_classification | certified_beta_1 | 2.1250 | 2.0000 | 4 |
| node_classification | min_distortion | 2.3750 | 2.5000 | 4 |

# Scientific claim gates

| name | passed | value | criterion |
| --- | --- | --- | --- |
| full_model_present | False | None | At least one Atlas-Het GraphAtlas run exists with intervention metrics |
| invariance_separates_graphatlas_no_transport | None | None | both models must be present |
| invariance_separates_graphatlas_free_transition | None | None | both models must be present |
| boundary_node_gain | None | None | full and no-transport boundary metrics must exist |
| induced_transition_gain | None | None | full and free-transition runs must exist |
