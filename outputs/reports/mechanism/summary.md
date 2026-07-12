# GraphAtlas experiment summary

| dataset | task | model | test_metric_count | test_metric_mean | test_metric_std | test_metric_ci95 | boundary_accuracy_mean | invariance_error_nonlinear_mean | path_consistency_error_mean | metric_compatibility_error_mean | runtime_seconds_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| atlas_het | node_classification | graphatlas | 3 | 0.7387 | 0.1133 | 0.1282 | 0.7366 | 0.0000 | 0.3476 | 0.8163 | 13.2971 |
| atlas_het | node_classification | graphatlas_free_transition | 3 | 0.7252 | 0.0563 | 0.0637 | 0.6754 | 0.1432 | 0.1486 | 0.4516 | 11.6567 |
| atlas_het | node_classification | graphatlas_no_transport | 3 | 0.7027 | 0.1523 | 0.1723 | 0.6940 | 0.0662 | 0.1456 | 0.6531 | 10.2739 |

# Paired tests against GraphAtlas

| dataset | task | baseline | n_pairs | mean_delta | median_delta | wins | ties | losses | paired_t_p | wilcoxon_p | paired_effect_d | wilcoxon_holm_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| atlas_het | node_classification | graphatlas_free_transition | 3 | 0.0135 | 0.0270 | 2 | 0 | 1 | 0.7418 | 0.7500 | 0.2182 | 0.7500 |
| atlas_het | node_classification | graphatlas_no_transport | 3 | 0.0360 | 0.0135 | 3 | 0 | 0 | 0.2507 | 0.2500 | 0.9238 | 0.5000 |

# Mean model ranks

| task | model | mean_rank | median_rank | evaluated_runs |
| --- | --- | --- | --- | --- |
| node_classification | graphatlas | 1.3333 | 1.0000 | 3 |
| node_classification | graphatlas_free_transition | 2.3333 | 3.0000 | 3 |
| node_classification | graphatlas_no_transport | 2.3333 | 2.0000 | 3 |

# Scientific claim gates

| name | passed | value | criterion |
| --- | --- | --- | --- |
| nonlinear_coordinate_invariance | True | 0.0000 | maximum nonlinear intervention error < 1e-6 |
| invariance_separates_graphatlas_no_transport | True | 988205.6110 | ablation error / full error >= 10 |
| invariance_separates_graphatlas_free_transition | True | 2135775.2426 | ablation error / full error >= 10 |
| boundary_node_gain | True | 0.0426 | mean paired gain >= 0.020 |
| induced_transition_gain | True | 0.0135 | mean paired test-metric gain > 0 |
