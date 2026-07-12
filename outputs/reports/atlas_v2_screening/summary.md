# GraphAtlas experiment summary

| dataset | task | model | test_metric_count | test_metric_mean | test_metric_std | test_metric_ci95 | boundary_accuracy_mean | invariance_error_nonlinear_mean | path_consistency_error_mean | metric_compatibility_error_mean | runtime_seconds_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| atlas_het_boundary_stress | node_classification | ambient_vector_gnn | 3 | 0.7634 | 0.0162 | 0.0183 | 0.7641 |  |  |  | 35.6997 |
| atlas_het_boundary_stress | node_classification | geometry_moe | 3 | 0.7145 | 0.0372 | 0.0421 | 0.7187 |  |  |  | 15.9867 |
| atlas_het_boundary_stress | node_classification | graphatlas | 3 | 0.7733 | 0.0145 | 0.0164 | 0.7696 | 0.0000 | 0.0861 | 0.3467 | 298.2316 |
| atlas_het_boundary_stress | node_classification | graphatlas_free_transition | 3 | 0.7634 | 0.0245 | 0.0278 | 0.7586 | 0.1949 | 0.0438 | 0.1644 | 120.7482 |
| atlas_het_boundary_stress | node_classification | graphatlas_no_cocycle | 3 | 0.7504 | 0.0007 | 0.0008 | 0.7513 | 0.0000 | 0.4667 | 0.3276 | 70.0187 |
| atlas_het_boundary_stress | node_classification | graphatlas_no_metric | 3 | 0.7324 | 0.0274 | 0.0310 | 0.7258 | 0.0000 | 0.1836 | 1.4908 | 70.5358 |
| atlas_het_boundary_stress | node_classification | graphatlas_no_overlap | 3 | 0.7586 | 0.0515 | 0.0583 | 0.7475 | 0.0000 | 0.0000 | 0.0021 | 412.2697 |
| atlas_het_boundary_stress | node_classification | graphatlas_no_rank | 3 | 0.7781 | 0.0100 | 0.0114 | 0.7750 | 0.0000 | 0.0774 | 0.3185 | 96.0959 |
| atlas_het_boundary_stress | node_classification | graphatlas_no_transport | 3 | 0.7259 | 0.0251 | 0.0284 | 0.7222 | 0.1351 | 0.0403 | 0.1728 | 343.1485 |
| atlas_het_boundary_stress | node_classification | signature_gnn | 3 | 0.7291 | 0.0448 | 0.0507 | 0.7331 |  |  |  | 2.2897 |
| atlas_het_coordinate | node_classification | ambient_vector_gnn | 3 | 0.8422 | 0.0124 | 0.0140 | 0.8399 |  |  |  | 28.6112 |
| atlas_het_coordinate | node_classification | geometry_moe | 3 | 0.8279 | 0.0323 | 0.0365 | 0.8298 |  |  |  | 18.4632 |
| atlas_het_coordinate | node_classification | graphatlas | 3 | 0.8197 | 0.0371 | 0.0420 | 0.8164 | 0.0000 | 0.0539 | 0.2522 | 39.3693 |
| atlas_het_coordinate | node_classification | graphatlas_free_transition | 3 | 0.8484 | 0.0229 | 0.0260 | 0.8386 | 0.0393 | 0.0522 | 0.1490 | 70.0869 |
| atlas_het_coordinate | node_classification | graphatlas_no_cocycle | 3 | 0.8238 | 0.0441 | 0.0500 | 0.8314 | 0.0000 | 0.2457 | 0.1704 | 17.9476 |
| atlas_het_coordinate | node_classification | graphatlas_no_metric | 3 | 0.8341 | 0.0338 | 0.0382 | 0.8261 | 0.0000 | 0.0910 | 1.4405 | 13.4527 |
| atlas_het_coordinate | node_classification | graphatlas_no_overlap | 3 | 0.8320 | 0.0243 | 0.0275 | 0.8273 | 0.0000 | 0.0378 | 0.2456 | 51.1479 |
| atlas_het_coordinate | node_classification | graphatlas_no_rank | 3 | 0.8238 | 0.0391 | 0.0443 | 0.8131 | 0.0000 | 0.0471 | 0.2431 | 71.1541 |
| atlas_het_coordinate | node_classification | graphatlas_no_transport | 3 | 0.8443 | 0.0279 | 0.0316 | 0.8323 | 0.0596 | 0.0175 | 0.1002 | 113.2946 |
| atlas_het_coordinate | node_classification | signature_gnn | 3 | 0.8320 | 0.0395 | 0.0447 | 0.8191 |  |  |  | 10.3194 |
| atlas_het_mixed_metric | node_classification | ambient_vector_gnn | 3 | 0.6553 | 0.0261 | 0.0295 | 0.7355 |  |  |  | 14.7032 |
| atlas_het_mixed_metric | node_classification | geometry_moe | 3 | 0.6634 | 0.0032 | 0.0037 | 0.6946 |  |  |  | 27.3988 |
| atlas_het_mixed_metric | node_classification | graphatlas | 3 | 0.7024 | 0.0043 | 0.0049 | 0.7718 | 0.0000 | 0.0713 | 0.3179 | 167.0705 |
| atlas_het_mixed_metric | node_classification | graphatlas_free_transition | 3 | 0.6765 | 0.0395 | 0.0447 | 0.7665 | 0.1895 | 0.0420 | 0.1560 | 114.6312 |
| atlas_het_mixed_metric | node_classification | graphatlas_no_cocycle | 3 | 0.6813 | 0.0227 | 0.0257 | 0.7389 | 0.0000 | 0.4000 | 0.3025 | 31.1774 |
| atlas_het_mixed_metric | node_classification | graphatlas_no_metric | 3 | 0.7267 | 0.0306 | 0.0346 | 0.8098 | 0.0000 | 0.1586 | 1.5113 | 120.3675 |
| atlas_het_mixed_metric | node_classification | graphatlas_no_overlap | 3 | 0.7024 | 0.0043 | 0.0049 | 0.7257 | 0.0000 | 0.0184 | 0.1257 | 177.7483 |
| atlas_het_mixed_metric | node_classification | graphatlas_no_rank | 3 | 0.7057 | 0.0017 | 0.0020 | 0.7492 | 0.0000 | 0.0778 | 0.3112 | 87.4364 |
| atlas_het_mixed_metric | node_classification | graphatlas_no_transport | 3 | 0.6813 | 0.0226 | 0.0256 | 0.7244 | 0.1563 | 0.0376 | 0.1688 | 163.0793 |
| atlas_het_mixed_metric | node_classification | signature_gnn | 3 | 0.7024 | 0.0154 | 0.0174 | 0.8094 |  |  |  | 13.4002 |

# Paired tests against GraphAtlas

| dataset | task | baseline | n_pairs | mean_delta | median_delta | wins | ties | losses | paired_t_p | wilcoxon_p | paired_effect_d | wilcoxon_holm_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| atlas_het_boundary_stress | node_classification | ambient_vector_gnn | 3 | 0.0098 | 0.0196 | 2 | 0 | 1 | 0.5085 | 0.5000 | 0.4608 | 1.0000 |
| atlas_het_boundary_stress | node_classification | geometry_moe | 3 | 0.0588 | 0.0539 | 3 | 0 | 0 | 0.1109 | 0.2500 | 1.5860 | 1.0000 |
| atlas_het_boundary_stress | node_classification | graphatlas_free_transition | 3 | 0.0098 | 0.0147 | 2 | 0 | 1 | 0.5956 | 0.7500 | 0.3611 | 1.0000 |
| atlas_het_boundary_stress | node_classification | graphatlas_no_cocycle | 3 | 0.0229 | 0.0196 | 3 | 0 | 0 | 0.1185 | 0.2500 | 1.5243 | 1.0000 |
| atlas_het_boundary_stress | node_classification | graphatlas_no_metric | 3 | 0.0408 | 0.0245 | 3 | 0 | 0 | 0.2320 | 0.2500 | 0.9790 | 1.0000 |
| atlas_het_boundary_stress | node_classification | graphatlas_no_overlap | 3 | 0.0147 | 0.0000 | 1 | 1 | 1 | 0.7234 | 1.0000 | 0.2350 | 1.0000 |
| atlas_het_boundary_stress | node_classification | graphatlas_no_rank | 3 | -0.0049 | 0.0000 | 0 | 2 | 1 | 0.4226 | 1.0000 | -0.5774 | 1.0000 |
| atlas_het_boundary_stress | node_classification | graphatlas_no_transport | 3 | 0.0474 | 0.0637 | 3 | 0 | 0 | 0.1286 | 0.2500 | 1.4501 | 1.0000 |
| atlas_het_boundary_stress | node_classification | signature_gnn | 3 | 0.0441 | 0.0637 | 2 | 0 | 1 | 0.2782 | 0.5000 | 0.8515 | 1.0000 |
| atlas_het_coordinate | node_classification | ambient_vector_gnn | 3 | -0.0225 | 0.0000 | 0 | 2 | 1 | 0.4226 | 1.0000 | -0.5774 | 1.0000 |
| atlas_het_coordinate | node_classification | geometry_moe | 3 | -0.0082 | -0.0123 | 1 | 0 | 2 | 0.5305 | 0.7500 | -0.4341 | 1.0000 |
| atlas_het_coordinate | node_classification | graphatlas_free_transition | 3 | -0.0287 | -0.0307 | 0 | 0 | 3 | 0.0843 | 0.2500 | -1.8608 | 1.0000 |
| atlas_het_coordinate | node_classification | graphatlas_no_cocycle | 3 | -0.0041 | 0.0062 | 2 | 0 | 1 | 0.9234 | 1.0000 | -0.0627 | 1.0000 |
| atlas_het_coordinate | node_classification | graphatlas_no_metric | 3 | -0.0143 | -0.0123 | 0 | 0 | 3 | 0.0195 | 0.2500 | -4.0735 | 1.0000 |
| atlas_het_coordinate | node_classification | graphatlas_no_overlap | 3 | -0.0123 | -0.0062 | 0 | 1 | 2 | 0.3201 | 0.5000 | -0.7570 | 1.0000 |
| atlas_het_coordinate | node_classification | graphatlas_no_rank | 3 | -0.0041 | 0.0000 | 0 | 2 | 1 | 0.4226 | 1.0000 | -0.5774 | 1.0000 |
| atlas_het_coordinate | node_classification | graphatlas_no_transport | 3 | -0.0246 | -0.0247 | 0 | 1 | 2 | 0.2248 | 0.5000 | -1.0021 | 1.0000 |
| atlas_het_coordinate | node_classification | signature_gnn | 3 | -0.0123 | -0.0123 | 0 | 0 | 3 | 0.0750 | 0.2500 | -1.9877 | 1.0000 |
| atlas_het_mixed_metric | node_classification | ambient_vector_gnn | 3 | 0.0471 | 0.0539 | 3 | 0 | 0 | 0.1107 | 0.2500 | 1.5877 | 1.0000 |
| atlas_het_mixed_metric | node_classification | geometry_moe | 3 | 0.0390 | 0.0392 | 3 | 0 | 0 | 0.0050 | 0.2500 | 8.1718 | 1.0000 |
| atlas_het_mixed_metric | node_classification | graphatlas_free_transition | 3 | 0.0259 | 0.0098 | 2 | 0 | 1 | 0.3904 | 0.5000 | 0.6278 | 1.0000 |
| atlas_het_mixed_metric | node_classification | graphatlas_no_cocycle | 3 | 0.0212 | 0.0245 | 2 | 1 | 0 | 0.2040 | 0.5000 | 1.0737 | 1.0000 |
| atlas_het_mixed_metric | node_classification | graphatlas_no_metric | 3 | -0.0243 | -0.0293 | 1 | 0 | 2 | 0.3179 | 0.5000 | -0.7615 | 1.0000 |
| atlas_het_mixed_metric | node_classification | graphatlas_no_overlap | 3 | 0.0000 | 0.0049 | 2 | 0 | 1 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| atlas_het_mixed_metric | node_classification | graphatlas_no_rank | 3 | -0.0033 | 0.0000 | 0 | 2 | 1 | 0.4226 | 1.0000 | -0.5774 | 1.0000 |
| atlas_het_mixed_metric | node_classification | graphatlas_no_transport | 3 | 0.0211 | 0.0340 | 2 | 0 | 1 | 0.3063 | 0.5000 | 0.7864 | 1.0000 |
| atlas_het_mixed_metric | node_classification | signature_gnn | 3 | 0.0000 | 0.0049 | 2 | 0 | 1 | 0.9970 | 1.0000 | 0.0025 | 1.0000 |

# Mean model ranks

| task | model | mean_rank | median_rank | evaluated_runs |
| --- | --- | --- | --- | --- |
| node_classification | graphatlas_no_rank | 3.9444 | 3.0000 | 9 |
| node_classification | graphatlas_free_transition | 4.3889 | 4.0000 | 9 |
| node_classification | graphatlas_no_metric | 4.8889 | 5.5000 | 9 |
| node_classification | graphatlas | 5.0000 | 5.5000 | 9 |
| node_classification | graphatlas_no_overlap | 5.0556 | 5.0000 | 9 |
| node_classification | signature_gnn | 5.1667 | 5.0000 | 9 |
| node_classification | graphatlas_no_transport | 5.9444 | 7.5000 | 9 |
| node_classification | ambient_vector_gnn | 6.2778 | 7.0000 | 9 |
| node_classification | graphatlas_no_cocycle | 6.4444 | 7.0000 | 9 |
| node_classification | geometry_moe | 7.8889 | 8.5000 | 9 |

# Scientific claim gates

| name | passed | value | criterion |
| --- | --- | --- | --- |
| nonlinear_coordinate_invariance | True | 0.0000 | maximum nonlinear intervention error < 1e-6 |
| invariance_separates_graphatlas_no_transport | True | 1884641.3194 | ablation error / full error >= 10 |
| invariance_separates_graphatlas_free_transition | True | 2274661.8988 | ablation error / full error >= 10 |
| boundary_node_gain | True | 0.0263 | mean paired gain >= 0.020 |
| induced_transition_gain | True | 0.0024 | mean paired test-metric gain > 0 |
