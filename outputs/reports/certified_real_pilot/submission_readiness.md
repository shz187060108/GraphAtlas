# Submission readiness gates

| name | passed | value | criterion | claim_policy |
| --- | --- | --- | --- | --- |
| synthetic_significant_gain | False | {'passing_datasets': [], 'comparisons': []} | at least two Atlas-Het v2 datasets beat Ambient and GeometryMoE with paired significance |  |
| boundary_gain | False | [] | at least one boundary-stress dataset gains >= 0.03 over both matched baselines |  |
| interior_preservation | False | {'minimum_delta': None, 'n_pairs': 0} | full interior accuracy is no more than 0.01 below the best matched baseline |  |
| consistency_superiority | False | [] | full improves path and metric consistency against corresponding ablations |  |
| chart_recovery | False | {} | at least one Atlas-Het v2 dataset has mean chart ARI >= 0.50 | If this gate fails, remove chart identifiability and chart recovery claims from the paper. |
| four_real_datasets | False | {'datasets': 4, 'mean_rank': nan, 'wins': {'graphatlas_no_metric': 0, 'graphatlas_no_cocycle': 0}} | four real datasets, mean rank <= 2.0, and majority wins over both regularizer ablations |  |
| direct_external_comparisons | False | [] | at least three official external models share exact dataset/seed/split combinations |  |
