# Published-result comparison

Only rows with matching dataset version, split protocol, features and metric appear below.
Published aggregates are not used for paired tests.

| dataset              | metric   | model_graphatlas   |   graphatlas_mean |   graphatlas_std |   model_published |   mean_fraction |   std_fraction |   delta_vs_published_mean |   source_id |   table_id |
|:---------------------|:---------|:-------------------|------------------:|-----------------:|------------------:|----------------:|---------------:|--------------------------:|------------:|-----------:|
| atlas_het_coordinate | accuracy | ambient_vector_gnn |          0.846626 |              nan |               nan |             nan |            nan |                       nan |         nan |        nan |

## Contextual references

These rows are useful for orientation but are not apples-to-apples comparisons.

| dataset         | metric   | model      |   mean_fraction |   std_fraction | feature_condition   | split_protocol   | source_id         | table_id   |
|:----------------|:---------|:-----------|----------------:|---------------:|:--------------------|:-----------------|:------------------|:-----------|
| hetgb_cornell   | accuracy | GCN        |          0.5143 |          0.049 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | GCN        |          0.5409 |          0.017 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | GCN        |          0.5368 |          0.034 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | GCN        |          0.6333 |          0.013 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | GCN        |          0.4397 |          0.003 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | GraphSAGE  |          0.7143 |          0.019 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | GraphSAGE  |          0.6818 |          0.002 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | GraphSAGE  |          0.7135 |          0.041 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | GraphSAGE  |          0.7109 |          0.004 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | GraphSAGE  |          0.4798 |          0.003 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | GAT        |          0.481  |          0.031 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | GAT        |          0.5727 |          0.068 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | GAT        |          0.5579 |          0.069 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | GAT        |          0.4623 |          0.026 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | GAT        |          0.4332 |          0.001 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | H2GCN      |          0.6571 |          0.019 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | H2GCN      |          0.7909 |          0.027 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | H2GCN      |          0.793  |          0.013 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | H2GCN      |          0.7386 |          0.004 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | H2GCN      |          0.5046 |          0.003 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | FAGCN      |          0.7333 |          0.028 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | FAGCN      |          0.825  |          0.05  | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | FAGCN      |          0.8035 |          0.021 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | FAGCN      |          0.7379 |          0.006 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | FAGCN      |          0.475  |          0.004 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | JacobiConv |          0.7164 |          0.013 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | JacobiConv |          0.7341 |          0.049 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | JacobiConv |          0.742  |          0.054 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | JacobiConv |          0.7244 |          0.012 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | JacobiConv |          0.4877 |          0.002 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | GBK-GNN    |          0.6762 |          0.024 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | GBK-GNN    |          0.7955 |          0.025 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | GBK-GNN    |          0.7193 |          0.022 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | GBK-GNN    |          0.721  |          0.006 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | GBK-GNN    |          0.4721 |          0.006 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | OGNN       |          0.6667 |          0.03  | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | OGNN       |          0.7591 |          0.023 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | OGNN       |          0.7754 |          0.023 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | OGNN       |          0.7354 |          0.013 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | OGNN       |          0.5024 |          0.016 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | SEGSL      |          0.7333 |          0.039 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | SEGSL      |          0.8    |          0.019 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | SEGSL      |          0.8105 |          0.014 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | SEGSL      |          0.7136 |          0.003 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | SEGSL      |          0.4669 |          0.004 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | DisamGCL   |          0.5667 |          0.031 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | DisamGCL   |          0.65   |          0.012 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | DisamGCL   |          0.586  |          0.02  | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | DisamGCL   |          0.706  |          0.002 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | DisamGCL   |          0.4437 |          0.001 | shallow_features    | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | GCN        |          0.5286 |          0.018 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | GCN        |          0.4364 |          0.033 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | GCN        |          0.414  |          0.018 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | GCN        |          0.667  |          0.013 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | GCN        |          0.3933 |          0.01  | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | GraphSAGE  |          0.7571 |          0.018 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | GraphSAGE  |          0.8182 |          0.025 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | GraphSAGE  |          0.8035 |          0.013 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | GraphSAGE  |          0.7037 |          0.001 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | GraphSAGE  |          0.4663 |          0.001 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | GAT        |          0.5428 |          0.051 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | GAT        |          0.5136 |          0.023 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | GAT        |          0.5053 |          0.017 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | GAT        |          0.6374 |          0.067 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | GAT        |          0.3512 |          0.064 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | H2GCN      |          0.6976 |          0.03  | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | H2GCN      |          0.7909 |          0.035 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | H2GCN      |          0.8018 |          0.019 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | H2GCN      |          0.7073 |          0.009 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | H2GCN      |          0.4709 |          0.003 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | FAGCN      |          0.7643 |          0.031 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | FAGCN      |          0.8455 |          0.048 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | FAGCN      |          0.8316 |          0.014 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | FAGCN      |          0.7558 |          0.005 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | FAGCN      |          0.4983 |          0.006 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | JacobiConv |          0.7357 |          0.043 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | JacobiConv |          0.818  |          0.041 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | JacobiConv |          0.7631 |          0.013 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | JacobiConv |          0.7381 |          0.003 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | JacobiConv |          0.4943 |          0.005 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | GBK-GNN    |          0.6619 |          0.028 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | GBK-GNN    |          0.8    |          0.03  | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | GBK-GNN    |          0.7298 |          0.033 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | GBK-GNN    |          0.7249 |          0.01  | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | GBK-GNN    |          0.449  |          0.003 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | OGNN       |          0.7191 |          0.018 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | OGNN       |          0.85   |          0.023 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | OGNN       |          0.793  |          0.021 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | OGNN       |          0.7208 |          0.024 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | OGNN       |          0.4779 |          0.016 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | SEGSL      |          0.6667 |          0.041 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | SEGSL      |          0.85   |          0.02  | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | SEGSL      |          0.793  |          0.018 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | SEGSL      |          0.7273 |          0.008 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | SEGSL      |          0.4738 |          0.002 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_cornell   | accuracy | DisamGCL   |          0.5048 |          0.02  | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_texas     | accuracy | DisamGCL   |          0.65   |          0.012 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_wisconsin | accuracy | DisamGCL   |          0.5789 |          0     | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_actor     | accuracy | DisamGCL   |          0.6778 |          0.003 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
| hetgb_amazon    | accuracy | DisamGCL   |          0.439  |          0.004 | llm_features        | hetgb_paper      | hetgb_2025_table2 | Table 2    |
