# GraphAtlas experiment protocol

## Separation of exploratory and confirmatory evidence

The benchmark is intentionally split into two stages.

1. `screening.yaml` runs three seeds over all available medium-sized datasets and all internal reference baselines. It is exploratory. It may be used to diagnose where the method works, but not to remove losing datasets from the paper.
2. `confirmatory.yaml` fixes the main dataset list, model list, ten seeds, metrics, and stopping policy before the final run. Main claims must be based on this table.

Hyperparameters are selected using validation metrics only. `tuning.yaml` and `scripts/select_hyperparameters.py` never rank configurations by the test set.

## Dataset groups

### Challenging heterophily

Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions, Actor, Chameleon-filtered, Squirrel-filtered, Cornell, Texas, and Wisconsin.

### Homophilous and general controls

Cora, CiteSeer, PubMed, WikiCS, DBLP, Coauthor-CS, and Coauthor-Physics.

### Scale and task-boundary tests

ogbn-arxiv, ogbn-products, and ogbn-proteins. OGB datasets use official splits. `ogbn-proteins` is handled as multi-label classification and uses mean task ROC-AUC. GraphAtlas is disabled by default for `ogbn-products` until a sampling backend is selected.

### Recent novelty extension

`novelty.yaml` contains optional entries from H2GB and HeTGB-style text-attributed graphs. They are disabled until the corresponding datasets are imported. This prevents a normal benchmark run from silently downloading multi-gigabyte datasets.

## Primary metrics

- Standard multi-class node classification: Accuracy.
- Strong class imbalance: Macro-F1 or Balanced Accuracy, fixed in the dataset entry.
- Minesweeper, Tolokers, and Questions: ROC-AUC.
- Multi-label tasks: mean task ROC-AUC.
- Link prediction: ROC-AUC, Average Precision, MRR, Hits@10, and Hits@50.

All runs also save Macro-F1, Micro-F1, Weighted-F1, Balanced Accuracy, MCC, worst-class recall, NLL, Brier score, and ECE when applicable.

## GraphAtlas diagnostics

Every GraphAtlas run reports:

- affine and nonlinear coordinate intervention error;
- maximum logit deviation;
- prediction flip rate;
- intervention test-metric drop;
- boundary and interior accuracy when a fixed boundary mask exists;
- chart utilization, overlap ratio, active chart count, and membership entropy;
- inverse-cycle, path-consistency, metric-compatibility, and geometry diagnostics;
- runtime, parameter count, and peak CUDA memory.

A diagnostic is not automatically a positive claim. Path and metric errors should be reported even when they do not improve over an ablation.

## Statistical reporting

The reporting pipeline saves means, standard deviations, confidence intervals, paired t-tests, Wilcoxon tests, Holm-corrected p-values, win/tie/loss counts, and average model ranks. All pairwise tests match models on dataset, split, and seed.

## Fairness rules

- Same data split and initialization seed across paired models.
- Same maximum epoch and early-stopping policy within a table.
- Similar hyperparameter-search budgets across model families.
- External methods are not retrained in the default paper workflow. Their primary-paper or official-repository aggregates are imported through `published_results/` with protocol metadata. Internal implementations are engineering checks only.
- Original Chameleon and Squirrel are not used in the main confirmatory table. Filtered versions are used to avoid duplicate-node leakage.
