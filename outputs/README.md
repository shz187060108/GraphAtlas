# Output index

Use these dataset-level files first:

- `results/*.csv`: combined metrics.
- `reports/`: summaries and figures.
- `reports/paper_atlas/`: current homogeneous main/supplementary paper-figure atlases; each figure has plotting data and an explicit manifest.
- `real_best_single/`: selected one-seed real runs.
- `selected_optuna/`: HPO and promoted configurations.
- `best_config_search/`: configuration-search summaries, trial tables and diagnostics.
- `manifests/`: exact materialized job plans.
- `archive/`: retained runtime logs and legacy smoke outputs; not part of the active result set.

Large `runs/` trees contain checkpoints and per-epoch histories. They are retained for recovery, but are not summary outputs. Cache and runtime folders are disposable and should not be committed.
