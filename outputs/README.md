# Output index

Use these dataset-level files first:

- `best_config_search/search_summary.csv`: canonical best-observed dataset summary, including the producing trial, seed and parameter JSON.
- `results/`: mechanism tables and the retained H2GB PDNS result; transient `latest.csv` files are not canonical.
- `reports/`: summaries and figures.
- `reports/paper_atlas/`: current homogeneous main/supplementary paper-figure atlases; each figure has plotting data and an explicit manifest.
- `selected_optuna/`: legacy HPO checkpoints retained locally for recovery and excluded from new commits.
- `best_config_search/`: the canonical summary plus local trial artifacts and studies.
- `manifests/`: exact materialized job plans.
- `archive/`: retained runtime logs and legacy smoke outputs; not part of the active result set.

Large `runs/` trees contain checkpoints and per-epoch histories. They are retained for recovery, but are not summary outputs. Cache and runtime folders are disposable and should not be committed.
