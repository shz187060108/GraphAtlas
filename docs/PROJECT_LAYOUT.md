# GraphAtlas project layout

Use the following paths as the canonical project interface.

## Source and configuration

- `src/graphatlas/`: library code, models, datasets, training and reporting.
- `configs/base.yaml`: shared defaults.
- `configs/presets/`: reusable experiment presets. One-off best-result presets are generated on demand by `scripts/run_real_best_single.py` instead of being retained as stale static files.
- `scripts/`: command-line entry points. See `scripts/README.md`.

## Results

- `outputs/best_config_search/search_summary.csv`: canonical best-observed result and parameter table.
- `outputs/results/`: mechanism tables and retained non-search benchmark summaries.
- `outputs/reports/`: summaries, gates and paper figures.
- `outputs/selected_optuna/`: legacy studies retained locally for recovery, not a reporting source.
- `outputs/best_config_search/`: canonical summary plus local resumable search artifacts.
- `outputs/manifests/`: reproducible job manifests.

Runtime caches and logs are not part of the paper result interface. Do not use checkpoint directories as summary tables; use the CSV and JSON files at the dataset level instead.

## Reproducibility rule

Every best-observed row records the metric source, producing trial, seed, split protocol and exact parameter JSON. Validation-selected and test-selected component metrics remain separate columns for auditability.
