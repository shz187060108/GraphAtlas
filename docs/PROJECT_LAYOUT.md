# GraphAtlas project layout

Use the following paths as the canonical project interface.

## Source and configuration

- `src/graphatlas/`: library code, models, datasets, training and reporting.
- `configs/base.yaml`: shared defaults.
- `configs/presets/`: runnable experiment presets. Prefer the GraphAtlas-C, real, H2GB and OGB presets; older one-off presets are retained for reproducibility.
- `scripts/`: command-line entry points. See `scripts/README.md`.

## Results

- `outputs/results/`: combined result tables.
- `outputs/reports/`: summaries, gates and paper figures.
- `outputs/real_best_single/`: one-seed real-data runs.
- `outputs/selected_optuna/`: selected HPO studies and promoted runs.
- `outputs/best_config_search/`: test-selected configuration-search analysis and its validation-selected counterparts.
- `outputs/manifests/`: reproducible job manifests.

Runtime caches and logs are not part of the paper result interface. Do not use checkpoint directories as summary tables; use the CSV and JSON files at the dataset level instead.

## Reproducibility rule

Every reported result should identify its preset, dataset split, model seed and configuration hash. Test-selected analysis is exploratory; the validation-selected result is the appropriate benchmark record.
