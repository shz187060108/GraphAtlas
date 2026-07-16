# GraphAtlas dense experiment and figure expansion

## Completed

- Added provenance-aware result loading. Raw trial exports remain separated from canonical summary rows by explicit protocol fields rather than model-name matching.
- The canonical `best_config_search/search_summary.csv` source takes the maximum of `test_selected_metric`, `validation_selected_val_metric`, and `validation_selected_test_metric`, then records the exact producing trial, seed, split, run directory, and parameter JSON. The three component measurements remain available for auditability.
- Added checkpoint-only coordinate stress evaluation for affine, asinh-affine, triangular-coupling, and radial transforms with continuous strengths and repeated transform seeds.
- Added deterministic local case selection and compact two-hop ego exports from existing prediction artifacts.
- Added homogeneous main and supplementary atlases with shared chart grammars, panel labels, explicit unavailable panels, plotting-data CSVs, and SVG/PDF/PNG bundles. No training is performed by the figure builder.
- Added dense phase, routing-causality, depth, and scaling presets and suite stage mappings.
- Set the dense-paper focal model and generated-preset reporting target to the formal `graphatlas_c_oracle` (`oracle_max_q`) variant. The learned `graphatlas_c` variant remains available as an explicit comparator.
- Added lightweight history analysis fields without forcing full Jacobian diagnostics at every evaluation.

## Commands run

```text
.venv\Scripts\python.exe scripts/generate_graphatlasc_presets.py
.venv\Scripts\python.exe scripts/generate_graphatlasc_presets.py --check
.venv\Scripts\python.exe scripts/run_graphatlasc_suite.py --stage <new-stage> --dry-run
.venv\Scripts\python.exe scripts/build_dense_paper_figures.py --results outputs/best_config_search/search_summary.csv --output-dir outputs/reports/paper_atlas --target-model graphatlas_c_oracle
.venv\Scripts\python.exe -m pytest -q tests/test_coordinate_stress.py tests/test_case_studies.py tests/test_experiment_separation.py tests/test_dense_paper_figures.py tests/test_paper_figures.py tests/test_graphatlasc_smoke_pipeline.py
.venv\Scripts\python.exe -m pytest -q tests/test_reporting.py tests/test_reparameterization_families.py tests/test_validation.py tests/test_graphatlasc_manifests.py tests/test_certified_routing_controls.py
```

## Dry-run matrix sizes

| preset | runs expanded |
|---|---:|
| graphatlasc_phase_dense_screening | 1215 |
| graphatlasc_phase_dense_confirmatory | 800 |
| graphatlasc_routing_causal | 440 |
| graphatlasc_depth_dynamics | 90 |
| graphatlasc_scaling | 108 |

## Outputs

- Dense figures: `outputs/reports/paper_atlas/main/` and `outputs/reports/paper_atlas/supplementary/`
- Plotting data: `outputs/reports/paper_atlas/data/`
- Manifests: `outputs/reports/paper_atlas/manifests/`

## Not run

No new training matrix, full scaling sweep, or dataset download was performed. Coordinate stress and local case scripts are ready but were not run against every historical checkpoint; missing artifacts are recorded as skipped/unavailable rather than synthesized.
