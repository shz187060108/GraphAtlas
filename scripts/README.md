# GraphAtlas command guide

Run commands from the project root with the intended Python environment.

## Main entry points

```text
run_pipeline.py                 Run a materialized preset.
run_all.py                      Train one preset through the standard trainer.
run_graphatlasc_suite.py        Run the GraphAtlas-C suite stages.
run_selected_hpo.py             Legacy tune/promote/final runner retained for checkpoint recovery.
run_best_config_search.py       Run or refresh the canonical best-configuration search.
run_real_best_single.py         Run one selected real-data configuration.
clean_outputs.py                Dry-run/apply removal of superseded reports and low-value summaries.
```

## Data preparation

```text
download_data.py                Public dataset preparation.
download_h2gb.py                H2GB conversion.
download_hetgb.py               HeTGB preparation.
check_h2gb.py                   H2GB cache validation.
h2gb_bridge.py                 H2GB format bridge.
```

## Reporting and figures

```text
collect_results.py              Collect run metrics.
summarize.py                    Build result summaries.
visualize.py                    Build standard diagnostics.
build_paper_figures.py          Build paper figure bundles.
```

## Operational helpers

```text
preflight.py                    Validate a preset before execution.
materialize_jobs.py             Materialize a job manifest.
run_overnight_experiments.ps1   Detached overnight workflow.
watch_overnight_experiments.ps1 Monitor that workflow.
```

The remaining scripts are maintenance, legacy compatibility or specialized ablations. Keep them for reproducibility, but do not use them as the default entry point unless a preset explicitly requires them.
