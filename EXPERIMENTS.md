# Experiment protocol

## Entry points

| Command | Purpose |
|---|---|
| `bash setup.sh` | Create `.venv`, install the package and validate the environment |
| `bash run_mechanism.sh` | Run the focused three-seed coordinate-mechanism matrix and figures |
| `bash run_smoke.sh` | Run 24 fast synthetic jobs, summaries, figures and both test suites |
| `bash run_paper.sh` | Run the ten-seed node-classification matrix |
| `bash run_link.sh` | Run the ten-seed link-prediction matrix |
| `bash run_everything.sh` | Download data, run both complete matrices, combine results and create figures |
| `python scripts/status.py` | Display progress from the persisted status files |

Every experiment run has a stable configuration and source hash. Completed runs are skipped unless `--force` is passed. Interrupted runs resume from `last.pt` when possible. The pipeline uses `scripts/run_preset_isolated.sh`, which starts each job through a fresh worker process and displays an experiment progress bar; training can display its own epoch bar.


## Full matrix size

The checked manifests contain 1,440 node-classification runs and 1,170 link-prediction runs, for 2,610 resumable jobs in total. These counts include all configured datasets, models, splits and ten random seeds. The smoke preset contains 24 jobs and is only a software and mechanism check.

## Output layout

```text
outputs/
  runs/<dataset>/<task>/<model>/split_<s>/seed_<s>/cfg_<hash>/
  results/<preset>.csv
  reports/<preset>/
    summary.csv
    paired_tests.csv
    model_ranks.csv
    scientific_gates.json
    summary.md
    figures/
  status/
    experiment_<preset>.json
    pipeline_<preset>.json
  manifests/
    <preset>.csv
    <preset>.jsonl
  cache/
```

Each run directory contains the resolved configuration, environment metadata, best and last checkpoints, training history, metrics and predictions.

## Scientific acceptance gates

The reporting code evaluates five gates on Atlas-Het:

1. Full GraphAtlas nonlinear coordinate-intervention error below `1e-6`.
2. No-transport intervention error at least ten times larger.
3. Free-transition intervention error at least ten times larger.
4. Mean boundary-node gain over no transport of at least two percentage points.
5. Positive paired task-metric gain over free transitions.

The first three validate the coordinate-invariance mechanism. The final two are required before claiming predictive benefit from atlas transport. The validated one-layer three-seed matrix passes all five gates, but final submission claims still require the ten-seed real-data matrix and additional Atlas-Het settings.

## Practical execution

The complete matrix is intentionally large. Start with `bash run_smoke.sh`. Full runs are resumable, so the same command can be restarted after interruption. Use `GRAPHATLAS_PROGRESS=0` to disable terminal progress bars in a batch scheduler.

## Process and memory isolation

The matrix runner is intentionally lightweight. Each training job starts in a fresh Python process and exits after its artifacts are validated. This releases PyTorch, BLAS and CUDA allocator state between runs. Set `GRAPHATLAS_IN_PROCESS=1` only for debugging. Default shell entry points constrain native thread pools to one thread; override with `GRAPHATLAS_NUM_THREADS=<n>` after measuring the target machine.

## Verified three-seed mechanism result

The packaged result table is `outputs/results/mechanism.csv`. Across three fixed seeds, full GraphAtlas reaches 0.739 mean test metric and 0.737 mean boundary accuracy. The paired boundary gain over no transport is 0.043 and the paired test-metric gain over free transitions is 0.014. Full nonlinear coordinate-intervention error is 6.7e-8, versus 6.6e-2 for no transport and 1.43e-1 for free transitions. Runtime is 13.3 seconds per run on the packaging CPU, compared with 10.3 and 11.7 seconds for the two ablations. Path-consistency and metric-compatibility diagnostics do not improve under the current low regularization, so the repository does not present them as positive empirical claims.
