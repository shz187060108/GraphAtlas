# GraphAtlas

GraphAtlas is a reproducible research implementation of coordinate-invariant message passing over learned local graph charts. The code uses standard PyTorch and does not require PyTorch Geometric.

## Core correction

The initial source-only transition was not equivariant under nonlinear local coordinate changes because a source message and a target self-state are based at different points. The implemented operator uses point-to-point transport through a shared Euclidean observation tangent space:

\[
T_{j,l\rightarrow i,k}
=
J\phi_k(h_i)J\psi_l(u_j^{(l)}).
\]

Learnable matrices act only on vector channels. Nonlinear updates use scalar gates computed from observation-space invariants. The final readout pushes every chart state back into observation space before aggregation. Under the assumptions in `THEORY.md`, the complete GraphAtlas forward pass is equivariant in chart coordinates and invariant at the prediction level.

## Installation

```bash
bash setup.sh
```

The script creates `.venv`, installs the package and dependencies, then validates the environment. A manually managed environment can instead use:

```bash
python -m pip install -e ".[dev]"
```

Python 3.10 or later is required. CUDA is used automatically when available.

## Fast verification

```bash
bash run_smoke.sh
```

This command validates the manifest, executes the small synthetic node-classification and link-prediction matrix, computes summaries, generates PDF and PNG figures, then runs the regular and visualization test suites in fresh processes. It displays stage, experiment and epoch progress bars and writes resumable artifacts throughout.

The smoke configuration verifies software behavior. It is not intended as a paper result.

For a focused three-seed check of the paper's central mechanism, run:

```bash
bash run_mechanism.sh
```

This executes the validated one-layer, low-regularization three-seed matrix for full GraphAtlas, no transport and free transitions. It produces separate figures for coordinate stability, boundary-node behavior, induced transitions, task performance and runtime.

## One-command complete experiments

```bash
bash run_everything.sh
```

This downloads every real dataset referenced by the paper and link-prediction presets, runs the ten-seed matrices, combines their results, performs paired statistical tests and generates all figures. Runs are resumable. Restarting the same command skips complete runs and resumes valid interrupted checkpoints. The default pipeline uses an isolated worker launcher so each job receives a fresh Python and PyTorch process.

Separate entry points are also available:

```bash
bash run_mechanism.sh   # focused three-seed mechanism check
bash run_paper.sh       # node classification
bash run_link.sh        # link prediction
bash run_visualize.sh outputs/results/complete.csv outputs/reports/complete
python scripts/status.py
```

Pass runner options through the shell entry points:

```bash
bash run_paper.sh --continue-on-error
bash run_paper.sh --force
bash run_paper.sh --limit 10
```

The low-level isolated runner is also available for schedulers and recovery:

```bash
bash scripts/run_preset_isolated.sh paper --retries 1
```

It materializes a deterministic JSONL plan, displays a job progress bar, validates every artifact, retries interrupted jobs from `last.pt`, and collects the final CSV.

## Included models

The default paper matrix trains only GraphAtlas and the following GraphAtlas mechanism variants:

- no transport;
- free chart-pair transitions;
- no metric compatibility;
- no cocycle consistency;
- no overlap;
- full GraphAtlas.

## Datasets and tasks

Synthetic:

- Atlas-Het with chart membership, overlap labels, chart coordinates, latent geometry and independent coordinate interventions.

Real:

- Roman-empire;
- Amazon-ratings;
- Minesweeper;
- Tolokers;
- Questions;
- Actor;
- filtered Chameleon;
- filtered Squirrel.

Tasks:

- node classification;
- leakage-controlled link prediction using a training graph that preserves a spanning forest.

## Results and figures

Each run is stored at:

```text
outputs/runs/<dataset>/<task>/<model>/split_<split>/seed_<seed>/cfg_<hash>/
```

A run directory contains:

- resolved `config.yaml`;
- `environment.json`;
- `best.pt` and resumable `last.pt`;
- `history.csv`;
- `metrics.json`;
- `predictions.pt`.

Preset-level files include manifests, progress status, failures and combined CSV tables. Reports include mean and standard deviation, 95% confidence intervals, paired t-tests, Wilcoxon tests with Holm correction, model ranks, scientific claim gates and vector/raster figures.

The mechanism figure set separates four questions:

1. whether nonlinear chart reparameterization changes predictions;
2. whether transport improves boundary nodes;
3. whether induced transitions outperform free matrices;
4. how the models compare on the primary task metric.

The generated figure report explicitly states which claims pass and which are unsupported by the current results.

## Reproducibility controls

- deterministic Python, NumPy and PyTorch seeds;
- fixed benchmark splits;
- stable configuration hashes;
- atomic checkpoint saves;
- corrupt-checkpoint quarantine and deterministic restart;
- safe checkpoint resumption;
- dataset validation and SHA-256 recording;
- environment metadata per run;
- subprocess isolation for each run so native allocator state is released;
- conservative BLAS, OpenMP and PyTorch thread limits by default;
- no external experiment manager;
- CPU and CUDA support.

## Packaging

Create a portable source archive without datasets or large checkpoints:

```bash
python scripts/package_project.py
```

The command writes a ZIP archive and a SHA-256 checksum. Use `--include-runs` only when checkpoints must be included.

## Current evidence status

The verified three-seed Atlas-Het mechanism study passes all five automated claim gates. Full GraphAtlas obtains mean test metric 0.739 and mean boundary accuracy 0.737. The no-transport values are 0.703 and 0.694; the free-transition values are 0.725 and 0.675. Full-model nonlinear intervention error is 6.7e-8, roughly one to two million times lower than the incorrect ablations. These are preliminary synthetic results with three seeds, not a substitute for the configured ten-seed real-data study. See `PROJECT_STATUS.md` and the generated figure report.

---

## Extended benchmark edition

This package includes an expanded experimental layer for broad local-data evaluation and recent benchmark extensions.

### Windows one-click benchmark

```powershell
powershell -ExecutionPolicy Bypass -File .\run_full_benchmark.ps1 `
  -SearchRoot "C:\Users\24953\PycharmProjects" `
  -Stage screening
```

The command discovers existing graph datasets, imports supported formats, audits data integrity, downloads missing core datasets, runs the selected matrix with progress bars and resume support, and generates statistics and figures.

### Experiment stages

```bash
bash run_screening.sh        # all available medium datasets, 3 seeds
bash run_main_tables.sh      # fixed confirmatory table, 10 seeds
bash run_all_ablations.sh    # Atlas-Het grid and mechanism ablations
bash run_large_scale.sh      # OGB scale tasks
```

### Added data formats

- Planetoid raw files
- PyG `processed/data.pt`
- generic NPZ files
- OGB local caches
- typed PyG `HeteroData` projection
- text and structural feature blocks for text-attributed graphs

### Optional implementation checks

- Logistic regression
- adjacency-only LINK
- sparse GAT
- GCNII
- FAGCN
- ACM-GCN

Legacy baseline implementations remain for engineering tests only. The default paper presets do not train them. External comparison numbers are loaded from the provenance-aware published result registry.

### Added metrics

The result files now include Accuracy, Macro/Micro/Weighted F1, Balanced Accuracy, MCC, worst-class recall, ROC-AUC, Average Precision, NLL, Brier score, ECE, MRR, Hits@10, Hits@50, coordinate-intervention flip rate, intervention performance drop, chart utilization, overlap ratio, membership entropy, runtime, parameters, and peak CUDA memory when applicable.

See:

- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/LOCAL_DATA_WINDOWS.md`
- `docs/NOVELTY_DATASETS.md`

## Published external results

The default workflow trains only GraphAtlas and its ablations. External methods are read from `published_results/data/`, validated, and separated into exact and contextual comparisons.

```bash
python scripts/validate_published_results.py
python scripts/build_comparison_tables.py \
  --results outputs/results/confirmatory.csv \
  --output-dir outputs/reports/confirmatory/published_comparison
```

See `docs/PUBLISHED_BASELINE_POLICY.md` and `docs/BENCHMARK_STRATEGY_2026.md`.
