# GraphAtlas-C experiments

`graphatlas_c` is the display label for the backward-compatible internal model
family `graphatlas_certified`. It compares original transport, canonical
minimum-distortion transport, certificate routing, anti-certificate routing,
and deterministic routing controls under identical splits.

Generate and inspect plans with `python scripts/generate_graphatlasc_presets.py`
and `python scripts/run_graphatlasc_suite.py --stage screening --dry-run`.
Runs are resumable under `outputs/.work/runs`; full and concise tables are
written separately under `outputs/results`. GitHub synchronization is opt-in.

`python scripts/build_paper_figures.py --results outputs/results/latest.csv`
creates high-density main and supplementary figure grids under
`outputs/reports/paper_figures`. The figure contract is SVG-only and exports a
CSV plotting-data table beside every requested figure, including skipped data
branches.

Certificate routing measures target-chart representability only. It is not a
semantic-edge classifier, a noise detector, or intrinsic parallel transport.
