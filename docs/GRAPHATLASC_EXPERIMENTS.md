# GraphAtlas-C experiments

`graphatlas_c` is the display label for the backward-compatible internal model
family `graphatlas_certified`. It compares original transport, canonical
minimum-distortion transport, certificate routing, anti-certificate routing,
and deterministic routing controls under identical splits.

Generate and inspect plans with `python scripts/generate_graphatlasc_presets.py`
and `python scripts/run_graphatlasc_suite.py --stage screening --dry-run`.
Runs are resumable under `outputs/.work/runs`; full and concise tables are
written separately under `outputs/results`. GitHub synchronization is opt-in.

Certificate routing measures target-chart representability only. It is not a
semantic-edge classifier, a noise detector, or intrinsic parallel transport.
