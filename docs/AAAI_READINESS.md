# AAAI readiness checklist

This repository is engineered for a reproducible submission study. Implementation quality and a successful synthetic mechanism test do not guarantee acceptance.

## Claims directly supported by the packaged three-seed study

1. Full GraphAtlas is numerically invariant under independent nonlinear chart reparameterizations.
2. Removing transport or using free transition matrices breaks that invariance by roughly six orders of magnitude.
3. In the matched one-layer Atlas-Het setting, full GraphAtlas improves mean boundary accuracy over no transport by 4.3 percentage points.
4. It improves mean task metric over free transitions by 1.4 percentage points.
5. The complete operator incurs about 29% runtime overhead over no transport on the packaging CPU.

All five automated claim gates pass for `outputs/results/mechanism.csv`. The evidence contains only three seeds and one synthetic setting, so these statements remain preliminary.

## Required evidence before submission

- All ten fixed splits on every real benchmark.
- At least three independent Atlas-Het settings varying overlap, heterophily and chart count.
- Mean, standard deviation, 95% confidence intervals and paired tests.
- Parameter count, runtime and peak-memory comparisons.
- Boundary, interior and cross-chart metrics.
- Coordinate-intervention figures generated from the exact reported result table.
- A clear explanation that current path-consistency and metric-compatibility diagnostics do not outperform the ablations under low regularization.
- No unsupported claim in the final `scientific_gates.json` or figure report.

## Execution

Run the focused validated study with `bash run_mechanism.sh`. Run the complete configured study with `bash run_everything.sh`. Both commands are resumable and use isolated workers. A scheduler-friendly low-level entry point is `bash scripts/run_preset_isolated.sh <preset>`.
