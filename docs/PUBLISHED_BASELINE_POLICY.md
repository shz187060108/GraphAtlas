# Published baseline policy

The paper workflow does not retrain external methods. It trains only GraphAtlas and its mechanism ablations. External numbers are imported from primary papers or official repositories.

## Comparison levels

- `exact`: dataset version, split files, node features, task and metric match.
- `contextual`: useful for orientation, but at least one protocol field differs.
- `incompatible`: shown only in the provenance audit.

The comparison builder automatically downgrades an `exact` row when its metadata does not match `configs/comparison_protocols.yaml`.

## Statistical rule

Published means and standard deviations do not provide paired seed-level observations. They are never used in paired tests, Holm correction, critical-difference diagrams or seed-level model ranks. Significance testing is restricted to GraphAtlas variants trained on identical splits.

## Curated sources

- ICLR 2023 Heterophilous Graph Benchmark official results notebook.
- ICLR 2023 filtered Chameleon and Squirrel results.
- HeTGB 2025 text-feature tables, marked contextual unless the exact released embeddings are used.
- H2GB KDD 2025, LINKX NeurIPS 2021 and CITE 2025 are registered as source families. Empty templates are provided for checked table transcription.

Run `python scripts/validate_published_results.py` before every paper build.
