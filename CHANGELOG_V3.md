# GraphAtlas AAAI package v3

## Experiment policy

- Default presets train only GraphAtlas and GraphAtlas ablations.
- Added a provenance-aware published result registry.
- Exact and contextual external comparisons are separated automatically.
- Published aggregates are excluded from paired significance tests and model-rank statistics.

## Data support

- Recursive Windows local-data discovery under arbitrary project roots.
- Planetoid raw, PyG `processed/data.pt`, OGB cache, Geom-GCN text files and GraphAtlas NPZ import.
- New optional groups for HeTGB, H2GB, LINKX large non-homophily datasets and CITE.
- Typed homogeneous projection preserves node and relation type features for heterogeneous inputs.

## Execution

- Cross-platform experiment lock for Linux and Windows.
- Optional presets automatically enable only materialized datasets.
- PowerShell switches for text heterophily, H2GB, large non-homophily and frontier text graphs.
- Comparison table generation is integrated into the report pipeline.

## Validation

- Curated 179 published result rows across 12 dataset versions.
- Unit tests cover schema validation, score normalization and exact/contextual isolation.
