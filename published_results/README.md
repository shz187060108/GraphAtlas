# Published baseline result registry

The main experiment presets train only GraphAtlas and GraphAtlas ablations. External methods are taken from primary-paper or official-repository aggregate tables.

Rules:

1. Every number must record dataset version, feature condition, split protocol, metric, table and source.
2. `exact` means those fields match the GraphAtlas run. A validator automatically downgrades mismatches to `contextual`.
3. Published aggregate means are never used for paired tests, seed-level ranks or significance claims.
4. Original and filtered Chameleon/Squirrel are different datasets.
5. H2GB numbers remain contextual while GraphAtlas uses a typed homogeneous projection rather than the native heterogeneous evaluator.

Run:

```bash
python scripts/validate_published_results.py
python scripts/build_comparison_tables.py --results outputs/results/confirmatory.csv --output-dir outputs/reports/confirmatory/published_comparison
```
