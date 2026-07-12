# Benchmark strategy

GraphAtlas is evaluated by mechanism, not by indiscriminately collecting datasets.

## Tier 1: homogeneous heterophily

Use HGB, filtered Chameleon/Squirrel, Actor and WebKB. These form the main performance and boundary-node tables.

## Tier 2: text-attributed heterophily

Use HeTGB as a semantic-coordinate stress test. Keep its reconstructed graphs separate from classic Cornell, Texas, Wisconsin, Actor and Amazon. Report the text encoder and embedding checksum.

## Tier 3: heterogeneous and heterophilic graphs

Use H2GB as a frontier extension. The current loader preserves node and relation type features but projects the graph to one typed homogeneous graph. Results must be labeled as a typed projection, not as a native heterogeneous GraphAtlas.

## Tier 4: large non-homophilous graphs

Penn94, genius, twitch-gamer, arxiv-year and snap-patents are optional scalability tests. Enable them only after sampled transport is available.

## Tier 5: heterogeneous text graphs

CITE is included as a disabled frontier configuration. It requires the text and heterogeneous data pipeline and is not part of the default AAAI table.

## Pre-registered conclusions

1. The primary claim is coordinate invariance under local reparameterization.
2. Boundary and cross-chart improvements are secondary mechanism claims.
3. Metric compatibility and path consistency remain diagnostics unless their ablations show consistent gains.
4. All available datasets may be explored, but the confirmatory dataset list is fixed before final test inspection.
