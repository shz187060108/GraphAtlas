# Recent benchmark extensions

## H2GB

H2GB studies graphs that are both heterogeneous and heterophilic. The benchmark was published at KDD 2025 and its repository describes nine real-world datasets across academia, finance, e-commerce, social science, and cybersecurity. Several datasets contain millions of nodes, so they are intended as a scale and applicability extension rather than a replacement for the main homogeneous-graph table.

Recommended first datasets:

- IEEE-CIS-G: fraud detection with strong imbalance;
- PDNS: malicious-domain detection;
- H-Pokec: social-network node classification;
- mag-year: a manageable target before the largest OAG and RCDD datasets.

GraphAtlas currently imports PyG `HeteroData` by creating a typed homogeneous projection. Node and edge types are preserved. `feature_source: native_plus_type_relation` appends node-type one-hot features and label-free per-relation in/out degree signatures. This is a controlled compatibility baseline, not a claim that the current model fully solves heterogeneous message passing. A stronger future extension would learn type-conditioned charts and relation-conditioned transports.

## HeTGB-style text-attributed heterophily

HeTGB introduced text-enriched versions of Cornell, Texas, Wisconsin, Actor, and Amazon in 2025. They are registered separately as `hetgb_cornell`, `hetgb_texas`, `hetgb_wisconsin`, `hetgb_actor`, and `hetgb_amazon`, because their reconstructed graphs are not identical to the classic benchmark copies. GraphAtlas accepts standardized NPZ files with optional `text_features` and `structural_features` arrays. This supports three pre-registered feature conditions:

- `feature_source: text`;
- `feature_source: structure`;
- `feature_source: text_plus_structure`.

Use `scripts/build_structural_features.py` to add label-free degree and PageRank signatures. Use `scripts/embed_text_features.py` to produce fixed text embeddings and cache them in the NPZ. The encoder name, pooling rule, normalization setting, source-text hash, and embedding hash are recorded in a sidecar manifest.

## Why these extensions matter

H2GB tests whether local coordinate charts can organize different entity and relation regimes. Text-attributed heterophily tests whether chart transport remains useful when semantic features are strong. Both are closer to the GraphAtlas premise than adding another small citation network.

## Activation

Optional datasets are disabled in `configs/presets/novelty.yaml`. After import, generate an active preset:

```bash
python scripts/activate_available_datasets.py \
  --preset novelty \
  --output configs/presets/novelty_available.yaml

python scripts/run_pipeline.py --preset configs/presets/novelty_available.yaml
```
