# Roman Empire P0 matched baselines

## Ambient Vector GNN

Ambient Vector GNN retains GraphAtlas's graph signature, observation encoder and dimension, vector channels, attention rule, channel mixing, scalar gates, layer count, and final readout dimension. Its routed observation-space adapters and routed gate experts provide a capacity-matched comparison.

It removes local charts, chart coordinates, encoder and decoder Jacobians, pull/push operations, chart transitions, and chart-tangent states. Every vector message remains in one shared Euclidean observation space.

## Signature GNN

Signature GNN directly consumes the same graph signature used by GraphAtlas: raw node features, normalized log degree, and the norm of the difference between a node feature and its neighbor mean. It tests whether this signature alone explains performance differences.

## Why GeometryMoE is not a substitute

GeometryMoE mixes predictions from independent GCN experts. It does not preserve GraphAtlas's observation-vector state, channel mixing, attention, scalar vector gates, or matched readout. Ambient Vector GNN is therefore the direct ablation of atlas geometry; GeometryMoE answers a separate expert-fusion question.

## Roman P0 matrix

Roman P0 evaluates ten fixed Roman Empire splits for ten models: Ambient Vector GNN, Signature GNN, GeometryMoE, GraphAtlas without transport, GraphAtlas with free transitions, GraphAtlas without metric regularization, GraphAtlas without cocycle regularization, GraphAtlas without overlap, GraphAtlas without chart-rank regularization, and full GraphAtlas. The complete manifest contains 100 jobs.

## Expansion rule

Full GraphAtlas should be extended beyond Roman Empire only after P0 establishes that its gain is not explained by graph signature, model capacity, generic expert fusion, transport removal, transition parameterization, or individual regularizers. Expansion decisions must use completed multi-seed results and uncertainty estimates; this document makes no performance claim for the new baselines.
