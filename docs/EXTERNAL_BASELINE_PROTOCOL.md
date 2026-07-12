# External baseline protocol

External methods must consume the NPZ exported by `scripts/export_external_split.py`, including the exact GraphAtlas features, edges, masks, seed, split, and metric. A wrapper may adapt this standard input to an official repository pinned to a verified commit; it may not reimplement or simplify the method.

The registry deliberately leaves GraphMoRE, GeoMoE, ARGNN, and Neural Sheaf Diffusion unconfigured until official repositories and commits are verified. An unconfigured runner exits without creating metrics. Successful rows must come from parsed official output and use `protocol: direct_same_split` to enter direct-comparison gates.

Published means remain in the published-comparison system and must never be expanded into fake seed rows or used for paired inference.
