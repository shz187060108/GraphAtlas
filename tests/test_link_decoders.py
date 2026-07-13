from __future__ import annotations

import torch

from graphatlas.nn.link_decoders import build_link_decoder


def test_link_decoders_are_symmetric_and_finite():
    embedding = torch.randn(6, 5)
    edges = torch.tensor([[0, 1, 2], [3, 4, 5]])
    reverse = edges.flip(0)
    for kind in ("dot", "bilinear", "mlp"):
        decoder = build_link_decoder(kind, 5, 8)
        assert torch.allclose(decoder(embedding, edges), decoder(embedding, reverse), atol=1e-6)
