from __future__ import annotations

import torch
from torch import nn

from .functional import MLP, edge_dot_scores


class DotLinkDecoder(nn.Module):
    def forward(self, embedding: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return edge_dot_scores(embedding, edge_index)


class BilinearLinkDecoder(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dimension))

    def forward(self, embedding: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        return (embedding[source] @ self.weight * embedding[target]).sum(dim=-1)


class MLPLinkDecoder(nn.Module):
    def __init__(self, dimension: int, hidden_dim: int):
        super().__init__()
        self.network = MLP(dimension * 3, hidden_dim, 1, layers=3, dropout=0.0)

    def forward(self, embedding: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        left, right = embedding[source], embedding[target]
        features = torch.cat([left + right, (left - right).abs(), left * right], dim=-1)
        return self.network(features).squeeze(-1)


def build_link_decoder(kind: str, dimension: int, hidden_dim: int) -> nn.Module:
    if kind == "dot":
        return DotLinkDecoder()
    if kind == "bilinear":
        return BilinearLinkDecoder(dimension)
    if kind == "mlp":
        return MLPLinkDecoder(dimension, hidden_dim)
    raise ValueError(f"Unknown link decoder: {kind}")


class LinkPredictionModel(nn.Module):
    """Owns an encoder and its checkpointed symmetric link decoder."""
    def __init__(self, encoder: nn.Module, embedding_dim: int, decoder: str, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = build_link_decoder(decoder, embedding_dim, hidden_dim)

    def forward(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        return self.encoder(*args, **kwargs)  # type: ignore[no-any-return]

    def score_edges(self, embedding: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoder(embedding, edge_index)
