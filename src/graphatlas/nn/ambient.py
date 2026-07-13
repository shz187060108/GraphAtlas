from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from graphatlas.config import ModelConfig
from graphatlas.data import GraphData
from .functional import MLP, attention_aggregate, cached_graph_signature


class AmbientVectorLayer(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        channels: int,
        num_charts: int,
        hidden_dim: int,
        edge_chunk_size: int = 100_000,
        edge_chunk_threshold: int = 2_000_000,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout_probability = float(dropout)
        self.edge_chunk_size = int(edge_chunk_size)
        self.edge_chunk_threshold = int(edge_chunk_threshold)
        self.self_mix = nn.Parameter(torch.eye(channels) + 0.02 * torch.randn(channels, channels))
        self.message_mix = nn.Parameter(torch.eye(channels) + 0.02 * torch.randn(channels, channels))
        self.residual_mix = nn.Parameter(0.1 * torch.eye(channels))
        self.query = nn.Linear(observation_dim, hidden_dim, bias=False)
        self.key = nn.Linear(observation_dim, hidden_dim, bias=False)
        self.gates = nn.ModuleList(
            [
                MLP(
                    channels + observation_dim + 1,
                    hidden_dim,
                    channels,
                    layers=2,
                    dropout=dropout,
                )
                for _ in range(num_charts)
            ]
        )
        self.gate_router = MLP(
            observation_dim,
            hidden_dim,
            num_charts,
            layers=2,
            dropout=dropout,
        )

    def forward(
        self,
        vectors: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mixed = torch.einsum("ndc,ce->nde", vectors, self.message_mix)
        queries = self.query(h)
        keys = self.key(h)
        aggregated, alpha = attention_aggregate(
            mixed, edge_index, queries, keys, self.edge_chunk_size, self.edge_chunk_threshold,
        )

        self_term = torch.einsum("ndc,ce->nde", vectors, self.self_mix)
        preactivation = self_term + aggregated
        norms = torch.sqrt(preactivation.square().sum(dim=1) + 1e-10)

        constant_membership = torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype)
        gate_input = torch.cat([norms, h, constant_membership], dim=-1)
        candidate_gates = torch.stack(
            [torch.sigmoid(gate(gate_input)) for gate in self.gates],
            dim=1,
        )
        router = torch.softmax(self.gate_router(h), dim=-1)
        gate = (candidate_gates * router[:, :, None]).sum(dim=1)
        gate = F.dropout(gate, p=self.dropout_probability, training=self.training)

        gated = preactivation * gate[:, None, :]
        residual = torch.einsum("ndc,ce->nde", vectors, self.residual_mix)
        output = gated + residual
        return output, {"attention": alpha, "norms": norms, "gate_router": router}


class AmbientVectorGNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, config: ModelConfig):
        super().__init__()
        self.config = config
        signature_dim = input_dim + 2
        self.observation_encoder = MLP(
            signature_dim,
            config.hidden_dim,
            config.observation_dim,
            layers=3,
            dropout=config.dropout,
        )
        self.vector_initializer = MLP(
            signature_dim,
            config.hidden_dim,
            config.observation_dim * config.vector_channels,
            layers=2,
            dropout=config.dropout,
        )
        self.ambient_router = MLP(
            config.observation_dim,
            config.hidden_dim,
            config.num_charts,
            layers=2,
            dropout=config.dropout,
        )
        self.ambient_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    MLP(
                        config.observation_dim,
                        config.hidden_dim,
                        config.observation_dim,
                        layers=3,
                        dropout=config.dropout,
                    ),
                    MLP(
                        config.observation_dim,
                        config.hidden_dim,
                        config.observation_dim,
                        layers=3,
                        dropout=config.dropout,
                    ),
                )
                for _ in range(config.num_charts)
            ]
        )
        self.layers = nn.ModuleList(
            [
                AmbientVectorLayer(
                    observation_dim=config.observation_dim,
                    channels=config.vector_channels,
                    num_charts=config.num_charts,
                    hidden_dim=config.hidden_dim,
                    edge_chunk_size=config.edge_chunk_size,
                    edge_chunk_threshold=config.edge_chunk_threshold,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        readout_dim = config.observation_dim * config.vector_channels + config.observation_dim
        self.classifier = MLP(
            readout_dim,
            config.hidden_dim,
            num_classes,
            layers=3,
            dropout=config.dropout,
        )

    def forward(self, data: GraphData, compact: bool = False, **_: object) -> dict[str, object]:
        signature = cached_graph_signature(data, self.config.edge_chunk_size)
        h = self.observation_encoder(signature)
        adapter_weights = torch.softmax(self.ambient_router(h), dim=-1)
        adapter_values = torch.stack([adapter(h) for adapter in self.ambient_adapters], dim=1)
        h = h + (adapter_values * adapter_weights[:, :, None]).sum(dim=1)

        vectors = self.vector_initializer(signature).view(
            data.num_nodes,
            self.config.observation_dim,
            self.config.vector_channels,
        )
        diagnostics: list[dict[str, torch.Tensor]] = []
        for layer in self.layers:
            vectors, layer_diagnostics = layer(vectors, h, data.edge_index)
            diagnostics.append(layer_diagnostics)

        readout = torch.cat([h, vectors.flatten(start_dim=1)], dim=-1)
        logits = self.classifier(readout)
        if compact:
            return {"logits": logits, "embedding": readout}
        return {
            "logits": logits,
            "embedding": readout,
            "observation": h,
            "observation_vectors": vectors,
            "ambient_router": adapter_weights,
            "layer_diagnostics": diagnostics,
        }


class SignatureAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        edge_chunk_size: int = 100_000,
        edge_chunk_threshold: int = 2_000_000,
    ):
        super().__init__()
        self.dropout_probability = float(dropout)
        self.edge_chunk_size = int(edge_chunk_size)
        self.edge_chunk_threshold = int(edge_chunk_threshold)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        message = self.value(h)
        aggregated, alpha = attention_aggregate(
            message,
            edge_index,
            self.query(h),
            self.key(h),
            self.edge_chunk_size,
            self.edge_chunk_threshold,
        )
        update = self.output(aggregated)
        update = F.dropout(
            F.gelu(update),
            p=self.dropout_probability,
            training=self.training,
        )
        h = self.norm(h + update)
        return h, alpha


class SignatureGNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, config: ModelConfig):
        super().__init__()
        self.config = config
        signature_dim = input_dim + 2
        self.encoder = MLP(
            signature_dim,
            config.hidden_dim,
            config.hidden_dim,
            layers=3,
            dropout=config.dropout,
        )
        self.layers = nn.ModuleList(
            [
                SignatureAttentionLayer(
                    config.hidden_dim,
                    config.dropout,
                    config.edge_chunk_size,
                    config.edge_chunk_threshold,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.classifier = MLP(
            config.hidden_dim,
            config.hidden_dim,
            num_classes,
            layers=3,
            dropout=config.dropout,
        )

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        signature = cached_graph_signature(data, self.config.edge_chunk_size)
        h = self.encoder(signature)
        for layer in self.layers:
            h, _ = layer(h, data.edge_index)
        return {"logits": self.classifier(h), "embedding": h, "signature": signature}
