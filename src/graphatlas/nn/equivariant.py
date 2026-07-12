from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F

from .functional import MLP, attention_aggregate


class AtlasEquivariantLayer(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        chart_dim: int,
        channels: int,
        num_charts: int,
        hidden_dim: int,
        mode: str = "transport",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.chart_dim = chart_dim
        self.channels = channels
        self.num_charts = num_charts
        self.mode = mode
        self.dropout_probability = float(dropout)
        self.self_mix = nn.Parameter(torch.eye(channels) + 0.02 * torch.randn(channels, channels))
        self.message_mix = nn.Parameter(torch.eye(channels) + 0.02 * torch.randn(channels, channels))
        self.residual_mix = nn.Parameter(0.1 * torch.eye(channels))
        self.query = nn.Linear(observation_dim, hidden_dim, bias=False)
        self.key = nn.Linear(observation_dim, hidden_dim, bias=False)
        self.gates = nn.ModuleList(
            [MLP(channels + observation_dim + 1, hidden_dim, channels, layers=2, dropout=dropout) for _ in range(num_charts)]
        )
        if mode == "free_transition":
            matrices = torch.eye(chart_dim).view(1, 1, chart_dim, chart_dim).repeat(num_charts, num_charts, 1, 1)
            self.free_transition = nn.Parameter(matrices + 0.02 * torch.randn_like(matrices))
        else:
            self.register_parameter("free_transition", None)

    def forward(
        self,
        tangent: torch.Tensor,
        membership: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        push: Callable[[int, torch.Tensor], torch.Tensor],
        pull: Callable[[int, torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mixed = torch.einsum("nkdc,ce->nkde", tangent, self.message_mix)
        queries = self.query(h)
        keys = self.key(h)

        if self.mode == "transport":
            source_observation = torch.stack([push(chart, mixed[:, chart]) for chart in range(self.num_charts)], dim=1)
            source_observation = (source_observation * membership[:, :, None, None]).sum(dim=1)
            aggregated_observation, alpha = attention_aggregate(source_observation, edge_index, queries, keys)
            pulled = torch.stack([pull(chart, aggregated_observation) for chart in range(self.num_charts)], dim=1)
        elif self.mode == "no_transport":
            source_coordinates = (mixed * membership[:, :, None, None]).sum(dim=1)
            aggregated_coordinates, alpha = attention_aggregate(source_coordinates, edge_index, queries, keys)
            pulled = aggregated_coordinates[:, None].expand(-1, self.num_charts, -1, -1)
        elif self.mode == "free_transition":
            target_values = []
            alpha = torch.empty(0, device=tangent.device)
            for target in range(self.num_charts):
                per_source = [
                    torch.einsum("de,nec->ndc", self.free_transition[target, source], mixed[:, source])
                    for source in range(self.num_charts)
                ]
                stacked = torch.stack(per_source, dim=1)
                value = (stacked * membership[:, :, None, None]).sum(dim=1)
                aggregated, alpha = attention_aggregate(value, edge_index, queries, keys)
                target_values.append(aggregated)
            pulled = torch.stack(target_values, dim=1)
        else:
            raise ValueError(f"Unknown atlas layer mode: {self.mode}")

        self_term = torch.einsum("nkdc,ce->nkde", tangent, self.self_mix)
        preactivation = self_term + pulled
        pushed = torch.stack([push(chart, preactivation[:, chart]) for chart in range(self.num_charts)], dim=1)
        norms = torch.sqrt(pushed.square().sum(dim=2) + 1e-10)
        updated = []
        for chart in range(self.num_charts):
            gate_input = torch.cat([norms[:, chart], h, membership[:, chart:chart + 1]], dim=-1)
            gate = torch.sigmoid(self.gates[chart](gate_input))
            # Drop scalar channel gates, not coordinate components. This preserves
            # chart equivariance even during training.
            gate = F.dropout(gate, p=self.dropout_probability, training=self.training)
            gated = preactivation[:, chart] * gate[:, None, :]
            residual = torch.einsum("ndc,ce->nde", tangent[:, chart], self.residual_mix)
            updated.append(gated + residual)
        output = torch.stack(updated, dim=1)
        return output, {"attention": alpha, "preactivation_pushed": pushed, "norms": norms}
