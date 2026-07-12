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
        transport_mode: str = "original",
        transportability_beta: float = 1.0,
        transportability_eps: float = 1e-8,
        transportability_stop_gradient: bool = True,
        edge_chunk_size: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.chart_dim = chart_dim
        self.channels = channels
        self.num_charts = num_charts
        self.mode = mode
        self.transport_mode = transport_mode
        self.transportability_beta = float(transportability_beta)
        self.transportability_eps = float(transportability_eps)
        self.transportability_stop_gradient = bool(transportability_stop_gradient)
        self.edge_chunk_size = int(edge_chunk_size)
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
        decoder_jacobians: torch.Tensor | None = None,
        decoder_pinv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mixed = torch.einsum("nkdc,ce->nkde", tangent, self.message_mix)
        queries = self.query(h)
        keys = self.key(h)

        certificate_diagnostics: dict[str, torch.Tensor] = {}
        if self.mode == "transport" and self.transport_mode == "original":
            source_observation = torch.stack([push(chart, mixed[:, chart]) for chart in range(self.num_charts)], dim=1)
            source_observation = (source_observation * membership[:, :, None, None]).sum(dim=1)
            aggregated_observation, alpha = attention_aggregate(source_observation, edge_index, queries, keys)
            pulled = torch.stack([pull(chart, aggregated_observation) for chart in range(self.num_charts)], dim=1)
        elif self.mode == "transport" and self.transport_mode == "min_distortion":
            if decoder_pinv is None:
                raise RuntimeError("min_distortion transport requires cached decoder_pinv")
            source_observation = torch.stack([push(chart, mixed[:, chart]) for chart in range(self.num_charts)], dim=1)
            source_observation = (source_observation * membership[:, :, None, None]).sum(dim=1)
            aggregated_observation, alpha = attention_aggregate(source_observation, edge_index, queries, keys)
            # U_i^k = (J_i^k)^dagger W_i, [N,K,d,p] x [N,p,C] -> [N,K,d,C].
            pulled = torch.einsum("nkdp,npc->nkdc", decoder_pinv, aggregated_observation)
        elif self.mode == "transport" and self.transport_mode == "certified":
            if decoder_jacobians is None or decoder_pinv is None:
                raise RuntimeError("certified transport requires cached decoder_jacobians and decoder_pinv")
            src, dst = edge_index
            source_ambient = torch.stack([push(chart, mixed[:, chart]) for chart in range(self.num_charts)], dim=1)
            semantic_values = (source_ambient * membership[:, :, None, None]).sum(dim=1)
            _, alpha = attention_aggregate(semantic_values, edge_index, queries, keys)

            pulled = torch.zeros(
                tangent.shape[0], self.num_charts, self.chart_dim, self.channels,
                device=tangent.device, dtype=tangent.dtype,
            )
            statistics = {
                "count": 0,
                "q_sum": tangent.new_zeros(()),
                "q_square_sum": tangent.new_zeros(()),
                "q_min": tangent.new_tensor(float("inf")),
                "q_max": tangent.new_tensor(float("-inf")),
                "delta_sum": tangent.new_zeros(()),
                "below": tangent.new_zeros(()),
                "above": tangent.new_zeros(()),
            }
            chunk_size = max(1, self.edge_chunk_size)
            for start in range(0, src.numel(), chunk_size):
                stop = min(start + chunk_size, src.numel())
                chunk_src, chunk_dst = src[start:stop], dst[start:stop]
                edge_source_ambient = source_ambient[chunk_src]  # [B,L,p,C]
                edge_target_pinv = decoder_pinv[chunk_dst]  # [B,K,d,p]
                edge_target_jacobian = decoder_jacobians[chunk_dst]  # [B,K,p,d]
                # U_ij^(k,l) = (J_i^k)^dagger W_j^l: [B,K,L,d,C].
                u_star = torch.einsum("bkdp,blpc->bkldc", edge_target_pinv, edge_source_ambient)
                # W_proj = J_i^k U_ij^(k,l): [B,K,L,p,C].
                projected = torch.einsum("bkpd,bkldc->bklpc", edge_target_jacobian, u_star)
                source = edge_source_ambient[:, None]
                residual = source - projected
                source_energy = source.square().sum(dim=(-2, -1)).expand(-1, self.num_charts, -1)
                projected_energy = projected.square().sum(dim=(-2, -1))
                residual_energy = residual.square().sum(dim=(-2, -1))
                denominator = source_energy.clamp_min(self.transportability_eps)
                nonzero = source_energy > self.transportability_eps
                q = torch.where(nonzero, projected_energy / denominator, torch.zeros_like(projected_energy)).clamp(0.0, 1.0)
                delta = torch.where(nonzero, residual_energy / denominator, torch.zeros_like(residual_energy)).clamp(0.0, 1.0)

                source_membership = membership[chunk_src]
                q_for_routing = q.detach() if self.transportability_stop_gradient else q
                finite_log_membership = source_membership.clamp_min(
                    torch.finfo(source_membership.dtype).tiny
                ).log()
                log_membership = torch.where(
                    source_membership > 0,
                    finite_log_membership,
                    torch.full_like(source_membership, -torch.inf),
                )
                routing = torch.softmax(
                    log_membership[:, None, :] + self.transportability_beta * q_for_routing,
                    dim=-1,
                )
                edge_target_message = (routing[:, :, :, None, None] * u_star).sum(dim=2)
                pulled.index_add_(
                    0, chunk_dst,
                    edge_target_message * alpha[start:stop, None, None, None],
                )

                valid = (source_membership[:, None, :] > 0) & (membership[chunk_dst, :, None] > 0)
                valid_q = q[valid].detach()
                valid_delta = delta[valid].detach()
                if valid_q.numel():
                    statistics["count"] += valid_q.numel()
                    statistics["q_sum"] += valid_q.sum()
                    statistics["q_square_sum"] += valid_q.square().sum()
                    statistics["q_min"] = torch.minimum(statistics["q_min"], valid_q.min())
                    statistics["q_max"] = torch.maximum(statistics["q_max"], valid_q.max())
                    statistics["delta_sum"] += valid_delta.sum()
                    statistics["below"] += (valid_q < 0.25).to(valid_q.dtype).sum()
                    statistics["above"] += (valid_q > 0.75).to(valid_q.dtype).sum()

            count = max(int(statistics["count"]), 1)
            q_mean = statistics["q_sum"] / count
            q_variance = (statistics["q_square_sum"] / count - q_mean.square()).clamp_min(0.0)
            certificate_diagnostics = {
                "transportability_mean": q_mean,
                "transportability_std": q_variance.sqrt(),
                "transportability_min": statistics["q_min"] if statistics["count"] else q_mean,
                "transportability_max": statistics["q_max"] if statistics["count"] else q_mean,
                "distortion_mean": statistics["delta_sum"] / count,
                "fraction_q_below_025": statistics["below"] / count,
                "fraction_q_above_075": statistics["above"] / count,
            }
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
        return output, {
            "attention": alpha,
            "preactivation_pushed": pushed,
            "norms": norms,
            **certificate_diagnostics,
        }
