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
        certified_routing_mode: str = "certificate",
        routing_control_seed: int = 1729,
        routing_control_max_edges: int = 200000,
        coordinate_activation: str = "none",
        coordinate_left_linear: bool = False,
        edge_chunk_size: int = 2048,
        edge_chunk_threshold: int = 2_000_000,
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
        self.certified_routing_mode = certified_routing_mode
        self.routing_control_seed = int(routing_control_seed)
        self.routing_control_max_edges = int(routing_control_max_edges)
        self.coordinate_activation = coordinate_activation
        self.edge_chunk_size = int(edge_chunk_size)
        self.edge_chunk_threshold = int(edge_chunk_threshold)
        self.dropout_probability = float(dropout)
        self.self_mix = nn.Parameter(torch.eye(channels) + 0.02 * torch.randn(channels, channels))
        self.message_mix = nn.Parameter(torch.eye(channels) + 0.02 * torch.randn(channels, channels))
        self.residual_mix = nn.Parameter(0.1 * torch.eye(channels))
        if coordinate_left_linear:
            self.coordinate_left_linear = nn.Parameter(torch.eye(chart_dim) + 1e-3 * torch.randn(chart_dim, chart_dim))
        else:
            self.register_parameter("coordinate_left_linear", None)
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

    def _routing(
        self,
        q: torch.Tensor,
        membership: torch.Tensor,
        edge_offset: int,
        q_shuffle_edge: torch.Tensor | None = None,
        zero_energy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Source-chart routing only; all controls leave messages and attention intact."""
        mode = self.certified_routing_mode
        active = membership > 0
        log_membership = torch.where(
            active,
            membership.clamp_min(torch.finfo(membership.dtype).tiny).log(),
            torch.full_like(membership, -torch.inf),
        )
        q_used = q_shuffle_edge if q_shuffle_edge is not None else q
        if mode == "membership_only":
            logits = log_membership[:, None, :]
        elif mode == "q_shuffle_source":
            # Per-edge/target cyclic source-chart permutation; global indices make it chunk-invariant.
            edge_ids = torch.arange(edge_offset, edge_offset + q.shape[0], device=q.device)[:, None]
            target_ids = torch.arange(q.shape[1], device=q.device)[None, :]
            shift = (edge_ids + target_ids + self.routing_control_seed).remainder(q.shape[-1])
            source_ids = torch.arange(q.shape[-1], device=q.device)[None, None, :]
            q_used = q.gather(-1, (source_ids + shift[:, :, None]).remainder(q.shape[-1]))
            logits = log_membership[:, None, :] + self.transportability_beta * q_used
        elif mode == "random":
            edge_ids = torch.arange(edge_offset, edge_offset + q.shape[0], device=q.device, dtype=q.dtype)[:, None, None]
            target_ids = torch.arange(q.shape[1], device=q.device, dtype=q.dtype)[None, :, None]
            source_ids = torch.arange(q.shape[-1], device=q.device, dtype=q.dtype)[None, None, :]
            # Stateless deterministic pseudo-random scores, independent of chunking/global RNG.
            q_used = torch.frac(torch.sin((edge_ids + 1) * 12.9898 + (target_ids + 1) * 78.233 + (source_ids + 1) * 37.719 + self.routing_control_seed) * 43758.5453)
            logits = log_membership[:, None, :] + self.transportability_beta * q_used
        elif mode in {"q_reference_max", "oracle_max_q"}:
            masked = q.masked_fill(~active[:, None, :], -torch.inf)
            chosen = masked.argmax(dim=-1)
            routing = torch.zeros_like(q).scatter_(-1, chosen[..., None], 1.0)
            if zero_energy is not None:
                fallback = active.to(q.dtype)
                fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp_min(1.0)
                routing = torch.where(zero_energy[..., None], fallback[:, None, :], routing)
            return routing
        else:  # certificate and q_shuffle_edge
            if self.transportability_stop_gradient:
                q_used = q_used.detach()
            logits = log_membership[:, None, :] + self.transportability_beta * q_used
        return torch.softmax(logits, dim=-1)

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
        detailed_transport_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mixed = torch.einsum("nkdc,ce->nkde", tangent, self.message_mix)
        queries = self.query(h)
        keys = self.key(h)

        certificate_diagnostics: dict[str, torch.Tensor] = {}
        if self.mode == "transport" and self.transport_mode == "original":
            source_observation = torch.stack([push(chart, mixed[:, chart]) for chart in range(self.num_charts)], dim=1)
            source_observation = (source_observation * membership[:, :, None, None]).sum(dim=1)
            aggregated_observation, alpha = attention_aggregate(source_observation, edge_index, queries, keys, self.edge_chunk_size, self.edge_chunk_threshold)
            pulled = torch.stack([pull(chart, aggregated_observation) for chart in range(self.num_charts)], dim=1)
        elif self.mode == "transport" and self.transport_mode == "min_distortion":
            if decoder_pinv is None:
                raise RuntimeError("min_distortion transport requires cached decoder_pinv")
            source_observation = torch.stack([push(chart, mixed[:, chart]) for chart in range(self.num_charts)], dim=1)
            source_observation = (source_observation * membership[:, :, None, None]).sum(dim=1)
            aggregated_observation, alpha = attention_aggregate(source_observation, edge_index, queries, keys, self.edge_chunk_size, self.edge_chunk_threshold)
            # U_i^k = (J_i^k)^dagger W_i, [N,K,d,p] x [N,p,C] -> [N,K,d,C].
            pulled = torch.einsum("nkdp,npc->nkdc", decoder_pinv, aggregated_observation)
        elif self.mode == "transport" and self.transport_mode == "certified":
            if decoder_jacobians is None or decoder_pinv is None:
                raise RuntimeError("certified transport requires cached decoder_jacobians and decoder_pinv")
            src, dst = edge_index
            source_ambient = torch.stack([push(chart, mixed[:, chart]) for chart in range(self.num_charts)], dim=1)
            semantic_values = (source_ambient * membership[:, :, None, None]).sum(dim=1)
            aggregated_observation, alpha = attention_aggregate(semantic_values, edge_index, queries, keys, self.edge_chunk_size, self.edge_chunk_threshold)

            q_shuffle_edge_all = None
            if self.certified_routing_mode == "q_shuffle_edge":
                if src.numel() > self.routing_control_max_edges:
                    raise RuntimeError(
                        "q_shuffle_edge requires num_edges <= routing_control_max_edges; "
                        f"got {src.numel()} > {self.routing_control_max_edges}"
                    )
                # This is intentionally the only control allowed to retain edge-pair q values.
                q_chunks: list[torch.Tensor] = []
                for start in range(0, src.numel(), max(1, self.edge_chunk_size)):
                    stop = min(start + max(1, self.edge_chunk_size), src.numel())
                    edge_source = source_ambient[src[start:stop]]
                    target_pinv = decoder_pinv[dst[start:stop]]
                    target_jacobian = decoder_jacobians[dst[start:stop]]
                    transported = torch.einsum("bkdp,blpc->bkldc", target_pinv, edge_source)
                    projected = torch.einsum("bkpd,bkldc->bklpc", target_jacobian, transported)
                    source_energy = edge_source[:, None].square().sum(dim=(-2, -1)).expand(-1, self.num_charts, -1)
                    denominator = source_energy.clamp_min(self.transportability_eps)
                    q_chunks.append(torch.where(
                        source_energy > self.transportability_eps,
                        projected.square().sum(dim=(-2, -1)) / denominator,
                        torch.zeros_like(source_energy),
                    ).clamp(0.0, 1.0).detach())
                q_all = torch.cat(q_chunks, dim=0)
                generator = torch.Generator(device="cpu").manual_seed(self.routing_control_seed)
                permutation = torch.randperm(src.numel(), generator=generator, device="cpu").to(src.device)
                q_shuffle_edge_all = q_all[permutation]

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
                "closure_sum": tangent.new_zeros(()),
                "risk_sum": tangent.new_zeros(()),
            }
            detailed = {
                "active_sum": tangent.new_zeros(()),
                "edge_count": 0,
                "multiple": tangent.new_zeros(()),
                "kl_sum": tangent.new_zeros(()),
                "kl_count": 0,
                "spreads": [],
                "entropy_sum": tangent.new_zeros(()),
                "tv_sum": tangent.new_zeros(()),
                "target_active_sum": tangent.new_zeros(()),
                "target_multiple": tangent.new_zeros(()),
                "opportunity_sum": tangent.new_zeros(()),
                "cross_chart_sum": tangent.new_zeros(()),
                "q_reference_agreement_sum": tangent.new_zeros(()),
                "node_weight": None,
                "node_cross": None,
                "node_spread": None,
                "node_risk": None,
            }
            if detailed_transport_diagnostics:
                for key in ("node_weight", "node_cross", "node_spread", "node_risk"):
                    detailed[key] = torch.zeros(tangent.shape[0], device=tangent.device, dtype=tangent.dtype)
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
                routing = self._routing(
                    q, source_membership, start,
                    None if q_shuffle_edge_all is None else q_shuffle_edge_all[start:stop],
                    zero_energy=~nonzero.any(dim=-1),
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
                    statistics["closure_sum"] += (q[valid] + delta[valid] - 1.0).abs().detach().sum()
                    statistics["risk_sum"] += (1.0 - q[valid]).detach().sum()
                if detailed_transport_diagnostics:
                    active_mask = source_membership > 0
                    active_count = active_mask.sum(dim=-1)
                    detailed["active_sum"] += active_count.to(tangent.dtype).sum()
                    detailed["edge_count"] += active_count.numel()
                    detailed["multiple"] += (active_count > 1).to(tangent.dtype).sum()
                    base_routing = source_membership[:, None, :].expand_as(routing)
                    kl_terms = torch.where(
                        routing > 0,
                        routing * (
                            routing.clamp_min(self.transportability_eps).log()
                            - base_routing.clamp_min(self.transportability_eps).log()
                        ),
                        torch.zeros_like(routing),
                    )
                    detailed["kl_sum"] += kl_terms.sum(dim=-1).detach().sum()
                    detailed["kl_count"] += routing.shape[0] * routing.shape[1]
                    detailed["entropy_sum"] += (
                        -(routing * routing.clamp_min(self.transportability_eps).log()).sum(dim=-1)
                    ).detach().sum()
                    detailed["tv_sum"] += (0.5 * (routing - base_routing).abs().sum(dim=-1)).detach().sum()
                    target_count = (membership[chunk_dst] > 0).sum(dim=-1)
                    detailed["target_active_sum"] += target_count.to(tangent.dtype).sum()
                    detailed["target_multiple"] += (target_count > 1).to(tangent.dtype).sum()
                    target_weights = membership[chunk_dst]
                    cross_mass = 1.0 - (source_membership * target_weights).sum(dim=-1)
                    active_q = q.masked_fill(~active_mask[:, None, :], -torch.inf)
                    q_max_active = active_q.max(dim=-1).values
                    q_min_active = q.masked_fill(~active_mask[:, None, :], torch.inf).min(dim=-1).values
                    spread_per_target = torch.where(active_count[:, None] > 1, q_max_active - q_min_active, torch.zeros_like(q_max_active))
                    best_q = (q_max_active * target_weights).sum(dim=-1)
                    detailed["opportunity_sum"] += (cross_mass * (spread_per_target * target_weights).sum(dim=-1)).detach().sum()
                    detailed["cross_chart_sum"] += cross_mass.detach().sum()
                    reference_choice = active_q.argmax(dim=-1)
                    routed_choice = routing.argmax(dim=-1)
                    detailed["q_reference_agreement_sum"] += (reference_choice == routed_choice).to(tangent.dtype).detach().sum()
                    edge_weight = alpha[start:stop].detach()
                    detailed["node_weight"].index_add_(0, chunk_dst, edge_weight)
                    detailed["node_cross"].index_add_(0, chunk_dst, edge_weight * cross_mass.detach())
                    detailed["node_spread"].index_add_(0, chunk_dst, edge_weight * (spread_per_target * target_weights).sum(dim=-1).detach())
                    detailed["node_risk"].index_add_(0, chunk_dst, edge_weight * (1.0 - best_q).detach())
                    if (active_count > 1).any():
                        expanded_mask = active_mask[:, None, :].expand_as(q)
                        q_max = q.masked_fill(~expanded_mask, -torch.inf).max(dim=-1).values
                        q_min = q.masked_fill(~expanded_mask, torch.inf).min(dim=-1).values
                        spread = q_max - q_min
                        detailed["spreads"].append(spread[active_count > 1].detach())

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
                "q_delta_closure_error": statistics["closure_sum"] / count,
                "irreducible_transport_risk": statistics["risk_sum"] / count,
            }
            if detailed_transport_diagnostics:
                edge_count = max(int(detailed["edge_count"]), 1)
                kl_count = max(int(detailed["kl_count"]), 1)
                spread_values = (
                    torch.cat(detailed["spreads"])
                    if detailed["spreads"]
                    else tangent.new_zeros(1)
                )
                certificate_diagnostics.update({
                    "active_source_chart_count": detailed["active_sum"] / edge_count,
                    "fraction_edges_with_multiple_source_charts": detailed["multiple"] / edge_count,
                    "q_route_spread_mean": spread_values.mean(),
                    "q_route_spread_p90": torch.quantile(spread_values, 0.90),
                    "routing_kl_from_membership": detailed["kl_sum"] / kl_count,
                    "routing_total_variation_from_membership": detailed["tv_sum"] / kl_count,
                    "routing_entropy_mean": detailed["entropy_sum"] / kl_count,
                    "active_target_chart_count": detailed["target_active_sum"] / edge_count,
                    "fraction_edges_with_multiple_target_charts": detailed["target_multiple"] / edge_count,
                    "q_route_spread_median": spread_values.median(),
                    "q_route_spread_max": spread_values.max(),
                    "routing_opportunity": detailed["opportunity_sum"] / edge_count,
                    "cross_chart_mass": detailed["cross_chart_sum"] / edge_count,
                    "q_reference_agreement": detailed["q_reference_agreement_sum"] / kl_count,
                    "node_cross_chart_mass": detailed["node_cross"] / detailed["node_weight"].clamp_min(self.transportability_eps),
                    "node_q_spread": detailed["node_spread"] / detailed["node_weight"].clamp_min(self.transportability_eps),
                    "node_transport_risk": detailed["node_risk"] / detailed["node_weight"].clamp_min(self.transportability_eps),
                })
            if self.certified_routing_mode == "membership_only" or self.transportability_beta == 0.0:
                # Preserve the exact minimum-distortion accumulation order for
                # the beta-zero equivalence diagnostic and ablation.
                pulled = torch.einsum("nkdp,npc->nkdc", decoder_pinv, aggregated_observation)
        elif self.mode == "no_transport":
            source_coordinates = (mixed * membership[:, :, None, None]).sum(dim=1)
            aggregated_coordinates, alpha = attention_aggregate(source_coordinates, edge_index, queries, keys, self.edge_chunk_size, self.edge_chunk_threshold)
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
                aggregated, alpha = attention_aggregate(value, edge_index, queries, keys, self.edge_chunk_size, self.edge_chunk_threshold)
                target_values.append(aggregated)
            pulled = torch.stack(target_values, dim=1)
        else:
            raise ValueError(f"Unknown atlas layer mode: {self.mode}")

        self_term = torch.einsum("nkdc,ce->nkde", tangent, self.self_mix)
        preactivation = self_term + pulled
        if self.coordinate_left_linear is not None:
            preactivation = torch.einsum("nkdc,de->nkec", preactivation, self.coordinate_left_linear)
        if self.coordinate_activation == "relu":
            preactivation = F.relu(preactivation)
        elif self.coordinate_activation == "gelu":
            preactivation = F.gelu(preactivation)
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
