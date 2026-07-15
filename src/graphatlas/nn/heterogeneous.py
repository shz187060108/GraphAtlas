from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from graphatlas.config import ModelConfig
from graphatlas.data import HeteroGraphData


def _key(parts: tuple[str, str, str]) -> str:
    return "___".join(parts).replace(".", "_")


class TypedHeteroNetwork(nn.Module):
    """Type adapters plus relation-aware invariant message passing."""

    def __init__(self, data: HeteroGraphData, num_classes: int, config: ModelConfig, mode: str):
        super().__init__()
        self.config, self.mode, self.task_entity = config, mode, data.task_entity
        self.adapters = nn.ModuleDict()
        self.type_tokens = nn.ParameterDict()
        for node_type in data.node_types:
            x = data.x_dict[node_type]
            width = int(x.shape[1]) if x is not None else 2
            self.adapters[node_type] = nn.Linear(width, config.hidden_dim)
            self.type_tokens[node_type] = nn.Parameter(torch.zeros(config.hidden_dim))
        self.relation_gates = nn.ParameterDict({_key(edge_type): nn.Parameter(torch.zeros(())) for edge_type in data.edge_types})
        self.update = nn.Sequential(nn.Linear(2 * config.hidden_dim, config.hidden_dim), nn.GELU(), nn.Dropout(config.dropout))
        self.classifier = nn.Linear(config.hidden_dim, num_classes)

    def _features(self, data: HeteroGraphData, node_type: str, device: torch.device) -> torch.Tensor:
        value = data.x_dict[node_type]
        if value is None:
            degree = torch.zeros(data.num_nodes_dict[node_type], 2, device=device)
            for (source_type, _, target_type), edge_index in data.edge_index_dict.items():
                edge_index = edge_index.to(device)
                if source_type == node_type:
                    degree[:, 0].index_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=device))
                if target_type == node_type:
                    degree[:, 1].index_add_(0, edge_index[1], torch.ones(edge_index.shape[1], device=device))
            value = torch.log1p(degree)
        else:
            value = value.to(device)
        return self.adapters[node_type](value) + self.type_tokens[node_type]

    def forward(self, data: HeteroGraphData) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        device = next(self.parameters()).device
        hidden = {node_type: F.gelu(self._features(data, node_type, device)) for node_type in data.node_types}
        if self.mode != "mlp_typed":
            aggregate = {node_type: torch.zeros_like(value) for node_type, value in hidden.items()}
            counts = {node_type: value.new_zeros(value.shape[0], 1) for node_type, value in hidden.items()}
            for edge_type, edge_index_cpu in data.edge_index_dict.items():
                source_type, _, target_type = edge_type
                source, target = edge_index_cpu.to(device)
                gate = torch.sigmoid(self.relation_gates[_key(edge_type)])
                message = hidden[source_type][source] * gate
                aggregate[target_type].index_add_(0, target, message)
                counts[target_type].index_add_(0, target, torch.ones(target.shape[0], 1, device=device))
            hidden = {
                node_type: self.update(torch.cat([value, aggregate[node_type] / counts[node_type].clamp_min(1)], dim=-1))
                for node_type, value in hidden.items()
            }
        target = hidden[self.task_entity]
        return {"logits": self.classifier(target), "embedding": target, "layer_diagnostics": []}


class HeteroAtlasTransportNetwork(TypedHeteroNetwork):
    """Relation-aware ATN with decoder-range representability routing."""

    def __init__(self, data: HeteroGraphData, num_classes: int, config: ModelConfig):
        super().__init__(data, num_classes, config, mode="atn")
        k, p, d, c = config.num_charts, config.observation_dim, config.chart_dim, config.vector_channels
        self.membership = nn.ModuleDict({node_type: nn.Linear(config.hidden_dim, k) for node_type in data.node_types})
        self.vector_init = nn.ModuleDict({node_type: nn.Linear(config.hidden_dim, k * d * c) for node_type in data.node_types})
        self.decoders = nn.ParameterDict({
            node_type: nn.Parameter(torch.randn(k, p, d) / math.sqrt(max(1, d))) for node_type in data.node_types
        })
        self.relation_routing = nn.ParameterDict({
            _key(edge_type): nn.Parameter(torch.zeros(k, k)) for edge_type in data.edge_types
        })
        self.ambient_output = nn.Linear(p, config.hidden_dim)

    def forward(self, data: HeteroGraphData) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        device = next(self.parameters()).device
        hidden = {node_type: F.gelu(self._features(data, node_type, device)) for node_type in data.node_types}
        k, d, c = self.config.num_charts, self.config.chart_dim, self.config.vector_channels
        membership = {node_type: torch.softmax(self.membership[node_type](value), dim=-1) for node_type, value in hidden.items()}
        vectors = {
            node_type: self.vector_init[node_type](value).view(value.shape[0], k, d, c)
            for node_type, value in hidden.items()
        }
        ambient_acc = {node_type: value.new_zeros(value.shape[0], self.config.observation_dim, c) for node_type, value in hidden.items()}
        count = {node_type: value.new_zeros(value.shape[0], 1, 1) for node_type, value in hidden.items()}
        q_values, delta_values = [], []
        for edge_type, edge_index_cpu in data.edge_index_dict.items():
            source_type, _, target_type = edge_type
            source, target = edge_index_cpu.to(device)
            source_j = self.decoders[source_type]
            target_j = self.decoders[target_type]
            pinv_dtype = torch.float32 if target_j.dtype in {torch.float16, torch.bfloat16} else target_j.dtype
            target_pinv = torch.linalg.pinv(target_j.to(pinv_dtype), rtol=self.config.transportability_pinv_rtol).to(target_j.dtype)
            # J_l V_l -> [E,L,p,C].
            source_ambient = torch.einsum("lpd,eldc->elpc", source_j, vectors[source_type][source])
            # J_k^dagger W_l -> [E,K,L,d,C].
            transported = torch.einsum("kdp,elpc->ekldc", target_pinv, source_ambient)
            projected = torch.einsum("kpd,ekldc->eklpc", target_j, transported)
            source_view = source_ambient[:, None]
            energy = source_view.square().sum(dim=(-2, -1)).expand(-1, k, -1)
            projected_energy = projected.square().sum(dim=(-2, -1))
            residual_energy = (source_view - projected).square().sum(dim=(-2, -1))
            denominator = energy.clamp_min(self.config.transportability_eps)
            q = torch.where(energy > self.config.transportability_eps, projected_energy / denominator, torch.zeros_like(energy)).clamp(0, 1)
            delta = torch.where(energy > self.config.transportability_eps, residual_energy / denominator, torch.zeros_like(energy)).clamp(0, 1)
            q_route = q.detach() if self.config.transportability_stop_gradient else q
            logits = membership[source_type][source].clamp_min(1e-30).log()[:, None, :]
            logits = logits + self.config.transportability_beta * q_route + self.relation_routing[_key(edge_type)][None]
            routing = torch.softmax(logits, dim=-1)
            target_coordinates = (transported * routing[..., None, None]).sum(dim=2)
            target_ambient = torch.einsum("kpd,ekdc->ekpc", target_j, target_coordinates)
            target_ambient = (target_ambient * membership[target_type][target, :, None, None]).sum(dim=1)
            ambient_acc[target_type].index_add_(0, target, target_ambient)
            count[target_type].index_add_(0, target, torch.ones(target.shape[0], 1, 1, device=device))
            active = membership[source_type][source][:, None, :] > 0
            selected_q = q[active.expand_as(q)]
            selected_delta = delta[active.expand_as(delta)]
            # Neighbor sampling may retain a relation type with zero sampled
            # edges. Such a batch has no certificate population to summarize.
            if selected_q.numel() > 0:
                q_values.append(selected_q)
                delta_values.append(selected_delta)
        updated = {
            node_type: value + self.ambient_output((ambient_acc[node_type] / count[node_type].clamp_min(1)).mean(dim=-1))
            for node_type, value in hidden.items()
        }
        target = updated[self.task_entity]
        if q_values:
            q_all, delta_all = torch.cat(q_values), torch.cat(delta_values)
            diagnostics = [{
                "transportability_mean": q_all.mean(),
                "transportability_std": q_all.std(unbiased=False),
                "transportability_min": q_all.min(),
                "transportability_max": q_all.max(),
                "distortion_mean": delta_all.mean(),
                "fraction_q_below_025": (q_all < .25).float().mean(),
                "fraction_q_above_075": (q_all > .75).float().mean(),
            }]
        else:
            diagnostics = []
        return {"logits": self.classifier(target), "embedding": target, "membership": membership[self.task_entity], "layer_diagnostics": diagnostics}


def build_hetero_model(data: HeteroGraphData, config: ModelConfig) -> nn.Module:
    if config.name in {"atn", "graphatlas_typed"}:
        return HeteroAtlasTransportNetwork(data, data.num_classes, config)
    if config.name in {"mlp_typed", "gcn_homogeneous_projection", "rgcn", "hgt"}:
        return TypedHeteroNetwork(data, data.num_classes, config, mode=config.name)
    raise ValueError(f"Unsupported native heterogeneous model: {config.name}")
