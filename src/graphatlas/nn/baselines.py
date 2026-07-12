from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from graphatlas.data import GraphData
from .functional import MLP, normalized_aggregate


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.encoder = MLP(input_dim, hidden_dim, hidden_dim, layers=2, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        embedding = self.encoder(data.x)
        return {"logits": self.classifier(embedding), "embedding": embedding}


class GraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return normalized_aggregate(self.linear(x), edge_index)


class GCNBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, layers: int, dropout: float):
        super().__init__()
        depth = max(2, layers)
        dims = [input_dim] + [hidden_dim] * (depth - 1) + [num_classes]
        self.layers = nn.ModuleList(
            [GraphConvolution(dims[index], dims[index + 1]) for index in range(len(dims) - 1)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        h = data.x
        embedding = h
        for index, layer in enumerate(self.layers):
            h = layer(h, data.edge_index)
            if index < len(self.layers) - 1:
                h = self.dropout(F.gelu(h))
                embedding = h
        return {"logits": h, "embedding": embedding}


class ResidualGCNBaseline(nn.Module):
    """Residual GCN with layer normalization and skip connections."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, layers: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(max(1, layers))])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.layers])
        self.dropout = float(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        h = F.gelu(self.input(data.x))
        for linear, norm in zip(self.layers, self.norms, strict=True):
            message = normalized_aggregate(h, data.edge_index)
            update = F.dropout(F.gelu(linear(message)), p=self.dropout, training=self.training)
            h = norm(h + update)
        return {"logits": self.classifier(h), "embedding": h}


class SGCBaseline(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hops: int = 2):
        super().__init__()
        self.hops = hops
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        h = data.x
        for _ in range(self.hops):
            h = normalized_aggregate(h, data.edge_index)
        return {"logits": self.linear(h), "embedding": h}


class APPNPBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        steps: int,
        teleport: float,
        dropout: float,
    ):
        super().__init__()
        self.predictor = MLP(input_dim, hidden_dim, num_classes, layers=3, dropout=dropout)
        self.steps = max(1, int(steps))
        self.teleport = float(teleport)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        initial = self.predictor(data.x)
        propagated = initial
        for _ in range(self.steps):
            propagated = (
                (1.0 - self.teleport) * normalized_aggregate(propagated, data.edge_index)
                + self.teleport * initial
            )
        return {"logits": propagated, "embedding": propagated}


class GPRGNNBaseline(nn.Module):
    """Generalized PageRank propagation with trainable signed coefficients."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        steps: int,
        teleport: float,
        dropout: float,
    ):
        super().__init__()
        self.encoder = MLP(input_dim, hidden_dim, hidden_dim, layers=2, dropout=dropout)
        self.steps = max(1, int(steps))
        alpha = float(teleport)
        coefficients = torch.tensor(
            [alpha * (1.0 - alpha) ** k for k in range(self.steps)]
            + [(1.0 - alpha) ** self.steps],
            dtype=torch.float32,
        )
        self.coefficients = nn.Parameter(coefficients)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        current = self.encoder(data.x)
        states = [current]
        for _ in range(self.steps):
            current = normalized_aggregate(current, data.edge_index)
            states.append(current)
        embedding = sum(weight * state for weight, state in zip(self.coefficients, states, strict=True))
        return {
            "logits": self.classifier(embedding),
            "embedding": embedding,
            "propagation_coefficients": self.coefficients,
        }


class MixHopBaseline(nn.Module):
    """Sparse MixHop-style 0/1/2-hop feature mixing."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, layers: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.mix = nn.ModuleList([nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(max(1, layers))])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.mix])
        self.dropout = float(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        h = F.gelu(self.input(data.x))
        for linear, norm in zip(self.mix, self.norms, strict=True):
            one = normalized_aggregate(h, data.edge_index, add_self_loops=False)
            two = normalized_aggregate(one, data.edge_index, add_self_loops=False)
            update = linear(torch.cat([h, one, two], dim=-1))
            h = norm(h + F.dropout(F.gelu(update), p=self.dropout, training=self.training))
        return {"logits": self.classifier(h), "embedding": h}


class H2GCNBaseline(nn.Module):
    """Sparse H2GCN-style ego, one-hop, and two-hop separation."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, layers: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(max(1, layers))])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.layers])
        self.dropout = float(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        h = F.relu(self.input(data.x))
        for linear, norm in zip(self.layers, self.norms, strict=True):
            one = normalized_aggregate(h, data.edge_index, add_self_loops=False)
            two = normalized_aggregate(one, data.edge_index, add_self_loops=False)
            candidate = F.relu(linear(torch.cat([h, one, two], dim=-1)))
            h = norm(h + F.dropout(candidate, p=self.dropout, training=self.training))
        return {"logits": self.classifier(h), "embedding": h}


class LINKXBaseline(nn.Module):
    """Memory-safe LINKX-style model with sparse adjacency input."""

    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.adjacency_weight = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.adjacency_weight)
        self.adjacency_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, layers=2, dropout=dropout)
        self.feature_mlp = MLP(input_dim, hidden_dim, hidden_dim, layers=2, dropout=dropout)
        self.combine_hidden = MLP(2 * hidden_dim, hidden_dim, hidden_dim, layers=2, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        src, dst = data.edge_index
        adjacency_hidden = torch.zeros_like(self.adjacency_weight)
        adjacency_hidden.index_add_(0, dst, self.adjacency_weight[src])
        adjacency_hidden = self.adjacency_mlp(adjacency_hidden)
        feature_hidden = self.feature_mlp(data.x)
        embedding = self.combine_hidden(torch.cat([adjacency_hidden, feature_hidden], dim=-1))
        return {"logits": self.classifier(embedding), "embedding": embedding}


class GeometryMoEBaseline(nn.Module):
    """Controlled post-hoc expert-fusion proxy with matched GCN encoders."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_experts: int, layers: int, dropout: float):
        super().__init__()
        self.experts = nn.ModuleList(
            [GCNBaseline(input_dim, hidden_dim, num_classes, layers, dropout) for _ in range(num_experts)]
        )
        self.gate = MLP(input_dim, hidden_dim, num_experts, layers=2, dropout=dropout)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        outputs = [expert(data) for expert in self.experts]
        expert_logits = torch.stack([output["logits"] for output in outputs], dim=1)
        expert_embeddings = torch.stack([output["embedding"] for output in outputs], dim=1)
        gate = torch.softmax(self.gate(data.x), dim=-1)
        logits = (expert_logits * gate.unsqueeze(-1)).sum(dim=1)
        embedding = (expert_embeddings * gate.unsqueeze(-1)).sum(dim=1)
        return {
            "logits": logits,
            "embedding": embedding,
            "membership": gate,
            "expert_logits": expert_logits,
        }

class LinearBaseline(nn.Module):
    """Logistic-regression baseline on node features."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        return {"logits": self.classifier(data.x), "embedding": data.x}


class LINKBaseline(nn.Module):
    """Adjacency-only baseline implemented without a dense adjacency matrix."""

    def __init__(self, num_nodes: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.source_embedding = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.source_embedding)
        self.predictor = MLP(hidden_dim, hidden_dim, num_classes, layers=2, dropout=dropout)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        src, dst = data.edge_index
        embedding = torch.zeros_like(self.source_embedding)
        embedding.index_add_(0, dst, self.source_embedding[src])
        return {"logits": self.predictor(embedding), "embedding": embedding}


def _edge_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Stable per-destination softmax for scores shaped [E,H]."""
    index = dst[:, None].expand_as(scores)
    maximum = torch.full((num_nodes, scores.shape[1]), -torch.inf, device=scores.device, dtype=scores.dtype)
    maximum.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    exp = torch.exp(scores - maximum[dst])
    denominator = torch.zeros_like(maximum)
    denominator.scatter_add_(0, index, exp)
    return exp / denominator[dst].clamp_min(1e-12)


class SparseGATLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, heads: int, dropout: float, concat: bool = True):
        super().__init__()
        self.heads = max(1, int(heads))
        self.concat = concat
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, self.heads * output_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(self.heads, output_dim))
        self.att_dst = nn.Parameter(torch.empty(self.heads, output_dim))
        self.bias = nn.Parameter(torch.zeros(self.heads * output_dim if concat else output_dim))
        self.dropout = float(dropout)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        h = self.linear(x).view(x.shape[0], self.heads, self.output_dim)
        score = (h[src] * self.att_src).sum(-1) + (h[dst] * self.att_dst).sum(-1)
        score = F.leaky_relu(score, negative_slope=0.2)
        alpha = _edge_softmax(score, dst, x.shape[0])
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        message = h[src] * alpha[..., None]
        out = torch.zeros(x.shape[0], self.heads, self.output_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, message)
        if self.concat:
            return out.reshape(x.shape[0], self.heads * self.output_dim) + self.bias
        return out.mean(dim=1) + self.bias


class GATBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, layers: int, heads: int, dropout: float):
        super().__init__()
        heads = max(1, heads)
        per_head = max(1, hidden_dim // heads)
        self.input = SparseGATLayer(input_dim, per_head, heads, dropout, concat=True)
        actual_hidden = per_head * heads
        self.hidden = nn.ModuleList(
            [SparseGATLayer(actual_hidden, per_head, heads, dropout, concat=True) for _ in range(max(0, layers - 2))]
        )
        self.output = SparseGATLayer(actual_hidden, num_classes, 1, dropout, concat=False)
        self.dropout = float(dropout)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        h = F.elu(self.input(data.x, data.edge_index))
        for layer in self.hidden:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.elu(layer(h, data.edge_index))
        embedding = h
        logits = self.output(F.dropout(h, p=self.dropout, training=self.training), data.edge_index)
        return {"logits": logits, "embedding": embedding}


class GCNIIBaseline(nn.Module):
    """GCNII reference implementation with initial residual and identity mapping."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, layers: int, dropout: float, alpha: float, theta: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(max(1, layers))])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = float(dropout)
        self.alpha = float(alpha)
        self.theta = float(theta)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        h0 = F.relu(self.input(data.x))
        h = h0
        for index, linear in enumerate(self.layers, start=1):
            propagated = normalized_aggregate(h, data.edge_index)
            mixed = (1.0 - self.alpha) * propagated + self.alpha * h0
            beta = torch.log(torch.tensor(self.theta / index + 1.0, device=h.device, dtype=h.dtype))
            h = F.relu((1.0 - beta) * mixed + beta * linear(mixed))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return {"logits": self.classifier(h), "embedding": h}


class FAGCNLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.dropout = float(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        gate = torch.tanh(self.attention(torch.cat([h[src], h[dst]], dim=-1))).squeeze(-1)
        gate = F.dropout(gate, p=self.dropout, training=self.training)
        degree = torch.bincount(dst, minlength=h.shape[0]).to(h.dtype).clamp_min(1.0)
        message = h[src] * gate[:, None] / degree[dst, None]
        out = torch.zeros_like(h)
        out.index_add_(0, dst, message)
        return out


class FAGCNBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, layers: int, dropout: float, alpha: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([FAGCNLayer(hidden_dim, dropout) for _ in range(max(1, layers))])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.layers])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.alpha = float(alpha)
        self.dropout = float(dropout)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        h0 = F.relu(self.input(data.x))
        h = h0
        for layer, norm in zip(self.layers, self.norms, strict=True):
            update = layer(h, data.edge_index)
            h = norm(self.alpha * h0 + (1.0 - self.alpha) * update)
            h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        return {"logits": self.classifier(h), "embedding": h}


class ACMGCNBaseline(nn.Module):
    """Adaptive channel mixing over low-pass, high-pass and identity channels."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, layers: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.low = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(max(1, layers))])
        self.high = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(max(1, layers))])
        self.identity = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(max(1, layers))])
        self.gates = nn.ModuleList([nn.Linear(3 * hidden_dim, 3) for _ in range(max(1, layers))])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(max(1, layers))])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = float(dropout)

    def forward(self, data: GraphData, **_: object) -> dict[str, torch.Tensor]:
        h = F.relu(self.input(data.x))
        for low_layer, high_layer, identity_layer, gate_layer, norm in zip(
            self.low, self.high, self.identity, self.gates, self.norms, strict=True
        ):
            low_raw = normalized_aggregate(h, data.edge_index)
            high_raw = h - low_raw
            low = low_layer(low_raw)
            high = high_layer(high_raw)
            identity = identity_layer(h)
            gate = torch.softmax(gate_layer(torch.cat([low, high, identity], dim=-1)), dim=-1)
            update = gate[:, 0:1] * low + gate[:, 1:2] * high + gate[:, 2:3] * identity
            h = norm(h + F.dropout(F.gelu(update), p=self.dropout, training=self.training))
        return {"logits": self.classifier(h), "embedding": h}
