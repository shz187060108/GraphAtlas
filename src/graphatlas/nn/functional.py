from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.func import jvp, vmap


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits, _ = torch.sort(shifted, descending=True, dim=dim)
    cumulative = sorted_logits.cumsum(dim) - 1
    size = logits.shape[dim]
    range_values = torch.arange(1, size + 1, device=logits.device, dtype=logits.dtype)
    shape = [1] * logits.ndim
    shape[dim] = size
    range_values = range_values.view(shape)
    support = range_values * sorted_logits > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = cumulative.gather(dim, support_size - 1) / support_size.to(logits.dtype)
    return torch.clamp(shifted - tau, min=0.0)


def restrict_topk(probabilities: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= probabilities.shape[-1]:
        return probabilities
    _, indices = torch.topk(probabilities, k=k, dim=-1)
    mask = torch.zeros_like(probabilities).scatter_(-1, indices, 1.0)
    pruned = probabilities * mask
    return pruned / pruned.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def graph_signature(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    n = x.shape[0]
    degree = torch.zeros(n, device=x.device, dtype=x.dtype)
    degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
    neighbor_sum = torch.zeros_like(x)
    neighbor_sum.index_add_(0, dst, x[src])
    neighbor_mean = neighbor_sum / degree.clamp_min(1.0).unsqueeze(-1)
    diff = torch.linalg.vector_norm(x - neighbor_mean, dim=-1, keepdim=True)
    log_degree = torch.log1p(degree).unsqueeze(-1)
    log_degree = log_degree / log_degree.max().clamp_min(1.0)
    return torch.cat([x, log_degree, diff], dim=-1)


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    maxima = torch.full((num_segments,), -torch.inf, device=scores.device, dtype=scores.dtype)
    maxima.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    stabilized = scores - maxima[index]
    exp_scores = stabilized.exp()
    denominators = torch.zeros(num_segments, device=scores.device, dtype=scores.dtype)
    denominators.index_add_(0, index, exp_scores)
    return exp_scores / denominators[index].clamp_min(1e-12)


def normalized_aggregate(x: torch.Tensor, edge_index: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    n = x.shape[0]
    src, dst = edge_index
    if add_self_loops:
        nodes = torch.arange(n, device=x.device)
        src = torch.cat([src, nodes])
        dst = torch.cat([dst, nodes])
    degree = torch.zeros(n, device=x.device, dtype=x.dtype)
    degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
    weight = degree[src].clamp_min(1.0).rsqrt() * degree[dst].clamp_min(1.0).rsqrt()
    output = torch.zeros_like(x)
    expand = (slice(None),) + (None,) * (x.ndim - 1)
    output.index_add_(0, dst, x[src] * weight[expand])
    return output


def attention_aggregate(
    values: torch.Tensor,
    edge_index: torch.Tensor,
    queries: torch.Tensor,
    keys: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    src, dst = edge_index
    scores = (queries[dst] * keys[src]).sum(dim=-1) / math.sqrt(queries.shape[-1])
    alpha = segment_softmax(scores, dst, values.shape[0])
    output = torch.zeros_like(values)
    expand = (slice(None),) + (None,) * (values.ndim - 1)
    output.index_add_(0, dst, values[src] * alpha[expand])
    return output, alpha



def edge_dot_scores(embedding: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if embedding.ndim != 2:
        raise ValueError(f"Expected node embedding [N,D], got {tuple(embedding.shape)}")
    src, dst = edge_index
    return (embedding[src] * embedding[dst]).sum(dim=-1) / math.sqrt(max(embedding.shape[-1], 1))

def _chunk_ranges(length: int, chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size <= 0 or chunk_size >= length:
        return [(0, length)]
    return [(start, min(start + chunk_size, length)) for start in range(0, length, chunk_size)]


def channel_jvp(
    function: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    vectors: torch.Tensor,
    chunk_size: int = 0,
) -> torch.Tensor:
    """Apply a pointwise Jacobian to vector channels without materializing it.

    This generic fallback uses a channel-vmapped JVP and node chunking. The main
    GraphAtlas chart path uses cached analytic JVPs from ``SmoothMLP.linearize``.
    """
    outputs: list[torch.Tensor] = []
    for start, end in _chunk_ranges(x.shape[0], chunk_size):
        x_part = x[start:end]
        v_part = vectors[start:end]

        def one_channel(vector: torch.Tensor) -> torch.Tensor:
            return jvp(function, (x_part,), (vector,))[1]

        outputs.append(vmap(one_channel, in_dims=2, out_dims=2)(v_part))
    return torch.cat(outputs, dim=0)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int = 2,
        dropout: float = 0.0,
        smooth: bool = False,
    ):
        super().__init__()
        modules: list[nn.Module] = []
        dims = [input_dim] + [hidden_dim] * max(layers - 1, 0) + [output_dim]
        activation: type[nn.Module] = nn.Tanh if smooth else nn.GELU
        for index in range(len(dims) - 1):
            modules.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                modules.append(activation())
                if dropout > 0:
                    modules.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SmoothMLP(nn.Module):
    """Tanh MLP with a cached, exact Jacobian-vector-product operator.

    The linearization caches activation derivatives once per forward pass. A
    later JVP only performs batched matrix products and elementwise scaling.
    It is fully differentiable with respect to both parameters and base points.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int = 3):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * max(layers - 1, 0) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[index], dims[index + 1]) for index in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for index, layer in enumerate(self.layers):
            h = layer(h)
            if index < len(self.layers) - 1:
                h = torch.tanh(h)
        return h

    def linearize(
        self,
        x: torch.Tensor,
        chunk_size: int = 0,
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        h = x
        activation_derivatives: list[torch.Tensor | None] = []
        for index, layer in enumerate(self.layers):
            h = layer(h)
            if index < len(self.layers) - 1:
                h = torch.tanh(h)
                activation_derivatives.append(1.0 - h.square())
            else:
                activation_derivatives.append(None)
        output = h

        def apply(vectors: torch.Tensor) -> torch.Tensor:
            if vectors.ndim != 3 or vectors.shape[0] != x.shape[0] or vectors.shape[1] != x.shape[1]:
                raise ValueError(
                    f"Expected tangent [N,{x.shape[1]},C] for base {tuple(x.shape)}, got {tuple(vectors.shape)}"
                )
            pieces: list[torch.Tensor] = []
            for start, end in _chunk_ranges(x.shape[0], chunk_size):
                tangent = vectors[start:end]
                for layer, derivative in zip(self.layers, activation_derivatives, strict=True):
                    tangent = torch.einsum("oi,nic->noc", layer.weight, tangent)
                    if derivative is not None:
                        tangent = tangent * derivative[start:end, :, None]
                pieces.append(tangent)
            return torch.cat(pieces, dim=0)

        return output, apply
