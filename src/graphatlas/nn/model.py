from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from graphatlas.config import ModelConfig
from graphatlas.data import GraphData
from .ambient import AmbientVectorGNN, SignatureGNN
from .baselines import (
    ACMGCNBaseline,
    APPNPBaseline,
    FAGCNBaseline,
    GATBaseline,
    GCNBaseline,
    GCNIIBaseline,
    GPRGNNBaseline,
    GeometryMoEBaseline,
    H2GCNBaseline,
    LINKBaseline,
    LINKXBaseline,
    LinearBaseline,
    MLPBaseline,
    MixHopBaseline,
    ResidualGCNBaseline,
    SGCBaseline,
)
from .charts import ChartLinearization, LocalChart, SmoothDiffeomorphism
from .equivariant import AtlasEquivariantLayer
from .functional import MLP, channel_jvp, graph_signature, restrict_topk, sparsemax


class GraphAtlas(nn.Module):
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
        self.membership_network = MLP(
            signature_dim,
            config.hidden_dim,
            config.num_charts,
            layers=3,
            dropout=config.dropout,
        )
        self.charts = nn.ModuleList(
            [LocalChart(config.observation_dim, config.chart_dim, config.hidden_dim) for _ in range(config.num_charts)]
        )
        self.vector_initializer = MLP(
            signature_dim,
            config.hidden_dim,
            config.observation_dim * config.vector_channels,
            layers=2,
            dropout=config.dropout,
        )
        mode = "transport"
        if config.name == "graphatlas_no_transport":
            mode = "no_transport"
        elif config.name == "graphatlas_free_transition":
            mode = "free_transition"
        self.layers = nn.ModuleList(
            [
                AtlasEquivariantLayer(
                    observation_dim=config.observation_dim,
                    chart_dim=config.chart_dim,
                    channels=config.vector_channels,
                    num_charts=config.num_charts,
                    hidden_dim=config.hidden_dim,
                    mode=mode,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        readout_dim = config.observation_dim * config.vector_channels + config.observation_dim
        self.classifier = MLP(readout_dim, config.hidden_dim, num_classes, layers=3, dropout=config.dropout)

    def chart_functions(
        self,
        chart_index: int,
        reparameterization: SmoothDiffeomorphism | None = None,
    ) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
        chart = self.charts[chart_index]
        if reparameterization is None:
            return chart.encoder, chart.decoder

        def phi(h: torch.Tensor) -> torch.Tensor:
            return reparameterization.forward(chart.encoder(h))

        def psi(u_tilde: torch.Tensor) -> torch.Tensor:
            return chart.decoder(reparameterization.inverse(u_tilde))

        return phi, psi

    def linearize_chart(
        self,
        chart_index: int,
        observation: torch.Tensor,
        reparameterization: SmoothDiffeomorphism | None = None,
    ) -> ChartLinearization:
        return self.charts[chart_index].linearize(
            observation,
            reparameterization,
            chunk_size=self.config.jacobian_chunk_size,
        )

    def pull_vectors(
        self,
        chart_index: int,
        h: torch.Tensor,
        observation_vectors: torch.Tensor,
        reparameterization: SmoothDiffeomorphism | None = None,
    ) -> torch.Tensor:
        chart = self.linearize_chart(chart_index, h, reparameterization)
        return chart.pull(observation_vectors)

    def push_vectors(
        self,
        chart_index: int,
        coordinates: torch.Tensor,
        tangent_vectors: torch.Tensor,
        reparameterization: SmoothDiffeomorphism | None = None,
    ) -> torch.Tensor:
        # For arbitrary coordinates used by diagnostics and regularizers, build
        # only the decoder-side linearization. The main forward path reuses a
        # cached full chart linearization instead.
        chart = self.charts[chart_index]
        if reparameterization is None:
            _, decoder_jvp = chart.decoder.linearize(coordinates, self.config.jacobian_chunk_size)
            return decoder_jvp(tangent_vectors)
        base_coordinate, inverse_jvp = reparameterization.linearize_inverse(
            coordinates,
            self.config.jacobian_chunk_size,
        )
        _, decoder_jvp = chart.decoder.linearize(base_coordinate, self.config.jacobian_chunk_size)
        return decoder_jvp(inverse_jvp(tangent_vectors))

    def generic_pull_vectors(
        self,
        chart_index: int,
        h: torch.Tensor,
        observation_vectors: torch.Tensor,
        reparameterization: SmoothDiffeomorphism | None = None,
    ) -> torch.Tensor:
        """Reference autograd JVP used only for validation tests."""
        phi, _ = self.chart_functions(chart_index, reparameterization)
        return channel_jvp(phi, h, observation_vectors, self.config.jacobian_chunk_size)

    def forward(
        self,
        data: GraphData,
        reparameterizations: Sequence[SmoothDiffeomorphism | None] | None = None,
        compact: bool = False,
    ) -> dict[str, torch.Tensor]:
        signature = graph_signature(data.x, data.edge_index)
        h = self.observation_encoder(signature)
        membership_logits = self.membership_network(signature)
        membership_soft = torch.softmax(membership_logits, dim=-1)
        membership = sparsemax(membership_logits, dim=-1)
        topk = 1 if self.config.name == "graphatlas_no_overlap" else self.config.membership_topk
        membership = restrict_topk(membership, topk)

        chart_states: list[ChartLinearization] = []
        for chart_index in range(self.config.num_charts):
            rep = None if reparameterizations is None else reparameterizations[chart_index]
            chart_states.append(self.linearize_chart(chart_index, h, rep))

        coordinates = torch.stack([state.coordinate for state in chart_states], dim=1)
        reconstruction = torch.stack([state.reconstruction for state in chart_states], dim=1)

        def pull(chart: int, vectors: torch.Tensor) -> torch.Tensor:
            return chart_states[chart].pull(vectors)

        def push(chart: int, vectors: torch.Tensor) -> torch.Tensor:
            return chart_states[chart].push(vectors)

        base_vectors = self.vector_initializer(signature).view(
            data.num_nodes,
            self.config.observation_dim,
            self.config.vector_channels,
        )
        tangent = torch.stack([pull(chart, base_vectors) for chart in range(self.config.num_charts)], dim=1)
        layer_diagnostics: list[dict[str, torch.Tensor]] = []
        for layer in self.layers:
            tangent, diagnostics = layer(tangent, membership, h, data.edge_index, push, pull)
            layer_diagnostics.append(diagnostics)

        pushed = torch.stack([push(chart, tangent[:, chart]) for chart in range(self.config.num_charts)], dim=1)
        observation_vectors = (pushed * membership[:, :, None, None]).sum(dim=1)
        readout = torch.cat([h, observation_vectors.flatten(start_dim=1)], dim=-1)
        logits = self.classifier(readout)
        if compact:
            return {"logits": logits, "embedding": readout}
        return {
            "logits": logits,
            "embedding": readout,
            "membership": membership,
            "membership_soft": membership_soft,
            "observation": h,
            "coordinates": coordinates,
            "reconstruction": reconstruction,
            "tangent": tangent,
            "observation_vectors": observation_vectors,
            "layer_diagnostics": layer_diagnostics,
            "chart_reparameterizations": (
                list(reparameterizations)
                if reparameterizations is not None
                else [None] * self.config.num_charts
            ),
        }


def build_model(input_dim: int, num_classes: int, config: ModelConfig, num_nodes: int | None = None) -> nn.Module:
    name = config.name
    if name == "ambient_vector_gnn":
        return AmbientVectorGNN(input_dim, num_classes, config)
    if name == "signature_gnn":
        return SignatureGNN(input_dim, num_classes, config)
    if name == "linear":
        return LinearBaseline(input_dim, num_classes)
    if name == "mlp":
        return MLPBaseline(input_dim, config.hidden_dim, num_classes, config.dropout)
    if name == "gcn":
        return GCNBaseline(input_dim, config.hidden_dim, num_classes, config.num_layers, config.dropout)
    if name == "gat":
        return GATBaseline(input_dim, config.hidden_dim, num_classes, config.num_layers, config.attention_heads, config.dropout)
    if name == "gcnii":
        return GCNIIBaseline(input_dim, config.hidden_dim, num_classes, config.num_layers, config.dropout, config.alpha, config.theta)
    if name == "fagcn":
        return FAGCNBaseline(input_dim, config.hidden_dim, num_classes, config.num_layers, config.dropout, config.alpha)
    if name == "acmgcn":
        return ACMGCNBaseline(input_dim, config.hidden_dim, num_classes, config.num_layers, config.dropout)
    if name == "resgcn":
        return ResidualGCNBaseline(input_dim, config.hidden_dim, num_classes, config.num_layers, config.dropout)
    if name == "sgc":
        return SGCBaseline(input_dim, num_classes, hops=max(1, config.num_layers))
    if name == "appnp":
        return APPNPBaseline(
            input_dim, config.hidden_dim, num_classes, config.propagation_steps, config.teleport, config.dropout
        )
    if name == "gprgnn":
        return GPRGNNBaseline(
            input_dim, config.hidden_dim, num_classes, config.propagation_steps, config.teleport, config.dropout
        )
    if name == "mixhop":
        return MixHopBaseline(input_dim, config.hidden_dim, num_classes, config.num_layers, config.dropout)
    if name == "h2gcn":
        return H2GCNBaseline(input_dim, config.hidden_dim, num_classes, config.num_layers, config.dropout)
    if name == "link":
        if num_nodes is None:
            raise ValueError("LINK requires num_nodes")
        return LINKBaseline(num_nodes, config.hidden_dim, num_classes, config.dropout)
    if name == "linkx":
        if num_nodes is None:
            raise ValueError("LINKX requires num_nodes")
        return LINKXBaseline(num_nodes, input_dim, config.hidden_dim, num_classes, config.dropout)
    if name == "geometry_moe":
        return GeometryMoEBaseline(
            input_dim, config.hidden_dim, num_classes, config.num_charts, config.num_layers, config.dropout
        )
    if name.startswith("graphatlas"):
        return GraphAtlas(input_dim, num_classes, config)
    raise ValueError(f"Unknown model: {name}")
