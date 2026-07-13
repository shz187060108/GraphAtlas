from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import deep_update, load_yaml


@dataclass
class DatasetConfig:
    name: str = "atlas_het"
    root: str = "data"
    num_nodes: int = 600
    num_charts: int = 3
    latent_dim: int = 2
    feature_dim: int = 16
    num_classes: int = 3
    overlap: float = 0.22
    heterophily: float = 0.7
    relation_edges_per_node: int = 2
    knn: int = 8
    noise: float = 0.05
    coordinate_shift_strength: float = 0.80
    split: int = 0
    task: str = "node_classification"
    link_val_ratio: float = 0.1
    link_test_ratio: float = 0.1
    negative_ratio: float = 1.0
    local_path: str | None = None
    format: str = "auto"
    metric: str | None = None
    target_node_type: str | None = None
    feature_source: str = "native"
    directed: bool = False
    max_nodes: int | None = None
    enabled: bool = True
    atlas_variant: str = "coordinate_only"
    surface_positive_amplitude: float = 0.55
    surface_negative_amplitude: float = 0.55
    surface_region_width: float = 0.85
    cross_chart_edge_fraction: float = 0.15
    geodesic_pair_count: int = 512
    intrinsic_candidate_neighbors: int = 24
    # Stable identifiers for generated condition grids and interventions.
    condition_id: str | None = None
    intervention_profile: str | None = None


@dataclass
class ModelConfig:
    name: str = "graphatlas"
    hidden_dim: int = 64
    observation_dim: int = 12
    chart_dim: int = 2
    vector_channels: int = 6
    num_charts: int = 3
    num_layers: int = 2
    dropout: float = 0.2
    membership_topk: int = 2
    jacobian_chunk_size: int = 2048
    edge_chunk_size: int = 100000
    # Keep normal public graphs on the original vectorized attention path.
    # ``edge_chunk_size`` is only the block size after this threshold is crossed.
    edge_chunk_threshold: int = 2_000_000
    propagation_steps: int = 10
    teleport: float = 0.1
    attention_heads: int = 4
    alpha: float = 0.1
    theta: float = 0.5
    label: str | None = None
    transport_mode: str = "original"
    transportability_beta: float = 1.0
    transportability_eps: float = 1e-8
    transportability_pinv_rtol: float = 1e-5
    transportability_stop_gradient: bool = True
    certified_routing_mode: str = "certificate"
    routing_control_seed: int = 1729
    routing_control_max_edges: int = 200000
    readout_mode: str = "full"
    coordinate_activation: str = "none"
    coordinate_left_linear: bool = False
    link_decoder: str = "dot"
    link_decoder_hidden_dim: int = 64


@dataclass
class LossConfig:
    task: float = 1.0
    reconstruction: float = 0.2
    cocycle: float = 0.05
    inverse_cycle: float | None = None
    path_consistency: float | None = None
    metric: float = 0.05
    metric_probes: int = 4
    metric_scale_weight: float = 0.10
    chart_rank: float = 0.0
    rank_margin: float = 0.20
    rank_max_condition: float = 20.0
    rank_condition_weight: float = 0.05
    cover: float = 0.02
    balance: float = 0.01
    sparsity: float = 0.01
    geometry: float = 0.0
    sample_nodes: int = 128

    def resolved_inverse_cycle(self) -> float:
        return self.cocycle if self.inverse_cycle is None else self.inverse_cycle

    def resolved_path_consistency(self) -> float:
        return self.cocycle if self.path_consistency is None else self.path_consistency


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "auto"
    epochs: int = 200
    patience: int = 40
    learning_rate: float = 0.003
    weight_decay: float = 0.0005
    grad_clip: float = 5.0
    progress: bool = True
    eval_every: int = 5
    regularization_every: int = 10
    light_regularization_nodes: int = 32
    light_metric_probes: int = 2
    # Checkpoints and raw predictions are resumability internals, not the
    # user-facing result surface.  Human-readable summaries live in
    # outputs/results and outputs/reports.
    output_dir: str = "outputs/.work/runs"
    num_threads: int = 1
    resume: bool = True
    neighbor_sampling: bool = False
    batch_size: int = 1024
    num_neighbors: list[int] = field(default_factory=lambda: [15])
    num_workers: int = 0


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            dataset=DatasetConfig(**data.get("dataset", {})),
            model=ModelConfig(**data.get("model", {})),
            loss=LossConfig(**data.get("loss", {})),
            train=TrainConfig(**data.get("train", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "ExperimentConfig":
        data = load_yaml(path)
        if overrides:
            data = deep_update(data, overrides)
        return cls.from_dict(data)

    def validate(self) -> None:
        from .validation import validate_config

        validate_config(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.__dict__,
            "model": self.model.__dict__,
            "loss": self.loss.__dict__,
            "train": self.train.__dict__,
        }

    def fingerprint_dict(self) -> dict[str, Any]:
        """Configuration fields that determine scientific results.

        Runtime-only fields are deliberately excluded so changing progress
        display, device selection, worker threads, output location, or resume
        behavior does not create a second logical experiment.
        """
        payload = self.to_dict()
        train = dict(payload["train"])
        for key in ("device", "progress", "output_dir", "num_threads", "resume"):
            train.pop(key, None)
        payload["train"] = train
        return payload
