from __future__ import annotations

from dataclasses import asdict

from graphatlas.config import ExperimentConfig


SUPPORTED_MODELS = {
    "linear",
    "link",
    "gat",
    "gcnii",
    "fagcn",
    "acmgcn",
    "mlp",
    "gcn",
    "resgcn",
    "sgc",
    "h2gcn",
    "appnp",
    "gprgnn",
    "mixhop",
    "linkx",
    "geometry_moe",
    "graphatlas",
    "graphatlas_no_transport",
    "graphatlas_free_transition",
    "graphatlas_no_metric",
    "graphatlas_no_cocycle",
    "graphatlas_no_overlap",
}


def validate_config(config: ExperimentConfig) -> None:
    errors: list[str] = []
    d, m, loss, train = config.dataset, config.model, config.loss, config.train
    if d.num_nodes < 4:
        errors.append("dataset.num_nodes must be at least 4")
    if d.num_charts < 1:
        errors.append("dataset.num_charts must be positive")
    if not 0.0 <= d.overlap <= 1.0:
        errors.append("dataset.overlap must be in [0, 1]")
    if not 0.0 <= d.heterophily <= 1.0:
        errors.append("dataset.heterophily must be in [0, 1]")
    if not 0.0 <= d.coordinate_shift_strength <= 1.0:
        errors.append("dataset.coordinate_shift_strength must be in [0, 1]")
    if d.feature_dim < 1 or d.num_classes < 2:
        errors.append("feature_dim must be positive and num_classes at least 2")
    if d.task not in {"node_classification", "link_prediction"}:
        errors.append("dataset.task must be node_classification or link_prediction")
    if not 0.0 < d.link_val_ratio < 1.0 or not 0.0 < d.link_test_ratio < 1.0:
        errors.append("link validation and test ratios must be in (0, 1)")
    if d.link_val_ratio + d.link_test_ratio >= 1.0:
        errors.append("link validation and test ratios must sum to less than one")
    if d.negative_ratio <= 0:
        errors.append("dataset.negative_ratio must be positive")
    if m.name not in SUPPORTED_MODELS:
        errors.append(f"unsupported model.name={m.name!r}")
    for field in ("hidden_dim", "observation_dim", "chart_dim", "vector_channels", "num_charts", "num_layers"):
        if getattr(m, field) < 1:
            errors.append(f"model.{field} must be positive")
    if m.membership_topk < 1 or m.membership_topk > m.num_charts:
        errors.append("model.membership_topk must be between 1 and model.num_charts")
    if m.name.startswith("graphatlas") and m.num_charts != d.num_charts:
        errors.append("GraphAtlas model.num_charts must match dataset.num_charts")
    if not 0.0 <= m.dropout < 1.0:
        errors.append("model.dropout must be in [0, 1)")
    if m.propagation_steps < 1:
        errors.append("model.propagation_steps must be positive")
    if m.attention_heads < 1:
        errors.append("model.attention_heads must be positive")
    if not 0.0 <= m.alpha <= 1.0:
        errors.append("model.alpha must be in [0, 1]")
    if m.theta <= 0:
        errors.append("model.theta must be positive")
    if not 0.0 < m.teleport <= 1.0:
        errors.append("model.teleport must be in (0, 1]")
    for name, value in asdict(loss).items():
        if name != "sample_nodes" and value < 0:
            errors.append(f"loss.{name} must be non-negative")
    if loss.sample_nodes < 1:
        errors.append("loss.sample_nodes must be positive")
    if train.epochs < 1 or train.patience < 1 or train.eval_every < 1 or train.regularization_every < 1:
        errors.append("training epoch and cadence values must be positive")
    if train.learning_rate <= 0 or train.weight_decay < 0 or train.grad_clip <= 0:
        errors.append("optimizer values are invalid")
    if errors:
        raise ValueError("Invalid GraphAtlas configuration:\n- " + "\n- ".join(errors))
