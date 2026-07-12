from __future__ import annotations

from dataclasses import asdict

from graphatlas.config import ExperimentConfig


SUPPORTED_MODELS = {
    "ambient_vector_gnn",
    "signature_gnn",
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
    "graphatlas_min_distortion",
    "graphatlas_certified",
    "graphatlas_no_transport",
    "graphatlas_free_transition",
    "graphatlas_no_metric",
    "graphatlas_no_cocycle",
    "graphatlas_no_overlap",
    "graphatlas_no_rank",
}


def validate_config(config: ExperimentConfig) -> None:
    errors: list[str] = []
    d, m, loss, train = config.dataset, config.model, config.loss, config.train
    if d.num_nodes < 4:
        errors.append("dataset.num_nodes must be at least 4")
    if d.num_charts < 1:
        errors.append("dataset.num_charts must be positive")
    if d.atlas_variant not in {"coordinate_only", "mixed_metric", "boundary_stress"}:
        errors.append("dataset.atlas_variant must be coordinate_only, mixed_metric, or boundary_stress")
    if d.surface_positive_amplitude <= 0 or d.surface_negative_amplitude <= 0:
        errors.append("surface amplitudes must be positive")
    if d.surface_region_width <= 0:
        errors.append("surface_region_width must be positive")
    if not 0 <= d.cross_chart_edge_fraction <= 1:
        errors.append("cross_chart_edge_fraction must be in [0, 1]")
    if d.geodesic_pair_count < 1:
        errors.append("geodesic_pair_count must be positive")
    if d.intrinsic_candidate_neighbors < d.knn:
        errors.append("intrinsic_candidate_neighbors must be at least dataset.knn")
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
    if m.transport_mode not in {"original", "min_distortion", "certified"}:
        errors.append("model.transport_mode must be original, min_distortion, or certified")
    if m.transportability_beta < 0:
        errors.append("model.transportability_beta must be non-negative")
    if m.transportability_eps <= 0:
        errors.append("model.transportability_eps must be positive")
    if m.transportability_pinv_rtol <= 0:
        errors.append("model.transportability_pinv_rtol must be positive")
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
        if value is not None and name not in {
            "sample_nodes", "metric_probes", "rank_margin", "rank_max_condition"
        } and value < 0:
            errors.append(f"loss.{name} must be non-negative")
    if loss.sample_nodes < 1:
        errors.append("loss.sample_nodes must be positive")
    if loss.metric_probes < 1:
        errors.append("loss.metric_probes must be positive")
    if loss.metric_scale_weight < 0:
        errors.append("loss.metric_scale_weight must be non-negative")
    if loss.chart_rank < 0:
        errors.append("loss.chart_rank must be non-negative")
    if not 0 < loss.rank_margin <= 1:
        errors.append("loss.rank_margin must be in (0, 1]")
    if loss.rank_max_condition <= 1:
        errors.append("loss.rank_max_condition must be greater than one")
    if loss.rank_condition_weight < 0:
        errors.append("loss.rank_condition_weight must be non-negative")
    if train.epochs < 1 or train.patience < 1 or train.eval_every < 1 or train.regularization_every < 1:
        errors.append("training epoch and cadence values must be positive")
    if train.learning_rate <= 0 or train.weight_decay < 0 or train.grad_clip <= 0:
        errors.append("optimizer values are invalid")
    if train.light_regularization_nodes < 1:
        errors.append("train.light_regularization_nodes must be positive")
    if train.light_metric_probes < 1:
        errors.append("train.light_metric_probes must be positive")
    if errors:
        raise ValueError("Invalid GraphAtlas configuration:\n- " + "\n- ".join(errors))
