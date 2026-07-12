from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

from graphatlas.config import LossConfig
from graphatlas.data import GraphData
from graphatlas.nn.model import GraphAtlas
from graphatlas.nn.functional import edge_dot_scores


def _sample_indices(num_nodes: int, sample_size: int, device: torch.device) -> torch.Tensor:
    if sample_size <= 0 or sample_size >= num_nodes:
        return torch.arange(num_nodes, device=device)
    return torch.randperm(num_nodes, device=device)[:sample_size]


def atlas_regularization_terms(
    model: GraphAtlas,
    output: dict[str, torch.Tensor],
    data: GraphData,
    config: LossConfig,
) -> dict[str, torch.Tensor]:
    membership = output["membership"]
    membership_soft = output.get("membership_soft", membership)
    h = output["observation"]
    coordinates = output["coordinates"]
    reconstruction = output["reconstruction"]
    device = h.device
    zero = h.new_zeros(())

    rec_per_chart = (reconstruction - h[:, None]).square().mean(dim=-1)
    reconstruction_loss = (membership * rec_per_chart).sum() / membership.sum().clamp_min(1.0)
    cover_loss = torch.relu(0.34 - membership.max(dim=-1).values).square().mean()
    balance_loss = (membership.mean(dim=0) - 1.0 / membership.shape[1]).square().mean()
    sparsity_loss = (membership * (1.0 - membership)).mean()

    indices = _sample_indices(data.num_nodes, config.sample_nodes, device)
    pair_cycle_numerator = zero
    pair_cycle_denominator = zero
    path_numerator = zero
    path_denominator = zero
    triple_cocycle_numerator = zero
    triple_cocycle_denominator = zero
    k_count = model.config.num_charts
    inverse_weight = config.resolved_inverse_cycle()
    path_weight = config.resolved_path_consistency()
    if (inverse_weight > 0 or path_weight > 0) and k_count >= 2:
        # Soft chart occupancy defines a differentiable chart-overlap graph.
        # Propagation still uses sparse top-k membership, while this graph keeps
        # route consistency trainable when membership_topk=2 and no node belongs
        # to three charts simultaneously.
        overlap_graph = membership_soft.transpose(0, 1) @ membership_soft
        overlap_graph = overlap_graph / max(int(membership_soft.shape[0]), 1)
        overlap_graph = overlap_graph - torch.diag_embed(torch.diagonal(overlap_graph))

        # Pairwise inverse-cycle consistency checks that chart transitions retain
        # the same decoded observation when transported out and back.
        if inverse_weight > 0:
            for source in range(k_count):
                phi_source, psi_source = model.chart_functions(source)
                source_coordinate = coordinates[indices, source]
                source_observation = psi_source(source_coordinate)
                for target in range(k_count):
                    if target == source:
                        continue
                    phi_target, psi_target = model.chart_functions(target)
                    target_observation = psi_target(phi_target(source_observation))
                    roundtrip_observation = psi_source(phi_source(target_observation))
                    weight = membership_soft[indices, source] * membership_soft[indices, target]
                    error = (source_observation - roundtrip_observation).square().mean(dim=-1)
                    pair_cycle_numerator = pair_cycle_numerator + (weight * error).sum()
                    pair_cycle_denominator = pair_cycle_denominator + weight.sum()

        if path_weight > 0 and k_count >= 3:
            for source, middle, target in itertools.permutations(range(k_count), 3):
                phi_middle, psi_middle = model.chart_functions(middle)
                phi_target, psi_target = model.chart_functions(target)
                _, psi_source = model.chart_functions(source)
                source_coordinate = coordinates[indices, source]
                source_observation = psi_source(source_coordinate)
                direct_observation = psi_target(phi_target(source_observation))
                middle_observation = psi_middle(phi_middle(source_observation))
                via_observation = psi_target(phi_target(middle_observation))
                error = (direct_observation - via_observation).square().mean(dim=-1)

                route_mass = overlap_graph[source, middle] * overlap_graph[middle, target]
                route_weight = (
                    membership_soft[indices, source]
                    * membership_soft[indices, target]
                    * route_mass
                )
                path_numerator = path_numerator + (route_weight * error).sum()
                path_denominator = path_denominator + route_weight.sum()

                # Classical triple-overlap cocycle is retained as a diagnostic.
                triple_weight = (
                    membership[indices, source]
                    * membership[indices, middle]
                    * membership[indices, target]
                )
                triple_cocycle_numerator = triple_cocycle_numerator + (triple_weight * error).sum()
                triple_cocycle_denominator = triple_cocycle_denominator + triple_weight.sum()

    pair_cycle_loss = torch.where(
        pair_cycle_denominator > 0,
        pair_cycle_numerator / pair_cycle_denominator.clamp_min(1e-12),
        zero,
    )
    path_consistency_loss = torch.where(
        path_denominator > 0,
        path_numerator / path_denominator.clamp_min(1e-12),
        zero,
    )
    triple_cocycle_loss = torch.where(
        triple_cocycle_denominator > 0,
        triple_cocycle_numerator / triple_cocycle_denominator.clamp_min(1e-12),
        zero,
    )
    # Both inverse consistency and two-hop route consistency are required for
    # a usable atlas. The classical triple term is diagnostic because sparse
    # chart covers can legitimately have no node-level triple overlap.
    cocycle_loss = pair_cycle_loss + path_consistency_loss

    metric_numerator = zero
    metric_denominator = zero
    metric_direction_numerator = zero
    metric_scale_numerator = zero
    if config.metric > 0 and k_count >= 2:
        for source in range(k_count):
            _, psi_source = model.chart_functions(source)
            source_coordinate = coordinates[indices, source]
            source_observation = psi_source(source_coordinate)
            probe = torch.randn(
                indices.shape[0],
                model.config.chart_dim,
                config.metric_probes,
                device=device,
                dtype=h.dtype,
            )
            probe = probe / torch.linalg.vector_norm(probe, dim=1, keepdim=True).clamp_min(1e-8)
            source_pushed = model.push_vectors(source, source_coordinate, probe)
            for target in range(k_count):
                if target == source:
                    continue
                phi_target, _ = model.chart_functions(target)
                target_coordinate = phi_target(source_observation)
                target_tangent = model.pull_vectors(target, source_observation, source_pushed)
                reconstructed = model.push_vectors(target, target_coordinate, target_tangent)

                source_norm = torch.linalg.vector_norm(source_pushed, dim=1).clamp_min(1e-8)
                reconstructed_norm = torch.linalg.vector_norm(reconstructed, dim=1).clamp_min(1e-8)
                source_direction = source_pushed / source_norm[:, None, :]
                reconstructed_direction = reconstructed / reconstructed_norm[:, None, :]
                direction_error = (source_direction - reconstructed_direction).square().sum(dim=1)
                scale_error = (torch.log(source_norm) - torch.log(reconstructed_norm)).square()
                per_node_direction = direction_error.mean(dim=-1)
                per_node_scale = scale_error.mean(dim=-1)
                probe_error = per_node_direction + config.metric_scale_weight * per_node_scale
                weight = membership_soft[indices, source] * membership_soft[indices, target]
                metric_numerator = metric_numerator + (weight * probe_error).sum()
                metric_direction_numerator = metric_direction_numerator + (weight * per_node_direction).sum()
                metric_scale_numerator = metric_scale_numerator + (weight * per_node_scale).sum()
                metric_denominator = metric_denominator + weight.sum()
    metric_loss = metric_numerator / metric_denominator.clamp_min(1.0)
    metric_direction_error = metric_direction_numerator / metric_denominator.clamp_min(1.0)
    metric_scale_error = metric_scale_numerator / metric_denominator.clamp_min(1.0)

    chart_rank_numerator = zero
    chart_rank_denominator = zero
    relative_min_numerator = zero
    condition_numerator = zero
    if config.chart_rank > 0:
        basis = torch.eye(model.config.chart_dim, device=device, dtype=h.dtype).expand(
            indices.shape[0], -1, -1
        )
        for chart in range(k_count):
            jacobian_columns = model.push_vectors(
                chart,
                coordinates[indices, chart],
                basis,
            )
            singular_values = torch.linalg.svdvals(jacobian_columns)
            if not bool(torch.isfinite(singular_values).all()):
                raise FloatingPointError(f"Non-finite decoder singular values in chart {chart}")
            s_min = singular_values.min(dim=-1).values
            s_max = singular_values.max(dim=-1).values
            rms = torch.sqrt(singular_values.square().mean(dim=-1)).clamp_min(1e-8)
            relative_min = s_min / rms
            condition = s_max / s_min.clamp_min(1e-8)
            margin_defect = torch.relu(config.rank_margin - relative_min).square()
            condition_defect = torch.relu(torch.log(condition / config.rank_max_condition)).square()
            per_node_rank_loss = margin_defect + config.rank_condition_weight * condition_defect
            weight = membership_soft[indices, chart]
            chart_rank_numerator = chart_rank_numerator + (weight * per_node_rank_loss).sum()
            relative_min_numerator = relative_min_numerator + (weight * relative_min).sum()
            condition_numerator = condition_numerator + (weight * condition).sum()
            chart_rank_denominator = chart_rank_denominator + weight.sum()
    chart_rank_loss = chart_rank_numerator / chart_rank_denominator.clamp_min(1.0)
    chart_min_relative_singular_value = relative_min_numerator / chart_rank_denominator.clamp_min(1.0)
    chart_condition_number = condition_numerator / chart_rank_denominator.clamp_min(1.0)

    geometry_loss = zero
    if config.geometry > 0 and data.latent_positions is not None:
        src, dst = data.edge_index
        edge_count = min(config.sample_nodes * 4, src.numel())
        chosen = torch.randperm(src.numel(), device=device)[:edge_count]
        i, j = src[chosen], dst[chosen]
        true_distance = torch.linalg.vector_norm(data.latent_positions[i] - data.latent_positions[j], dim=-1)
        predicted = []
        weights = []
        for chart in range(k_count):
            delta = coordinates[j, chart] - coordinates[i, chart]
            local_vector = model.push_vectors(chart, coordinates[i, chart], delta.unsqueeze(-1)).squeeze(-1)
            predicted.append(torch.linalg.vector_norm(local_vector, dim=-1))
            weights.append(membership[i, chart] * membership[j, chart])
        predicted_tensor = torch.stack(predicted, dim=-1)
        weight_tensor = torch.stack(weights, dim=-1)
        error = (predicted_tensor - true_distance[:, None]).square()
        geometry_loss = (weight_tensor * error).sum() / weight_tensor.sum().clamp_min(1.0)

    return {
        "reconstruction": reconstruction_loss,
        "cover": cover_loss,
        "balance": balance_loss,
        "sparsity": sparsity_loss,
        "cocycle": cocycle_loss,
        "inverse_cycle": pair_cycle_loss,
        "path_consistency": path_consistency_loss,
        "triple_cocycle": triple_cocycle_loss,
        "metric": metric_loss,
        "metric_direction_error": metric_direction_error,
        "metric_scale_error": metric_scale_error,
        "chart_rank": chart_rank_loss,
        "chart_min_relative_singular_value": chart_min_relative_singular_value,
        "chart_condition_number": chart_condition_number,
        "geometry": geometry_loss,
    }


def compute_loss(
    model: torch.nn.Module,
    output: dict[str, torch.Tensor],
    data: GraphData,
    config: LossConfig,
    task_name: str = "node_classification",
) -> tuple[torch.Tensor, dict[str, float]]:
    if task_name == "node_classification":
        if data.is_multilabel:
            target = data.y[data.train_mask].to(output["logits"].dtype)
            score = output["logits"][data.train_mask]
            finite = torch.isfinite(target)
            task = F.binary_cross_entropy_with_logits(score[finite], target[finite])
        else:
            task = F.cross_entropy(output["logits"][data.train_mask], data.y[data.train_mask])
    elif task_name == "link_prediction":
        if data.link_split is None:
            raise ValueError("link_prediction requires data.link_split")
        embedding = output.get("embedding")
        if embedding is None:
            raise ValueError(f"{model.__class__.__name__} does not expose node embeddings for link prediction")
        positive = edge_dot_scores(embedding, data.link_split.train_pos)
        negative = edge_dot_scores(embedding, data.link_split.train_neg)
        scores = torch.cat([positive, negative])
        labels = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)])
        task = F.binary_cross_entropy_with_logits(scores, labels)
    else:
        raise ValueError(f"Unsupported task: {task_name}")
    terms: dict[str, torch.Tensor] = {
        "task": task,
        "reconstruction": task.new_zeros(()),
        "cocycle": task.new_zeros(()),
        "inverse_cycle": task.new_zeros(()),
        "path_consistency": task.new_zeros(()),
        "triple_cocycle": task.new_zeros(()),
        "metric": task.new_zeros(()),
        "metric_direction_error": task.new_zeros(()),
        "metric_scale_error": task.new_zeros(()),
        "chart_rank": task.new_zeros(()),
        "chart_min_relative_singular_value": task.new_zeros(()),
        "chart_condition_number": task.new_zeros(()),
        "cover": task.new_zeros(()),
        "balance": task.new_zeros(()),
        "sparsity": task.new_zeros(()),
        "geometry": task.new_zeros(()),
    }
    if isinstance(model, GraphAtlas):
        terms.update(atlas_regularization_terms(model, output, data, config))
    total = (
        config.task * terms["task"]
        + config.reconstruction * terms["reconstruction"]
        + config.resolved_inverse_cycle() * terms["inverse_cycle"]
        + config.resolved_path_consistency() * terms["path_consistency"]
        + config.metric * terms["metric"]
        + config.chart_rank * terms["chart_rank"]
        + config.cover * terms["cover"]
        + config.balance * terms["balance"]
        + config.sparsity * terms["sparsity"]
        + config.geometry * terms["geometry"]
    )
    values = {name: float(value.detach().cpu()) for name, value in terms.items()}
    values["total"] = float(total.detach().cpu())
    return total, values
