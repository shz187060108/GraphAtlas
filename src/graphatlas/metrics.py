from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (adjusted_rand_score, average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score, roc_auc_score)

from graphatlas.data import GraphData
from graphatlas.nn.charts import SmoothDiffeomorphism
from graphatlas.nn.functional import edge_dot_scores
from graphatlas.nn.model import GraphAtlas


OVERLAP_THRESHOLD = 0.05


def predicted_edge_lengths(
    model: GraphAtlas,
    output: dict[str, Any],
    data: GraphData,
) -> torch.Tensor:
    coordinates = output["coordinates"]
    membership = output["membership"]
    source, target = data.edge_index
    reparameterizations = output.get("chart_reparameterizations", [None] * model.config.num_charts)
    per_chart = []
    weights = []
    for chart in range(model.config.num_charts):
        reparameterization = reparameterizations[chart]
        if reparameterization is None:
            source_coordinate = coordinates[source, chart]
            target_coordinate = coordinates[target, chart]
        else:
            # Express a finite displacement in the chart's base coordinates.
            # This makes the diagnostic independent of the displayed chart
            # parameterization while still measuring its pushed tangent.
            source_coordinate = reparameterization.inverse(coordinates[source, chart])
            target_coordinate = reparameterization.inverse(coordinates[target, chart])
        delta = target_coordinate - source_coordinate
        pushed = model.push_vectors(
            chart,
            source_coordinate,
            delta.unsqueeze(-1),
            None,
        ).squeeze(-1)
        per_chart.append(torch.linalg.vector_norm(pushed, dim=-1))
        weights.append(membership[source, chart] * membership[target, chart])
    lengths = torch.stack(per_chart, dim=-1)
    weight = torch.stack(weights, dim=-1)
    denominator = weight.sum(dim=-1)
    shared = (lengths * weight).sum(dim=-1) / denominator.clamp_min(1e-12)
    # Observation-space differences are invariant to chart reparameterization
    # and provide a fixed fallback when an edge has no shared active chart.
    observation = output["observation"]
    fallback = torch.linalg.vector_norm(observation[target] - observation[source], dim=-1)
    return torch.where(denominator > 0, shared, fallback).clamp_min(1e-8)


def transition_error(
    model: GraphAtlas,
    output: dict[str, Any],
    data: GraphData,
    seed: int = 0,
    probes: int = 4,
) -> torch.Tensor:
    membership = output["membership"]
    h = output["observation"]
    coordinates = output["coordinates"]
    reps = output.get("chart_reparameterizations", [None] * model.config.num_charts)
    overlap = (membership > OVERLAP_THRESHOLD).sum(dim=-1) > 1
    indices = torch.nonzero(overlap, as_tuple=False).flatten()[:128]
    if not len(indices):
        return h.new_tensor(float("nan"))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_probe = torch.randn(
        len(indices), model.config.observation_dim, probes, generator=generator, dtype=h.dtype
    ).to(h.device)
    random_probe = random_probe / torch.linalg.vector_norm(random_probe, dim=1, keepdim=True).clamp_min(1e-8)
    numerator = h.new_zeros(())
    denominator = h.new_zeros(())
    for source_chart in range(model.config.num_charts):
        source_tangent = model.pull_vectors(
            source_chart, h[indices], random_probe, reps[source_chart]
        )
        source_observation = model.push_vectors(
            source_chart, coordinates[indices, source_chart], source_tangent, reps[source_chart]
        )
        for target_chart in range(model.config.num_charts):
            if target_chart == source_chart:
                continue
            target_tangent = model.pull_vectors(
                target_chart, h[indices], source_observation, reps[target_chart]
            )
            reconstructed = model.push_vectors(
                target_chart, coordinates[indices, target_chart], target_tangent, reps[target_chart]
            )
            source_norm = torch.linalg.vector_norm(source_observation, dim=1).clamp_min(1e-8)
            target_norm = torch.linalg.vector_norm(reconstructed, dim=1).clamp_min(1e-8)
            direction = (
                source_observation / source_norm[:, None]
                - reconstructed / target_norm[:, None]
            ).square().sum(dim=1)
            scale = (torch.log(source_norm) - torch.log(target_norm)).square()
            error = (direction + 0.1 * scale).mean(dim=-1)
            weight = membership[indices, source_chart] * membership[indices, target_chart]
            numerator = numerator + (weight * error).sum()
            denominator = denominator + weight.sum()
    return numerator / denominator.clamp_min(1e-12)


def intrinsic_geometry_diagnostics(
    model: GraphAtlas,
    output: dict[str, Any],
    data: GraphData,
    seed: int,
) -> dict[str, float]:
    result = {
        "metric_recovery_error": float("nan"),
        "metric_recovery_mean_error": float("nan"),
        "transition_error": float("nan"),
        "local_distortion": float("nan"),
        "cross_chart_distortion": float("nan"),
    }
    predicted = predicted_edge_lengths(model, output, data)
    if data.true_edge_lengths is not None:
        relative = (predicted - data.true_edge_lengths).abs() / data.true_edge_lengths.clamp_min(1e-6)
        result["metric_recovery_error"] = float(relative.median().detach().cpu())
        result["metric_recovery_mean_error"] = float(relative.mean().detach().cpu())
    result["transition_error"] = float(transition_error(model, output, data, seed).detach().cpu())
    if data.geodesic_pairs is not None and data.geodesic_distances is not None:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        source, target = data.edge_index.detach().cpu().numpy()
        graph = csr_matrix(
            (predicted.detach().cpu().numpy(), (source, target)),
            shape=(data.num_nodes, data.num_nodes),
        )
        pairs = data.geodesic_pairs.detach().cpu().numpy()
        unique_sources = sorted(set(map(int, pairs[0])))
        distances = dijkstra(graph, directed=False, indices=unique_sources)
        lookup = {node: row for row, node in enumerate(unique_sources)}
        predicted_distance = np.asarray([distances[lookup[int(i)], int(j)] for i, j in pairs.T])
        true_distance = data.geodesic_distances.detach().cpu().numpy()
        finite = np.isfinite(predicted_distance) & (predicted_distance > 0)
        distortion = np.full(len(true_distance), np.nan)
        distortion[finite] = np.abs(np.log(predicted_distance[finite] / true_distance[finite]))
        dominant = data.chart_membership.argmax(dim=-1).detach().cpu().numpy()
        same = dominant[pairs[0]] == dominant[pairs[1]]
        if np.any(finite & same):
            result["local_distortion"] = float(np.nanmean(distortion[finite & same]))
        if np.any(finite & ~same):
            result["cross_chart_distortion"] = float(np.nanmean(distortion[finite & ~same]))
    return result



def _safe_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def expected_calibration_error(probability: torch.Tensor, y: torch.Tensor, bins: int = 15) -> float:
    if y.ndim != 1 or probability.numel() == 0:
        return float("nan")
    confidence, prediction = probability.max(dim=-1)
    correctness = prediction.eq(y).float()
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probability.device)
    result = probability.new_zeros(())
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        selected = (confidence > left) & (confidence <= right)
        if bool(selected.any()):
            result = result + selected.float().mean() * (correctness[selected].mean() - confidence[selected].mean()).abs()
    return float(result.detach().cpu())


def single_label_metrics(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    if int(mask.sum()) == 0:
        return {name: float("nan") for name in ["accuracy", "macro_f1", "micro_f1", "weighted_f1", "balanced_accuracy", "mcc", "worst_class_recall", "nll", "brier", "ece"]}
    selected_logits = logits[mask]
    selected_y = y[mask]
    probability = torch.softmax(selected_logits, dim=-1)
    prediction = probability.argmax(dim=-1)
    true_np, pred_np = _safe_numpy(selected_y), _safe_numpy(prediction)
    recalls = recall_score(true_np, pred_np, average=None, zero_division=0)
    one_hot = torch.nn.functional.one_hot(selected_y, num_classes=logits.shape[-1]).to(probability.dtype)
    return {
        "accuracy": float((prediction == selected_y).float().mean().cpu()),
        "macro_f1": float(f1_score(true_np, pred_np, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(true_np, pred_np, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(true_np, pred_np, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true_np, pred_np)),
        "mcc": float(matthews_corrcoef(true_np, pred_np)),
        "worst_class_recall": float(np.min(recalls)) if len(recalls) else float("nan"),
        "nll": float(torch.nn.functional.cross_entropy(selected_logits, selected_y).cpu()),
        "brier": float((probability - one_hot).square().sum(dim=-1).mean().cpu()),
        "ece": expected_calibration_error(probability, selected_y),
    }


def multilabel_metrics(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    selected_logits = logits[mask]
    selected_y = y[mask].to(torch.float32)
    finite = torch.isfinite(selected_y)
    probability = torch.sigmoid(selected_logits)
    prediction = probability >= 0.5
    task_auc: list[float] = []
    task_ap: list[float] = []
    for column in range(selected_y.shape[1]):
        valid = finite[:, column]
        if int(valid.sum()) == 0:
            continue
        target = _safe_numpy(selected_y[valid, column])
        score = _safe_numpy(probability[valid, column])
        if np.unique(target).size >= 2:
            task_auc.append(float(roc_auc_score(target, score)))
            task_ap.append(float(average_precision_score(target, score)))
    valid_y = selected_y[finite]
    valid_logits = selected_logits[finite]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(valid_logits, valid_y) if valid_y.numel() else selected_logits.new_tensor(float("nan"))
    true_np = _safe_numpy(torch.where(finite, selected_y, torch.zeros_like(selected_y))).astype(int)
    pred_np = _safe_numpy(torch.where(finite, prediction, torch.zeros_like(prediction))).astype(int)
    return {
        "roc_auc": float(np.mean(task_auc)) if task_auc else float("nan"),
        "average_precision": float(np.mean(task_ap)) if task_ap else float("nan"),
        "macro_f1": float(f1_score(true_np, pred_np, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(true_np, pred_np, average="micro", zero_division=0)),
        "nll": float(bce.detach().cpu()),
        "brier": float(((probability[finite] - selected_y[finite]).square().mean()).cpu()) if bool(finite.any()) else float("nan"),
    }

def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if y.ndim == 2:
        return float("nan")
    return single_label_metrics(logits, y, mask)["accuracy"]


def roc_auc(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum()) == 0 or logits.shape[-1] != 2:
        return float("nan")
    y_true = y[mask].detach().cpu().numpy()
    if np.unique(y_true).size < 2:
        return float("nan")
    score = torch.softmax(logits[mask], dim=-1)[:, 1].detach().cpu().numpy()
    return float(roc_auc_score(y_true, score))


def primary_metric_name(data: GraphData) -> str:
    if data.metadata and data.metadata.get("task") == "link_prediction":
        return "roc_auc"
    metric = (data.metadata or {}).get("metric")
    if metric:
        return str(metric)
    if data.is_multilabel:
        return "roc_auc"
    return "accuracy"


def primary_metric(logits: torch.Tensor, data: GraphData, mask: torch.Tensor) -> float:
    name = primary_metric_name(data)
    if data.is_multilabel:
        values = multilabel_metrics(logits, data.y, mask)
        return values.get(name, values["roc_auc"])
    if name == "roc_auc":
        return roc_auc(logits, data.y, mask)
    if name == "macro_f1":
        return single_label_metrics(logits, data.y, mask)["macro_f1"]
    if name == "balanced_accuracy":
        return single_label_metrics(logits, data.y, mask)["balanced_accuracy"]
    return accuracy(logits, data.y, mask)


def _link_split_edges(data: GraphData, split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if data.link_split is None:
        raise ValueError("Link-prediction metrics require data.link_split")
    positive = getattr(data.link_split, f"{split_name}_pos")
    negative = getattr(data.link_split, f"{split_name}_neg")
    return positive, negative


def link_prediction_metrics(embedding: torch.Tensor, data: GraphData, split_name: str) -> dict[str, float]:
    positive, negative = _link_split_edges(data, split_name)
    positive_score = edge_dot_scores(embedding, positive)
    negative_score = edge_dot_scores(embedding, negative)
    scores = torch.cat([positive_score, negative_score]).detach().cpu().numpy()
    labels = np.concatenate([np.ones(positive_score.numel()), np.zeros(negative_score.numel())])
    if positive_score.numel() == 0 or negative_score.numel() == 0:
        return {"roc_auc": float("nan"), "average_precision": float("nan"), "mrr": float("nan"), "hits_at_10": float("nan"), "hits_at_50": float("nan")}
    rank = 1 + (negative_score[None, :] >= positive_score[:, None]).sum(dim=1)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "mrr": float((1.0 / rank.to(torch.float32)).mean().cpu()),
        "hits_at_10": float((rank <= 10).float().mean().cpu()),
        "hits_at_50": float((rank <= 50).float().mean().cpu()),
    }


def output_primary_metric(output: dict[str, torch.Tensor], data: GraphData, split_name: str) -> float:
    if data.metadata and data.metadata.get("task") == "link_prediction":
        embedding = output.get("embedding")
        if embedding is None:
            raise ValueError("Model output does not contain embedding for link prediction")
        return link_prediction_metrics(embedding, data, split_name)["roc_auc"]
    mask = getattr(data, f"{split_name}_mask")
    return primary_metric(output["logits"], data, mask)


def coordinate_intervention_diagnostics(
    model: GraphAtlas,
    data: GraphData,
    seed: int = 1234,
    nonlinear: bool = True,
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        original = model(data, compact=True)
        reps = [
            SmoothDiffeomorphism.random(
                model.config.chart_dim,
                data.x.device,
                data.x.dtype,
                seed + chart,
                nonlinear=nonlinear,
            )
            for chart in range(model.config.num_charts)
        ]
        transformed = model(data, reparameterizations=reps, compact=True)
        if data.metadata and data.metadata.get("task") == "link_prediction":
            if data.link_split is None:
                raise ValueError("link prediction intervention requires link_split")
            edges = torch.cat([data.link_split.test_pos, data.link_split.test_neg], dim=1)
            score = edge_dot_scores(original["embedding"], edges)
            score_tilde = edge_dot_scores(transformed["embedding"], edges)
            probability = torch.sigmoid(score)
            probability_tilde = torch.sigmoid(score_tilde)
            mean_error = (probability - probability_tilde).abs().mean()
            max_error = (score - score_tilde).abs().max()
            flip_rate = ((probability >= 0.5) != (probability_tilde >= 0.5)).float().mean()
            accuracy_before = accuracy_after = accuracy_drop = probability.new_tensor(float("nan"))
        else:
            logits = original["logits"]
            logits_tilde = transformed["logits"]
            if data.is_multilabel:
                probability = torch.sigmoid(logits)
                probability_tilde = torch.sigmoid(logits_tilde)
                flip_rate = ((probability >= 0.5) != (probability_tilde >= 0.5)).float().mean()
                before = multilabel_metrics(logits, data.y, data.test_mask)["roc_auc"]
                after = multilabel_metrics(logits_tilde, data.y, data.test_mask)["roc_auc"]
            else:
                probability = torch.softmax(logits, dim=-1)
                probability_tilde = torch.softmax(logits_tilde, dim=-1)
                test = data.test_mask
                flip_rate = (logits.argmax(-1)[test] != logits_tilde.argmax(-1)[test]).float().mean()
                before = accuracy(logits, data.y, test)
                after = accuracy(logits_tilde, data.y, test)
            mean_error = (probability - probability_tilde).abs().sum(dim=-1).mean()
            max_error = (logits - logits_tilde).abs().max()
            accuracy_before = probability.new_tensor(before)
            accuracy_after = probability.new_tensor(after)
            accuracy_drop = accuracy_before - accuracy_after
    return {
        "mean_error": float(mean_error.detach().cpu()),
        "max_logit_error": float(max_error.detach().cpu()),
        "prediction_flip_rate": float(flip_rate.detach().cpu()),
        "test_metric_before": float(accuracy_before.detach().cpu()),
        "test_metric_after": float(accuracy_after.detach().cpu()),
        "test_metric_drop": float(accuracy_drop.detach().cpu()),
    }


def coordinate_intervention_errors(
    model: GraphAtlas,
    data: GraphData,
    seed: int = 1234,
    nonlinear: bool = True,
) -> tuple[float, float]:
    diagnostics = coordinate_intervention_diagnostics(model, data, seed, nonlinear)
    return diagnostics["mean_error"], diagnostics["max_logit_error"]


def coordinate_intervention_error(model: GraphAtlas, data: GraphData, seed: int = 1234, nonlinear: bool = True) -> float:
    return coordinate_intervention_errors(model, data, seed, nonlinear)[0]


def evaluate_output(
    model: torch.nn.Module,
    output: dict[str, torch.Tensor],
    data: GraphData,
    seed: int,
    include_intervention: bool = True,
    include_test: bool = True,
) -> dict[str, Any]:
    task_name = (data.metadata or {}).get("task", "node_classification")
    logits = output["logits"].detach()
    metric_name = primary_metric_name(data)
    if task_name == "link_prediction":
        embedding = output.get("embedding")
        if embedding is None:
            raise ValueError("Model output does not contain embedding for link prediction")
        train_link = link_prediction_metrics(embedding, data, "train")
        val_link = link_prediction_metrics(embedding, data, "val")
        test_link = link_prediction_metrics(embedding, data, "test") if include_test else {
            "roc_auc": float("nan"), "average_precision": float("nan"),
            "mrr": float("nan"), "hits_at_10": float("nan"), "hits_at_50": float("nan"),
        }
        metrics: dict[str, Any] = {
            "task": task_name,
            "metric_name": "roc_auc",
            "train_metric": train_link["roc_auc"],
            "val_metric": val_link["roc_auc"],
            "test_metric": test_link["roc_auc"],
            "train_link_roc_auc": train_link["roc_auc"],
            "val_link_roc_auc": val_link["roc_auc"],
            "test_link_roc_auc": test_link["roc_auc"],
            "train_link_average_precision": train_link["average_precision"],
            "val_link_average_precision": val_link["average_precision"],
            "test_link_average_precision": test_link["average_precision"],
            "train_link_mrr": train_link["mrr"],
            "val_link_mrr": val_link["mrr"],
            "test_link_mrr": test_link["mrr"],
            "test_link_hits_at_10": test_link["hits_at_10"],
            "test_link_hits_at_50": test_link["hits_at_50"],
            "train_accuracy": float("nan"),
            "val_accuracy": float("nan"),
            "test_accuracy": float("nan"),
            "train_roc_auc": float("nan"),
            "val_roc_auc": float("nan"),
            "test_roc_auc": float("nan"),
            "boundary_accuracy": float("nan"),
            "interior_accuracy": float("nan"),
        }
    else:
        split_values: dict[str, dict[str, float]] = {}
        for split in (("train", "val", "test") if include_test else ("train", "val")):
            mask = getattr(data, f"{split}_mask")
            split_values[split] = multilabel_metrics(logits, data.y, mask) if data.is_multilabel else single_label_metrics(logits, data.y, mask)
        metrics = {
            "task": task_name,
            "metric_name": metric_name,
            "train_metric": primary_metric(logits, data, data.train_mask),
            "val_metric": primary_metric(logits, data, data.val_mask),
            "test_metric": primary_metric(logits, data, data.test_mask) if include_test else float("nan"),
            "train_link_roc_auc": float("nan"), "val_link_roc_auc": float("nan"), "test_link_roc_auc": float("nan"),
            "train_link_average_precision": float("nan"), "val_link_average_precision": float("nan"), "test_link_average_precision": float("nan"),
            "train_link_mrr": float("nan"), "val_link_mrr": float("nan"), "test_link_mrr": float("nan"),
            "test_link_hits_at_10": float("nan"), "test_link_hits_at_50": float("nan"),
        }
        names = {key for values in split_values.values() for key in values}
        for split, values in split_values.items():
            for name in names:
                metrics[f"{split}_{name}"] = values.get(name, float("nan"))
        for required in ("accuracy", "roc_auc"):
            for split in ("train", "val", "test"):
                metrics.setdefault(f"{split}_{required}", float("nan"))
        if include_test and data.boundary_mask is not None and not data.is_multilabel:
            boundary_test = data.test_mask & data.boundary_mask
            interior_test = data.test_mask & ~data.boundary_mask
            metrics["boundary_accuracy"] = accuracy(logits, data.y, boundary_test)
            metrics["interior_accuracy"] = accuracy(logits, data.y, interior_test)
        else:
            metrics["boundary_accuracy"] = float("nan")
            metrics["interior_accuracy"] = float("nan")

    membership = output.get("membership")
    if membership is not None and data.chart_membership is not None:
        predicted_chart = membership.argmax(dim=-1).detach().cpu().numpy()
        true_chart = data.chart_membership.argmax(dim=-1).detach().cpu().numpy()
        metrics["chart_ari"] = float(adjusted_rand_score(true_chart, predicted_chart))
        predicted_overlap = ((membership > OVERLAP_THRESHOLD).sum(dim=-1) > 1).detach().cpu().numpy()
        true_overlap = ((data.chart_membership > OVERLAP_THRESHOLD).sum(dim=-1) > 1).detach().cpu().numpy()
        metrics["overlap_f1"] = float(f1_score(true_overlap, predicted_overlap, zero_division=0))
    else:
        metrics["chart_ari"] = float("nan")
        metrics["overlap_f1"] = float("nan")

    if isinstance(model, GraphAtlas):
        rec = output["reconstruction"] - output["observation"][:, None]
        metrics["reconstruction_error"] = float(rec.square().mean().detach().cpu())
        if include_intervention:
            affine = coordinate_intervention_diagnostics(model, data, seed + 1000, nonlinear=False)
            nonlinear = coordinate_intervention_diagnostics(model, data, seed + 2000, nonlinear=True)
            metrics["invariance_error_affine"] = affine["mean_error"]
            metrics["invariance_error_nonlinear"] = nonlinear["mean_error"]
            metrics["invariance_logit_max_affine"] = affine["max_logit_error"]
            metrics["invariance_logit_max_nonlinear"] = nonlinear["max_logit_error"]
            metrics["intervention_flip_rate_affine"] = affine["prediction_flip_rate"]
            metrics["intervention_flip_rate_nonlinear"] = nonlinear["prediction_flip_rate"]
            metrics["intervention_test_metric_drop_affine"] = affine["test_metric_drop"]
            metrics["intervention_test_metric_drop_nonlinear"] = nonlinear["test_metric_drop"]
            metrics["intervention_test_metric_nonlinear"] = nonlinear["test_metric_after"]
        else:
            for name in (
                "invariance_error_affine", "invariance_error_nonlinear",
                "invariance_logit_max_affine", "invariance_logit_max_nonlinear",
                "intervention_flip_rate_affine", "intervention_flip_rate_nonlinear",
                "intervention_test_metric_drop_affine", "intervention_test_metric_drop_nonlinear",
                "intervention_test_metric_nonlinear",
            ):
                metrics[name] = float("nan")
        if data.true_edge_lengths is not None:
            metrics.update(intrinsic_geometry_diagnostics(model, output, data, seed))
        else:
            for name in (
                "metric_recovery_error", "metric_recovery_mean_error", "transition_error",
                "local_distortion", "cross_chart_distortion",
            ):
                metrics[name] = float("nan")
    else:
        metrics["reconstruction_error"] = float("nan")
        for name in (
            "invariance_error_affine", "invariance_error_nonlinear",
            "invariance_logit_max_affine", "invariance_logit_max_nonlinear",
            "intervention_flip_rate_affine", "intervention_flip_rate_nonlinear",
            "intervention_test_metric_drop_affine", "intervention_test_metric_drop_nonlinear",
            "intervention_test_metric_nonlinear",
        ):
            metrics[name] = float("nan")
        for name in (
            "metric_recovery_error", "metric_recovery_mean_error", "transition_error",
            "local_distortion", "cross_chart_distortion",
        ):
            metrics[name] = float("nan")
    if data.geometry_region is not None:
        metrics["curvature_flat_fraction"] = float((data.geometry_region == 0).float().mean().cpu())
        metrics["curvature_positive_fraction"] = float((data.geometry_region == 1).float().mean().cpu())
        metrics["curvature_negative_fraction"] = float((data.geometry_region == 2).float().mean().cpu())
    else:
        metrics["curvature_flat_fraction"] = float("nan")
        metrics["curvature_positive_fraction"] = float("nan")
        metrics["curvature_negative_fraction"] = float("nan")
    return metrics
