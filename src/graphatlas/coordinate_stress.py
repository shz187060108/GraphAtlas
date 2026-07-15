"""Checkpoint-only coordinate reparameterization stress evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .nn.charts import make_reparameterization


DEFAULT_KINDS = ("affine", "asinh_affine", "triangular_coupling", "radial")
DEFAULT_STRENGTHS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
REQUIRED_COLUMNS = (
    "dataset", "task", "model", "seed", "split", "run_id", "transform_kind",
    "transform_strength", "transform_seed", "mean_probability_error", "max_logit_error",
    "prediction_flip_rate", "test_metric_before", "test_metric_after", "test_metric_drop",
)


@dataclass
class StressResult:
    rows: list[dict[str, object]]
    skipped: list[dict[str, object]]


def _metric(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if labels.ndim == 2:
        probability = torch.sigmoid(logits)
        prediction = (probability >= 0.5).to(labels.dtype)
        return float((prediction[mask] == labels[mask]).float().mean().cpu())
    return float((logits[mask].argmax(-1) == labels[mask]).float().mean().cpu())


def _probabilities(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.ndim == 2:
        return torch.sigmoid(logits), (logits >= 0).to(labels.dtype)
    return torch.softmax(logits, dim=-1), logits.argmax(-1)


def _build_reps(model: torch.nn.Module, kind: str, strength: float, seed: int, device: torch.device, dtype: torch.dtype):
    count = int(model.config.num_charts)
    return [make_reparameterization(kind, int(model.config.chart_dim), device, dtype, seed + chart, strength) for chart in range(count)]


def evaluate_model_checkpoint(model: torch.nn.Module, data, *, run_id: str, dataset: str, task: str,
                             seed: int, split: int, kinds: Iterable[str] = DEFAULT_KINDS,
                             strengths: Iterable[float] = DEFAULT_STRENGTHS,
                             transform_seeds: Iterable[int] = range(10)) -> StressResult:
    """Evaluate a loaded GraphAtlas checkpoint without optimizer/training state."""
    if not hasattr(model, "config") or not hasattr(model, "charts"):
        return StressResult([], [{"run_id": run_id, "reason": "unsupported_model"}])
    metadata = getattr(data, "metadata", None) or {}
    if getattr(data, "is_multilabel", False) or metadata.get("task") == "link_prediction":
        return StressResult([], [{"run_id": run_id, "reason": "stress currently supports node classification"}])
    model.eval()
    mask = data.test_mask
    with torch.inference_mode():
        original = model(data, compact=True)
        base_probability, base_prediction = _probabilities(original["logits"], data.y)
        before = _metric(original["logits"], data.y, mask)
    rows: list[dict[str, object]] = []
    for kind in kinds:
        for strength in strengths:
            for transform_seed in transform_seeds:
                try:
                    reps = _build_reps(model, kind, float(strength), int(transform_seed), data.x.device, data.x.dtype)
                    with torch.inference_mode():
                        transformed = model(data, reparameterizations=reps, compact=True)
                        probability, prediction = _probabilities(transformed["logits"], data.y)
                        test_mask = mask
                        probability_error = (probability[test_mask] - base_probability[test_mask]).abs().mean(dim=-1)
                        logit_error = (transformed["logits"][test_mask] - original["logits"][test_mask]).abs()
                        flips = (prediction[test_mask] != base_prediction[test_mask]).any(dim=-1) if data.y.ndim == 2 else prediction[test_mask] != base_prediction[test_mask]
                        after = _metric(transformed["logits"], data.y, mask)
                    rows.append({
                        "dataset": dataset, "task": task, "model": getattr(model.config, "label", None) or model.config.name,
                        "seed": int(seed), "split": int(split), "run_id": run_id,
                        "transform_kind": kind, "transform_strength": float(strength), "transform_seed": int(transform_seed),
                        "mean_probability_error": float(probability_error.mean().cpu()),
                        "max_probability_error": float(probability_error.max().cpu()) if probability_error.numel() else 0.0,
                        "mean_logit_error": float(logit_error.mean().cpu()), "max_logit_error": float(logit_error.max().cpu()) if logit_error.numel() else 0.0,
                        "prediction_flip_rate": float(flips.float().mean().cpu()), "test_metric_before": before,
                        "test_metric_after": after, "test_metric_drop": before - after,
                        "identity_check": bool(float(strength) == 0.0),
                    })
                except Exception as exc:  # keep one malformed transform from hiding the rest
                    rows.append({"dataset": dataset, "task": task, "model": getattr(model.config, "name", "unknown"), "seed": int(seed), "split": int(split), "run_id": run_id, "transform_kind": kind, "transform_strength": float(strength), "transform_seed": int(transform_seed), "status": "failed", "skip_reason": f"{type(exc).__name__}: {exc}"})
    return StressResult(rows, [])


def normalize_stress_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable long-table schema even when all runs were skipped."""
    out = frame.copy()
    for column in REQUIRED_COLUMNS:
        if column not in out:
            out[column] = np.nan
    return out[[*REQUIRED_COLUMNS, *[c for c in out.columns if c not in REQUIRED_COLUMNS]]]
