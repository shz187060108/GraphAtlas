#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from graphatlas.config import ExperimentConfig
from graphatlas.datasets import load_dataset
from graphatlas.nn.model import GraphAtlas, build_model
from graphatlas.utils import seed_everything


ROOT = Path(PROJECT_ROOT)


def _set_transport(model: GraphAtlas, mode: str, beta: float) -> None:
    model.transport_mode = mode
    for layer in model.layers:
        layer.transport_mode = mode
        layer.transportability_beta = beta


def _diagnostics(output: dict[str, object]) -> dict[str, float]:
    layers = output["layer_diagnostics"]
    names = (
        "transportability_mean", "transportability_std", "distortion_mean",
        "q_route_spread_mean", "q_route_spread_median", "q_route_spread_p90",
        "q_route_spread_max", "routing_entropy_mean", "routing_kl_from_membership",
        "routing_total_variation_from_membership", "active_source_chart_count",
        "active_target_chart_count", "fraction_edges_with_multiple_source_charts",
        "fraction_edges_with_multiple_target_charts", "fraction_q_below_025",
        "fraction_q_above_075",
    )
    result = {}
    for name in names:
        values = [entry[name] for entry in layers if name in entry]
        if values:
            result[name] = float(torch.stack(values).mean().cpu())
    membership = output["membership"]
    usage = membership.mean(dim=0)
    entropy = -(usage.clamp_min(1e-12) * usage.clamp_min(1e-12).log()).sum()
    active = (membership > 0).sum(dim=-1)
    result.update({
        **{f"chart_usage_{index}": float(value.cpu()) for index, value in enumerate(usage)},
        "chart_usage_min": float(usage.min().cpu()),
        "chart_usage_max": float(usage.max().cpu()),
        "chart_usage_entropy": float(entropy.cpu()),
        "fraction_nodes_with_1_active_chart": float((active == 1).float().mean().cpu()),
        "fraction_nodes_with_2_or_more_active_charts": float((active >= 2).float().mean().cpu()),
    })
    return result


def _gradient_snapshot(model: GraphAtlas, data, mode: str, beta: float):
    original_x = data.x
    data.x = original_x.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    _set_transport(model, mode, beta)
    loss = model(data)["logits"].square().mean()
    loss.backward()
    input_gradient = data.x.grad.detach().clone()
    parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    data.x = original_x
    return input_gradient, parameter_gradients


def _parameter_error(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    names = set(left) & set(right)
    return max((left[name] - right[name]).abs().max().item() for name in names)


def _checkpoint_rows(manifest: Path, datasets: set[str]) -> list[dict[str, str]]:
    frame = pd.read_csv(manifest)
    frame = frame[(frame.dataset.isin(datasets)) & (frame.model == "graphatlas_certified")]
    rows = []
    for row in frame.to_dict("records"):
        run_dir = ROOT / str(row["run_dir"])
        if (run_dir / "best.pt").exists() and (run_dir / "metrics.json").exists():
            rows.append({**row, "run_dir": str(run_dir)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact same-checkpoint certified routing diagnostics.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", default="outputs/manifests/certified_real_pilot.csv")
    parser.add_argument("--output-dir", default="outputs/reports/certified_diagnostics_exact")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    records = []

    for row in _checkpoint_rows(ROOT / args.manifest, set(args.datasets)):
        run_dir = Path(row["run_dir"])
        config = ExperimentConfig.from_yaml(run_dir / "config.yaml")
        config.train.device = "cpu"
        seed_everything(args.seed)
        data = load_dataset(config.dataset, args.seed)
        model = build_model(data.num_features, data.num_classes, config.model, data.num_nodes)
        if not isinstance(model, GraphAtlas):
            continue
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
        model.eval()

        with torch.no_grad():
            _set_transport(model, "min_distortion", 0.0)
            minimum = model(data)
            _set_transport(model, "certified", 0.0)
            beta_zero = model(data, transport_diagnostics=True)
            _set_transport(model, "certified", 1.0)
            beta_one = model(data, transport_diagnostics=True)

        min_input, min_parameters = _gradient_snapshot(model, data, "min_distortion", 0.0)
        zero_input, zero_parameters = _gradient_snapshot(model, data, "certified", 0.0)
        record = {
            "dataset": row["dataset"],
            "seed": args.seed,
            "beta0_logits_max_abs_error": float((minimum["logits"] - beta_zero["logits"]).abs().max()),
            "beta0_embeddings_max_abs_error": float((minimum["embedding"] - beta_zero["embedding"]).abs().max()),
            "beta0_input_gradient_max_abs_error": float((min_input - zero_input).abs().max()),
            "beta0_parameter_gradient_max_abs_error": _parameter_error(min_parameters, zero_parameters),
            "beta1_vs_beta0_logits_linf": float((beta_one["logits"] - beta_zero["logits"]).abs().max()),
            "beta1_vs_beta0_embeddings_l2": float(torch.linalg.vector_norm(beta_one["embedding"] - beta_zero["embedding"])),
            **_diagnostics(beta_one),
        }
        record["routing_kl_beta1_from_beta0"] = record["routing_kl_from_membership"]
        record["routing_total_variation_beta1_from_beta0"] = record["routing_total_variation_from_membership"]
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    frame = pd.DataFrame(records).sort_values("dataset")
    frame.to_csv(output / "certified_diagnostics_exact.csv", index=False)
    (output / "certified_diagnostics_exact.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"datasets": len(frame), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
