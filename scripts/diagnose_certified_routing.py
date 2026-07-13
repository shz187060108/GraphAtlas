#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from graphatlas.config import ExperimentConfig
from graphatlas.datasets import load_dataset
from graphatlas.nn.model import GraphAtlas, build_model
from graphatlas.utils import count_parameters, seed_everything


ROOT = Path(PROJECT_ROOT)


def _set_transport(model: GraphAtlas, mode: str, beta: float) -> None:
    model.transport_mode = mode
    for layer in model.layers:
        layer.transport_mode = mode
        layer.transportability_beta = beta


def _layer_scalars(output: dict[str, object]) -> dict[str, float]:
    names = (
        "transportability_mean", "transportability_std", "transportability_min",
        "transportability_max", "distortion_mean", "fraction_q_below_025",
        "fraction_q_above_075", "active_source_chart_count",
        "q_route_spread_mean", "q_route_spread_p90",
        "routing_kl_from_membership", "fraction_edges_with_multiple_source_charts",
    )
    layers = output.get("layer_diagnostics", [])
    return {
        name: float(torch.stack([entry[name] for entry in layers if name in entry]).mean().cpu())
        for name in names
        if any(name in entry for entry in layers)
    }


def _gradients(model: GraphAtlas, data, mode: str, beta: float) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    _set_transport(model, mode, beta)
    model(data)["logits"].square().mean().backward()
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _gradient_error(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> tuple[float, float]:
    shared = sorted(set(left) & set(right))
    maximum = max((left[name] - right[name]).abs().max().item() for name in shared)
    difference = torch.sqrt(sum((left[name] - right[name]).square().sum() for name in shared))
    reference = torch.sqrt(sum(left[name].square().sum() for name in shared)).clamp_min(1e-12)
    return maximum, float((difference / reference).cpu())


def _completed_rows(manifest_path: Path) -> list[dict[str, str]]:
    rows = pd.read_csv(manifest_path).to_dict("records")
    completed = []
    for row in rows:
        run_dir = ROOT / str(row["run_dir"])
        if str(row["model"]) == "graphatlas_certified" and (run_dir / "metrics.json").exists():
            completed.append({**row, "run_dir": str(run_dir)})
    return completed


def _plot(frame: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    labels = frame["dataset"].str.replace("_filtered", "", regex=False).str.replace("_", " ", regex=False)
    x = np.arange(len(frame))
    colors = {"original": "#6B7280", "min_distortion": "#7AA6C2", "certified": "#D58A5A"}
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.1), constrained_layout=True)

    width = 0.25
    for offset, column, label, color in (
        (-width, "original_test", "Original", colors["original"]),
        (0, "min_test", "Min-dist.", colors["min_distortion"]),
        (width, "certified_test", "Certified", colors["certified"]),
    ):
        axes[0, 0].bar(x + offset, frame[column], width, label=label, color=color)
    axes[0, 0].set_ylabel("Test metric")
    axes[0, 0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0, 0].legend(ncol=3, fontsize=6)
    axes[0, 0].set_title("a  Predictive performance", loc="left", fontweight="bold")

    axes[0, 1].errorbar(x, frame["transportability_mean"], yerr=frame["transportability_std"], fmt="o", color="#D58A5A", capsize=2)
    axes[0, 1].axhline(1.0, color="#9CA3AF", lw=0.8, ls="--")
    axes[0, 1].set_ylim(0, 1.04)
    axes[0, 1].set_xticks(x, labels, rotation=35, ha="right")
    axes[0, 1].set_ylabel("Transportability q")
    axes[0, 1].set_title("b  Certificate saturation", loc="left", fontweight="bold")

    axes[0, 2].bar(x, frame["active_source_chart_count"], color="#7AA6C2", label="Active source charts")
    axes[0, 2].plot(x, frame["fraction_edges_with_multiple_source_charts"], "o-", color="#B45309", label="Multi-chart edge fraction")
    axes[0, 2].set_xticks(x, labels, rotation=35, ha="right")
    axes[0, 2].set_ylim(0, max(2.1, frame["active_source_chart_count"].max() * 1.15))
    axes[0, 2].legend(fontsize=6)
    axes[0, 2].set_title("c  Available routing choices", loc="left", fontweight="bold")

    axes[1, 0].bar(x - width / 2, frame["q_route_spread_mean"], width, color="#A7C7B7", label="Mean")
    axes[1, 0].bar(x + width / 2, frame["q_route_spread_p90"], width, color="#4F8A70", label="P90")
    axes[1, 0].set_xticks(x, labels, rotation=35, ha="right")
    axes[1, 0].set_ylabel("max q − min q")
    axes[1, 0].legend(fontsize=6)
    axes[1, 0].set_title("d  Within-route q spread", loc="left", fontweight="bold")

    axes[1, 1].scatter(frame["q_route_spread_mean"], frame["routing_kl_from_membership"], c="#D58A5A", s=28)
    for _, row in frame.iterrows():
        axes[1, 1].annotate(row["dataset"].split("_")[0], (row["q_route_spread_mean"], row["routing_kl_from_membership"]), xytext=(3, 2), textcoords="offset points", fontsize=6)
    axes[1, 1].set_xlabel("Mean q route spread")
    axes[1, 1].set_ylabel("KL(route β=1 || membership)")
    axes[1, 1].set_title("e  Geometric routing effect", loc="left", fontweight="bold")

    floor = 1e-12
    axes[1, 2].bar(x - width / 2, np.maximum(frame["beta0_min_logits_max_abs"], floor), width, color="#7AA6C2", label="β=0 vs min-dist.")
    axes[1, 2].bar(x + width / 2, np.maximum(frame["beta1_beta0_logits_max_abs"], floor), width, color="#D58A5A", label="β=1 vs β=0")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xticks(x, labels, rotation=35, ha="right")
    axes[1, 2].set_ylabel("Max |Δ logits|")
    axes[1, 2].legend(fontsize=6)
    axes[1, 2].set_title("f  Same-weight output sensitivity", loc="left", fontweight="bold")

    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose certificate routing on completed real checkpoints.")
    parser.add_argument("--manifest", default="outputs/manifests/certified_real_pilot.csv")
    parser.add_argument("--output-dir", default="outputs/reports/certified_diagnostics")
    parser.add_argument("--gradient-dataset", default="cornell")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        frame = pd.read_csv(output_dir / "certified_diagnostics.csv")
        _plot(frame, output_dir / "certified_diagnostics")
        print(json.dumps({"rows": len(frame), "svg": str(output_dir / "certified_diagnostics.svg")}, ensure_ascii=False))
        return
    records = []
    gradient_summary = {}

    for row in _completed_rows(ROOT / args.manifest):
        run_dir = Path(row["run_dir"])
        config = ExperimentConfig.from_yaml(run_dir / "config.yaml")
        config.train.device = "cpu"
        seed_everything(config.train.seed)
        data = load_dataset(config.dataset, config.train.seed)
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

        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        family = {}
        manifest = pd.read_csv(ROOT / args.manifest)
        for label, key in (("graphatlas", "original_test"), ("graphatlas_min_distortion", "min_test")):
            match = manifest[(manifest.dataset == row["dataset"]) & (manifest.model == label)]
            if not match.empty:
                path = ROOT / str(match.iloc[0].run_dir) / "metrics.json"
                if path.exists():
                    family[key] = json.loads(path.read_text(encoding="utf-8"))["test_metric"]
        record = {
            "dataset": row["dataset"],
            "certified_test": metrics["test_metric"],
            "parameters": count_parameters(model),
            "beta0_min_logits_max_abs": float((beta_zero["logits"] - minimum["logits"]).abs().max()),
            "beta0_min_embedding_max_abs": float((beta_zero["embedding"] - minimum["embedding"]).abs().max()),
            "beta1_beta0_logits_max_abs": float((beta_one["logits"] - beta_zero["logits"]).abs().max()),
            **family,
            **_layer_scalars(beta_one),
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if row["dataset"] == args.gradient_dataset:
            minimum_gradients = _gradients(model, data, "min_distortion", 0.0)
            beta_zero_gradients = _gradients(model, data, "certified", 0.0)
            maximum, relative = _gradient_error(minimum_gradients, beta_zero_gradients)
            gradient_summary = {"dataset": row["dataset"], "gradient_max_abs": maximum, "gradient_relative_l2": relative}

    frame = pd.DataFrame(records).sort_values("dataset").reset_index(drop=True)
    frame.to_csv(output_dir / "certified_diagnostics.csv", index=False)
    payload = {"datasets": records, "gradient_equivalence": gradient_summary}
    (output_dir / "certified_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plot(frame, output_dir / "certified_diagnostics")
    print(json.dumps({"rows": len(frame), "gradient_equivalence": gradient_summary, "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
