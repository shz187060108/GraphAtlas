#!/usr/bin/env python
"""Evaluate coordinate reparameterizations from completed checkpoints."""
from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from graphatlas.config import ExperimentConfig
from graphatlas.coordinate_stress import DEFAULT_KINDS, DEFAULT_STRENGTHS, evaluate_model_checkpoint, normalize_stress_frame
from graphatlas.datasets import load_dataset
from graphatlas.nn.model import build_model


def _run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [Path(value) for value in args.run_dir]
    if args.results:
        frame = pd.read_csv(args.results)
        values = frame.get("run_dir", pd.Series(dtype=str)).dropna().astype(str).unique()
        return [Path(value) for value in values]
    return sorted(Path(args.runs_root).rglob("metrics.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append")
    parser.add_argument("--run-dirs", nargs="+")
    parser.add_argument("--results")
    parser.add_argument("--runs-root", default="outputs/.work/runs")
    parser.add_argument("--output", "--output-csv", dest="output", default="outputs/results/coordinate_stress.csv")
    parser.add_argument("--manifest", default="outputs/results/coordinate_stress_manifest.json")
    parser.add_argument("--skipped", "--skip-csv", dest="skipped", default="outputs/results/coordinate_stress_skipped.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--kinds", nargs="+", default=list(DEFAULT_KINDS))
    parser.add_argument("--strengths", nargs="+", type=float, default=list(DEFAULT_STRENGTHS))
    parser.add_argument("--transform-seeds", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    device = torch.device(args.device)
    if args.run_dirs:
        args.run_dir = args.run_dirs
    for metrics_path in _run_dirs(args):
        run_dir = metrics_path.parent if metrics_path.name == "metrics.json" else metrics_path
        try:
            config_path = run_dir / "config.yaml"
            checkpoint = run_dir / "best.pt"
            if not config_path.exists() or not checkpoint.exists():
                raise FileNotFoundError("config.yaml or best.pt missing")
            config = ExperimentConfig.from_yaml(config_path); config.validate()
            data = load_dataset(config.dataset, config.train.seed).to(device)
            model = build_model(data.num_features, data.num_classes, config.model, data.num_nodes).to(device)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state.get("model", state), strict=False)
            result = evaluate_model_checkpoint(model, data, run_id=run_dir.name, dataset=config.dataset.name, task=config.dataset.task, seed=config.train.seed, split=config.dataset.split, kinds=args.kinds, strengths=args.strengths, transform_seeds=args.transform_seeds)
            rows.extend(result.rows); skipped.extend(result.skipped)
        except Exception as exc:
            skipped.append({"run_dir": str(run_dir), "reason": f"{type(exc).__name__}: {exc}"})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    normalize_stress_frame(pd.DataFrame(rows)).to_csv(output, index=False)
    skipped_path = Path(args.skipped); skipped_path.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(skipped).to_csv(skipped_path, index=False)
    manifest = {"output": str(output), "skipped": str(skipped_path), "runs": len(_run_dirs(args)), "rows": len(rows), "kinds": args.kinds, "strengths": args.strengths, "transform_seeds": args.transform_seeds, "device": str(device)}
    manifest_path = Path(args.manifest); manifest_path.parent.mkdir(parents=True, exist_ok=True); manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
