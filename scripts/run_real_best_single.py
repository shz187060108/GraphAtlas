#!/usr/bin/env python
"""Run the eight real-stage datasets once with the best available HPO settings."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "real_best_single"
PRESET_DIR = ROOT / "configs" / "presets"
RESULT_PATH = ROOT / "outputs" / "results" / "real_best_single.csv"

REAL_DATASETS = (
    "actor",
    "questions",
    "dblp",
    "coauthor_cs",
    "coauthor_physics",
    "hetgb_texas",
    "hetgb_actor",
    "hetgb_amazon",
    "h2gb_pdns",
    "h2gb_mag_year",
    "h2gb_ieee_cis",
)

H2GB_DATASETS = {
    "h2gb_pdns": {"target_node_type": "domain_node", "metric": "f1"},
    "h2gb_mag_year": {"target_node_type": "paper", "metric": "accuracy"},
    "h2gb_ieee_cis": {"target_node_type": "transaction", "metric": "f1"},
}

# No direct HPO study exists for HeTGB yet. Transfer only from the closest
# already-tuned dataset and record the provenance explicitly in the manifest.
HPO_SOURCE = {
    "actor": "actor",
    "questions": "questions",
    "dblp": "dblp",
    "coauthor_cs": "coauthor_cs",
    "coauthor_physics": "coauthor_physics",
    "hetgb_texas": "actor",
    "hetgb_actor": "actor",
    "hetgb_amazon": "amazon_ratings",
}

LOSS_KEYS = (
    "reconstruction",
    "cocycle",
    "inverse_cycle",
    "path_consistency",
    "metric",
    "chart_rank",
    "cover",
    "balance",
    "sparsity",
    "geometry",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_tune(source: str) -> tuple[dict[str, Any], int, str, float]:
    promoted = ROOT / "outputs" / "selected_optuna" / "best_configs" / f"{source}.json"
    if promoted.exists():
        payload = _read_json(promoted)
        selected = payload["selected"]
        values = [float(value) for value in selected.get("values", [])]
        seed = max(range(len(values)), key=values.__getitem__) if values else 0
        return dict(selected["params"]), seed, "promote", max(values) if values else float("nan")

    candidates: list[tuple[float, int, dict[str, Any], Path]] = []
    tune_root = ROOT / "outputs" / "selected_optuna" / "tuning" / source
    for metrics_path in tune_root.glob("trial_*/metrics.json"):
        metrics = _read_json(metrics_path)
        value = metrics.get("best_val_metric")
        params = metrics.get("selected_params")
        if value is not None and params:
            candidates.append((float(value), int(metrics.get("model_seed", 0)), dict(params), metrics_path))
    if not candidates:
        raise FileNotFoundError(f"No completed tuning result is available for {source}: {tune_root}")
    value, seed, params, path = max(candidates, key=lambda item: item[0])
    return params, seed, str(path.relative_to(ROOT)), value


def _scaled_loss(scale: float) -> dict[str, float]:
    base = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    return {key: float(base["loss"][key]) * scale for key in LOSS_KEYS}


def _preset(dataset: str, params: dict[str, Any], seed: int) -> dict[str, Any]:
    if dataset in H2GB_DATASETS:
        details = H2GB_DATASETS[dataset]
        return {
            "base_config": "configs/base.yaml",
            "reporting": {"target_model": "atn"},
            "seeds": [seed],
            "train": {
                "device": "cuda",
                "mode": "hetero_neighbor",
                "batch_size": 256,
                "neighbor_sizes": [20, 15],
                "target_batching": True,
                "epochs": 200,
                "patience": 40,
                "eval_every": 5,
                "progress": True,
                "resume": True,
                "learning_rate": float(params["learning_rate"]),
                "weight_decay": float(params["weight_decay"]),
                "output_dir": "outputs/.work/runs_real_best_single",
            },
            "loss": _scaled_loss(float(params["geometry_loss_scale"])),
            "datasets": [{
                "name": dataset,
                "num_charts": 4,
                "split": seed,
                "heterogeneous": True,
                "target_node_type": details["target_node_type"],
                "metric": details["metric"],
            }],
            "models": [{
                "name": "atn",
                "label": "atn",
                "heterogeneous": True,
                "hidden_dim": int(params["hidden_dim"]),
                "dropout": float(params["dropout"]),
                "observation_dim": 24,
                "chart_dim": 4,
                "vector_channels": 6,
                "num_charts": 4,
                "num_layers": 1,
                "membership_topk": 2,
                "edge_chunk_size": 100000,
                "edge_chunk_threshold": 2000000,
            }],
        }
    dataset_entry: dict[str, Any] = {"name": dataset, "num_charts": 4, "split": seed}
    if dataset == "questions":
        dataset_entry["metric"] = "roc_auc"
    return {
        "base_config": "configs/base.yaml",
        "reporting": {"target_model": "graphatlas_c_oracle"},
        "seeds": [seed],
        "train": {
            "device": "cuda",
            "epochs": 200,
            "patience": 40,
            "eval_every": 5,
            "progress": True,
            "resume": True,
            "learning_rate": float(params["learning_rate"]),
            "weight_decay": float(params["weight_decay"]),
            "output_dir": "outputs/.work/runs_real_best_single",
        },
        "loss": _scaled_loss(float(params["geometry_loss_scale"])),
        "datasets": [dataset_entry],
        "models": [{
            "name": "graphatlas_certified",
            "label": "graphatlas_c_oracle",
            "certified_routing_mode": "oracle_max_q",
            "hidden_dim": int(params["hidden_dim"]),
            "dropout": float(params["dropout"]),
            "observation_dim": 24,
            "chart_dim": 4,
            "vector_channels": 6,
            "num_charts": 4,
            "num_layers": 1,
            "membership_topk": 2,
            "edge_chunk_size": 100000,
            "edge_chunk_threshold": 2000000,
        }],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=REAL_DATASETS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(REAL_DATASETS)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    result_frames: list[pd.DataFrame] = []

    for dataset in datasets:
        if dataset in H2GB_DATASETS:
            source = "h2gb_safe_default"
            params = {
                "learning_rate": 0.003,
                "weight_decay": 0.0005,
                "dropout": 0.05,
                "hidden_dim": 64,
                "geometry_loss_scale": 1.0,
            }
            seed, selection_source, validation = 0, "untuned_h2gb_default", float("nan")
        else:
            source = HPO_SOURCE[dataset]
            params, seed, selection_source, validation = _best_tune(source)
        preset_path = PRESET_DIR / f"real_best_single_{dataset}.yaml"
        preset_path.write_text(
            yaml.safe_dump(_preset(dataset, params, seed), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        manifest_rows.append({
            "dataset": dataset,
            "hpo_source_dataset": source,
            "seed": seed,
            "selection_source": selection_source,
            "source_validation_metric": validation,
            **params,
        })
        command = [sys.executable, "-X", "utf8", "-u", "scripts/run_all.py", "--preset", str(preset_path)]
        if args.dry_run:
            command.append("--dry-run")
        print(f"[real_best_single] {dataset}: seed={seed}, HPO source={source}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        result_file = ROOT / "outputs" / "results" / f"{preset_path.stem}.csv"
        if not args.dry_run and result_file.exists():
            result_frames.append(pd.read_csv(result_file))

    selection_path = OUTPUT / "selection_manifest.csv"
    selection_frame = pd.DataFrame(manifest_rows)
    if selection_path.exists():
        selection_frame = pd.concat([pd.read_csv(selection_path), selection_frame], ignore_index=True, sort=False)
    selection_frame.drop_duplicates(subset=["dataset"], keep="last").sort_values("dataset").to_csv(
        selection_path, index=False
    )
    if result_frames:
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(result_frames, ignore_index=True, sort=False)
        if RESULT_PATH.exists():
            combined = pd.concat([pd.read_csv(RESULT_PATH), combined], ignore_index=True, sort=False)
        keys = [key for key in ("dataset", "model", "seed", "split") if key in combined]
        if keys:
            combined = combined.drop_duplicates(subset=keys, keep="last").sort_values(keys)
        combined.to_csv(RESULT_PATH, index=False)


if __name__ == "__main__":
    main()
