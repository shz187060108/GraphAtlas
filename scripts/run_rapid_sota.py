#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from graphatlas.config import ExperimentConfig
from graphatlas.datasets import load_dataset
from graphatlas.metrics import output_primary_metric
from graphatlas.nn.model import build_model
from graphatlas.utils import count_parameters, deep_update, load_yaml, save_json, seed_everything


ROOT = Path(PROJECT_ROOT)
RESULT_COLUMNS = [
    "dataset", "model", "config_id", "seed", "stage", "selected",
    "best_epoch", "validation_metric", "test_metric", "runtime_seconds",
    "parameter_count", "peak_cuda_memory_bytes", "status", "error",
]


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _task_loss(logits: torch.Tensor, data) -> torch.Tensor:
    if data.is_multilabel:
        target = data.y[data.train_mask].to(logits.dtype)
        valid = torch.isfinite(target)
        return F.binary_cross_entropy_with_logits(logits[data.train_mask][valid], target[valid])
    return F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])


def _fit(
    config: ExperimentConfig,
    data,
    run_dir: Path,
    epochs: int,
    patience: int,
    eval_every: int,
) -> dict[str, Any]:
    metadata_path = run_dir / "result.json"
    checkpoint_path = run_dir / "best.pt"
    if metadata_path.exists() and checkpoint_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    seed_everything(config.train.seed)
    device = torch.device(config.train.device)
    model = build_model(data.num_features, data.num_classes, config.model, data.num_nodes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )
    best_val = -math.inf
    best_epoch = 0
    stale_epochs = 0
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = _task_loss(output["logits"], data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()

        if epoch % eval_every != 0 and epoch != epochs:
            continue
        model.eval()
        with torch.no_grad():
            val = output_primary_metric(model(data), data, "val")
        if val > best_val + 1e-12:
            best_val = val
            best_epoch = epoch
            stale_epochs = 0
            _atomic_torch_save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += eval_every
        if stale_epochs >= patience:
            break

    result = {
        "best_epoch": best_epoch,
        "validation_metric": best_val,
        "runtime_seconds": time.time() - started,
        "parameter_count": count_parameters(model),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "test_metric": None,
        "status": "completed",
        "error": "",
    }
    save_json(result, metadata_path)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _test_selected(config: ExperimentConfig, data, checkpoint: Path) -> float:
    model = build_model(data.num_features, data.num_classes, config.model, data.num_nodes).to(data.x.device)
    model.load_state_dict(torch.load(checkpoint, map_location=data.x.device, weights_only=True))
    model.eval()
    with torch.no_grad():
        value = output_primary_metric(model(data), data, "test")
    del model
    if data.x.device.type == "cuda":
        torch.cuda.empty_cache()
    return value


def _configuration(base: dict[str, Any], preset: dict[str, Any], dataset: dict[str, Any], model: dict[str, Any], candidate: dict[str, Any]) -> ExperimentConfig:
    payload = copy.deepcopy(base)
    payload = deep_update(payload, {"dataset": dataset, "model": model, "train": preset.get("train", {})})
    payload = deep_update(payload, {
        "model": {
            "hidden_dim": candidate["hidden_dim"],
            "num_layers": candidate["num_layers"],
            "dropout": candidate["dropout"],
        },
        "train": {
            "seed": 0,
            "learning_rate": candidate["learning_rate"],
            "weight_decay": candidate["weight_decay"],
        },
    })
    payload["model"]["num_charts"] = payload["dataset"]["num_charts"]
    return ExperimentConfig.from_dict(payload)


def _write_results(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rapid real-dataset baseline successive halving.")
    parser.add_argument("--preset", default="configs/presets/rapid_sota_baselines.yaml")
    args = parser.parse_args()
    preset_path = Path(args.preset)
    if not preset_path.is_absolute():
        preset_path = ROOT / preset_path
    preset = load_yaml(preset_path)
    base = load_yaml(ROOT / preset["base_config"])
    device_name = str(preset.get("train", {}).get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("rapid_sota_baselines requires CUDA; CPU fallback is disabled")

    output_root = ROOT / "outputs" / "rapid_sota" / "baselines"
    result_path = ROOT / "outputs" / "results" / "rapid_sota_baselines_search.csv"
    selected_path = ROOT / "outputs" / "results" / "rapid_sota_baselines.csv"
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    configs: dict[tuple[str, str, str], ExperimentConfig] = {}
    data_cache = {}
    stage_a = preset["search"]["stage_a"]
    stage_b = preset["search"]["stage_b"]

    for dataset_override in preset["datasets"]:
        dataset_name = dataset_override["name"]
        template = _configuration(base, preset, dataset_override, preset["models"][0], preset["search"]["configurations"][0])
        data = load_dataset(template.dataset, 0).to(torch.device(device_name))
        data_cache[dataset_name] = data
        for model_override in preset["models"]:
            model_name = model_override["name"]
            candidates = []
            for candidate in preset["search"]["configurations"]:
                config = _configuration(base, preset, dataset_override, model_override, candidate)
                config.validate()
                config_id = candidate["id"]
                configs[(dataset_name, model_name, config_id)] = config
                run_dir = output_root / dataset_name / model_name / config_id / "stage_a"
                try:
                    result = _fit(config, data, run_dir, **stage_a)
                except Exception as error:  # noqa: BLE001
                    result = {"best_epoch": 0, "validation_metric": -math.inf, "test_metric": None, "runtime_seconds": 0.0, "parameter_count": 0, "peak_cuda_memory_bytes": 0, "status": "failed", "error": repr(error)}
                row = {"dataset": dataset_name, "model": model_name, "config_id": config_id, "seed": 0, "stage": "A", "selected": False, **result}
                rows.append(row)
                candidates.append(row)
                _write_results(rows, result_path)
                print(json.dumps(row, ensure_ascii=False), flush=True)

            top = sorted(candidates, key=lambda item: item["validation_metric"], reverse=True)[:2]
            for item in top:
                item["selected"] = True
                config_id = item["config_id"]
                config = configs[(dataset_name, model_name, config_id)]
                run_dir = output_root / dataset_name / model_name / config_id / "stage_b"
                try:
                    result = _fit(config, data, run_dir, **stage_b)
                except Exception as error:  # noqa: BLE001
                    result = {"best_epoch": 0, "validation_metric": -math.inf, "test_metric": None, "runtime_seconds": 0.0, "parameter_count": 0, "peak_cuda_memory_bytes": 0, "status": "failed", "error": repr(error)}
                row = {"dataset": dataset_name, "model": model_name, "config_id": config_id, "seed": 0, "stage": "B", "selected": False, **result}
                rows.append(row)
                _write_results(rows, result_path)
                print(json.dumps(row, ensure_ascii=False), flush=True)

            stage_b_rows = [row for row in rows if row["dataset"] == dataset_name and row["model"] == model_name and row["stage"] == "B"]
            winner = max(stage_b_rows, key=lambda item: item["validation_metric"])
            winner["selected"] = True
            winner_config = configs[(dataset_name, model_name, winner["config_id"])]
            winner_checkpoint = output_root / dataset_name / model_name / winner["config_id"] / "stage_b" / "best.pt"
            winner["test_metric"] = _test_selected(winner_config, data, winner_checkpoint)
            selected_rows.append(dict(winner))
            _write_results(rows, result_path)
            _write_results(selected_rows, selected_path)

    selected = pd.DataFrame(selected_rows)
    selected["validation_rank"] = selected.groupby("dataset")["validation_metric"].rank(ascending=False, method="average")
    ranks = selected.groupby("model")["validation_rank"].mean().sort_values()
    eligible = [name for name in ("acmgcn", "gprgnn", "h2gcn", "appnp") if name in ranks]
    backbone = min(eligible, key=lambda name: ranks[name])
    gate = bool(
        selected[(selected.dataset == "chameleon_filtered")].test_metric.max() >= 0.35
        and selected[(selected.dataset == "squirrel_filtered")].test_metric.max() >= 0.34
    )
    summary = {
        "protocol_gate_passed": gate,
        "universal_backbone": backbone,
        "average_validation_rank": {name: float(value) for name, value in ranks.items()},
        "selected_results": str(selected_path),
    }
    save_json(summary, ROOT / "outputs" / "reports" / "rapid_sota_baselines" / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
