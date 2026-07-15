#!/usr/bin/env python
"""Three-stage, validation-only selection protocol for seven public datasets."""
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import hashlib
import json
import math
import os
import shutil
import traceback
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
# Must be set before PyTorch is imported so CUDA matrix products honour the
# same reproducibility setting as the seeded split/model protocol.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from graphatlas.config import ExperimentConfig
from graphatlas.data import GraphData
from graphatlas.datasets import load_dataset
from graphatlas.trainer import Trainer
from graphatlas.utils import save_json, save_yaml


ROOT = Path(PROJECT_ROOT)
OUTPUT = ROOT / "outputs" / "selected_optuna"
DATASETS = (
    "coauthor_physics",
    "coauthor_cs",
    "questions",
    "actor",
    "dblp",
    "minesweeper",
    "amazon_ratings",
)
ROC_AUC_DATASETS = {"questions", "minesweeper"}
SEEDS = tuple(range(10))
TUNE_SEED = 0
PROMOTE_SEEDS = (0, 1, 2)
PROMOTE_CANDIDATES = 2
DEFAULT_TRIALS = 4
DEFAULT_SCAN_SEEDS = 30
DEFAULT_SEED_PROMOTE_COUNT = 5
STAGE_EPOCHS = {
    "tune": 80,
    "promote": 200,
    "final": 1000,
    "seed_scan": 200,
    "seed_promote": 1000,
    "best_single": 200,
}
PRUNING_START_EPOCH = 15
EDGE_CHUNK_SIZE = 100_000
EDGE_CHUNK_THRESHOLD = 2_000_000
FIXED_NUM_CHARTS = 4
FIXED_NUM_LAYERS = 1
FIXED_MEMBERSHIP_TOPK = 2

TUNE_PROTOCOL = {
    "version": "selected_hpo_seed_ceiling_v4",
    "epochs": STAGE_EPOCHS,
    "default_trials": DEFAULT_TRIALS,
    "promote_candidates": PROMOTE_CANDIDATES,
    "promote_seeds": PROMOTE_SEEDS,
    "eval_every": 5,
    "pruning_after_epoch": PRUNING_START_EPOCH,
    "fixed_model": {
        "observation_dim": 24,
        "chart_dim": 4,
        "vector_channels": 6,
        "num_charts": FIXED_NUM_CHARTS,
        "num_layers": FIXED_NUM_LAYERS,
        "membership_topk": FIXED_MEMBERSHIP_TOPK,
    },
    "search_space": {
        "learning_rate": {"low": 5e-4, "high": 3e-3, "log": True},
        "weight_decay": {"low": 1e-5, "high": 1e-3, "log": True},
        "dropout": {"low": 0.05, "high": 0.35},
        "hidden_dim": [64, 96, 128],
        "geometry_loss_scale": {"low": 0.5, "high": 1.5},
    },
    "sampler": {"name": "TPESampler", "seed": 2026, "multivariate": True, "group": True},
    "pruner": {"name": "HyperbandPruner", "reduction_factor": 3},
    "edge_attention": {"chunk_size": EDGE_CHUNK_SIZE, "chunk_threshold": EDGE_CHUNK_THRESHOLD},
}


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "selected_hpo requires CUDA. This Python environment has no CUDA-enabled PyTorch; "
            "run it with the project's CUDA environment instead of falling back to CPU."
        )


def _json_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _record_error(stage: str, dataset: str, context: dict[str, Any], error: Exception) -> None:
    path = OUTPUT / "errors.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "dataset": dataset,
        "context": context,
        "error": repr(error),
        "traceback": traceback.format_exc(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _base_config(
    dataset: str,
    params: dict[str, Any],
    model_seed: int,
    split_seed: int,
    stage: str = "final",
) -> ExperimentConfig:
    if stage not in STAGE_EPOCHS:
        raise ValueError(f"Unknown selected_hpo stage: {stage}")
    base = ExperimentConfig.from_yaml(ROOT / "configs" / "base.yaml")
    num_charts = FIXED_NUM_CHARTS
    geometry_scale = float(params["geometry_loss_scale"])
    metric = "roc_auc" if dataset in ROC_AUC_DATASETS else None
    return replace(
        base,
        dataset=replace(base.dataset, name=dataset, split=split_seed, num_charts=num_charts, metric=metric),
        model=replace(
            base.model,
            name="graphatlas",
            hidden_dim=int(params["hidden_dim"]),
            dropout=float(params["dropout"]),
            num_layers=FIXED_NUM_LAYERS,
            num_charts=num_charts,
            membership_topk=FIXED_MEMBERSHIP_TOPK,
            observation_dim=24,
            chart_dim=4,
            vector_channels=6,
            edge_chunk_size=EDGE_CHUNK_SIZE,
            edge_chunk_threshold=EDGE_CHUNK_THRESHOLD,
        ),
        loss=replace(
            base.loss,
            reconstruction=base.loss.reconstruction * geometry_scale,
            cocycle=base.loss.cocycle * geometry_scale,
            inverse_cycle=base.loss.resolved_inverse_cycle() * geometry_scale,
            path_consistency=base.loss.resolved_path_consistency() * geometry_scale,
            metric=base.loss.metric * geometry_scale,
            chart_rank=base.loss.chart_rank * geometry_scale,
            cover=base.loss.cover * geometry_scale,
            balance=base.loss.balance * geometry_scale,
            sparsity=base.loss.sparsity * geometry_scale,
            geometry=base.loss.geometry * geometry_scale,
        ),
        train=replace(
            base.train,
            seed=model_seed,
            # These are validation-only budgets.  The final ten-seed stage
            # retains the requested 1,000-epoch budget.
            epochs=STAGE_EPOCHS[stage],
            eval_every=5,
            learning_rate=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
            device="cuda",
            neighbor_sampling=False,
            output_dir=f"outputs/selected_optuna/{stage}",
            resume=True,
        ),
    )


def _split_path(dataset: str, split_seed: int) -> Path:
    return OUTPUT / "splits" / dataset / f"split_{split_seed}.pt"


def _make_stratified_split(labels: torch.Tensor, split_seed: int) -> dict[str, torch.Tensor]:
    if labels.ndim != 1:
        raise ValueError("selected_hpo currently requires one single-label target per node for stratified splits")
    y = labels.detach().cpu().numpy()
    indices = np.arange(y.shape[0])
    train_val, test = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=split_seed).split(indices, y)
    )
    train_relative, val_relative = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=split_seed + 10_000).split(train_val, y[train_val])
    )
    return {
        "train": torch.from_numpy(train_val[train_relative]).long(),
        "val": torch.from_numpy(train_val[val_relative]).long(),
        "test": torch.from_numpy(test).long(),
    }


def _ensure_splits(dataset: str, data: GraphData) -> None:
    for split_seed in SEEDS:
        path = _split_path(dataset, split_seed)
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved.get("num_nodes") == data.num_nodes and saved.get("split_seed") == split_seed:
                continue
        split = _make_stratified_split(data.y, split_seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save({"dataset": dataset, "num_nodes": data.num_nodes, "split_seed": split_seed, **split}, path)


def _with_split(data: GraphData, dataset: str, split_seed: int) -> GraphData:
    split = torch.load(_split_path(dataset, split_seed), map_location="cpu", weights_only=True)
    masks = {}
    for name in ("train", "val", "test"):
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[split[name]] = True
        masks[f"{name}_mask"] = mask
    metadata = dict(data.metadata or {})
    metadata.update({"split": split_seed, "split_protocol": "stratified_60_20_20"})
    return replace(data, metadata=metadata, **masks)


def _load_split_data(dataset: str, split_seed: int) -> GraphData:
    config = _base_config(dataset, _default_params(), model_seed=0, split_seed=split_seed, stage="tune")
    data = load_dataset(config.dataset, seed=0)
    _ensure_splits(dataset, data)
    return _with_split(data, dataset, split_seed)


def _default_params() -> dict[str, Any]:
    return {
        "learning_rate": 0.003,
        "weight_decay": 0.0005,
        "dropout": 0.05,
        "hidden_dim": 64,
        "geometry_loss_scale": 1.0,
    }


def _sample_params(trial: Any) -> dict[str, Any]:
    return {
        "learning_rate": float(trial.suggest_float("learning_rate", 5e-4, 3e-3, log=True)),
        "weight_decay": float(trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)),
        "dropout": float(trial.suggest_float("dropout", 0.05, 0.35)),
        "hidden_dim": int(trial.suggest_categorical("hidden_dim", [64, 96, 128])),
        "geometry_loss_scale": float(trial.suggest_float("geometry_loss_scale", 0.5, 1.5)),
    }


def _run_hash(config: ExperimentConfig, split_seed: int, model_seed: int) -> str:
    return _json_hash({"config": config.to_dict(), "split_seed": split_seed, "model_seed": model_seed})


def _is_complete(run_dir: Path, stage: str, run_hash: str, require_test: bool) -> dict[str, Any] | None:
    path = run_dir / "metrics.json"
    required = ("config.yaml", "best.pt", "last.pt", "metrics.json")
    if not path.exists() or any(not (run_dir / name).exists() for name in required):
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selected_hpo_stage") != stage or payload.get("selected_hpo_hash") != run_hash:
        return None
    value = payload.get("test_metric") if require_test else payload.get("best_val_metric")
    return payload if value is not None and math.isfinite(float(value)) else None


def _run_one(
    *,
    stage: str,
    dataset: str,
    params: dict[str, Any],
    model_seed: int,
    split_seed: int,
    run_dir: Path,
    trial: Any | None = None,
    evaluate_test: bool,
) -> dict[str, Any]:
    config = _base_config(dataset, params, model_seed, split_seed, stage=stage)
    config.validate()
    run_hash = _run_hash(config, split_seed, model_seed)
    cached = _is_complete(run_dir, stage, run_hash, require_test=evaluate_test)
    if cached is not None:
        return cached
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config.to_dict(), run_dir / "selected_config.yaml")
    save_json({"stage": stage, "dataset": dataset, "params": params, "model_seed": model_seed,
               "split_seed": split_seed, "hash": run_hash}, run_dir / "selected_hpo_context.json")
    data = _load_split_data(dataset, split_seed)
    using_chunked_attention = data.edge_index.shape[1] > config.model.edge_chunk_threshold
    if trial is not None:
        cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(
            "[selected_hpo] "
            f"trial={trial.number} params={json.dumps(params, sort_keys=True)} "
            f"device={config.train.device} ({cuda_name}) nodes={data.num_nodes} edges={data.edge_index.shape[1]} "
            f"chunked_attention={using_chunked_attention}",
            flush=True,
        )

    def on_evaluation(epoch: int, value: float) -> None:
        if trial is None:
            return
        trial.report(value if math.isfinite(value) else -1e30, epoch)
        if not math.isfinite(value):
            import optuna

            raise optuna.TrialPruned(f"non-finite validation metric at epoch {epoch}")
        completed_trials = sum(item.state.name == "COMPLETE" for item in trial.study.trials)
        if (
            completed_trials >= PROMOTE_CANDIDATES
            and epoch > PRUNING_START_EPOCH
            and trial.should_prune()
        ):
            import optuna

            raise optuna.TrialPruned(f"pruned at epoch {epoch}")

    _, metrics = Trainer(config).fit(
        data,
        run_dir,
        on_evaluation=on_evaluation if trial is not None else None,
        evaluate_test=evaluate_test,
        include_intervention=False,
    )
    metrics.update({"selected_hpo_stage": stage, "selected_hpo_hash": run_hash,
                    "split_seed": split_seed, "model_seed": model_seed, "selected_params": params})
    save_json(metrics, run_dir / "metrics.json")
    return metrics


def _study_version() -> str:
    return _json_hash(TUNE_PROTOCOL)


def _study(dataset: str):
    import optuna

    version = _study_version()
    path = OUTPUT / "studies" / f"{dataset}_{version}.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=f"selected_hpo_{dataset}_{version}",
        storage=f"sqlite:///{path.resolve().as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2026, multivariate=True, group=True),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=PRUNING_START_EPOCH,
            max_resource=STAGE_EPOCHS["tune"],
            reduction_factor=3,
        ),
    )


def tune_dataset(dataset: str, n_trials: int) -> None:
    import optuna

    _ensure_splits(dataset, _load_split_data(dataset, 0))
    study = _study(dataset)

    def objective(trial: Any) -> float:
        params = _sample_params(trial)
        run_dir = OUTPUT / "tuning" / dataset / f"trial_{trial.number:03d}"
        try:
            metrics = _run_one(stage="tune", dataset=dataset, params=params, model_seed=TUNE_SEED,
                               split_seed=0, run_dir=run_dir, trial=trial, evaluate_test=False)
            value = float(metrics["best_val_metric"])
            if not math.isfinite(value):
                raise optuna.TrialPruned("non-finite validation metric")
            trial.set_user_attr("resolved_params", params)
            trial.set_user_attr("run_dir", str(run_dir.relative_to(ROOT)))
            return value
        except optuna.TrialPruned:
            raise
        except Exception as error:  # keep one failed configuration from stopping the study
            _record_error("tune", dataset, {"trial": trial.number, "params": params}, error)
            raise

    settled = sum(trial.state.name in {"COMPLETE", "PRUNED"} for trial in study.trials)
    remaining = max(0, n_trials - settled)
    if remaining:
        study.optimize(objective, n_trials=remaining, catch=(Exception,))
    rows = [
        {"number": trial.number, "state": trial.state.name, "value": trial.value,
         "params": json.dumps(trial.user_attrs.get("resolved_params", {}), sort_keys=True)}
        for trial in study.trials
    ]
    path = OUTPUT / "tuning" / dataset
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "trials.csv", index=False)


def _top_trials(dataset: str) -> list[Any]:
    import optuna

    completed = [
        trial for trial in _study(dataset).trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None and math.isfinite(float(trial.value))
        and trial.user_attrs.get("resolved_params")
    ]
    completed.sort(key=lambda trial: float(trial.value), reverse=True)
    if len(completed) < PROMOTE_CANDIDATES:
        raise RuntimeError(
            f"{dataset} needs {PROMOTE_CANDIDATES} finite completed Optuna trials before promotion"
        )
    return completed[:PROMOTE_CANDIDATES]


def promote_dataset(dataset: str) -> None:
    _ensure_splits(dataset, _load_split_data(dataset, 0))
    candidates: list[dict[str, Any]] = []
    for rank, trial in enumerate(_top_trials(dataset), start=1):
        params = dict(trial.user_attrs["resolved_params"])
        values: list[float] = []
        for seed in PROMOTE_SEEDS:
            run_dir = OUTPUT / "promote" / dataset / f"rank_{rank}" / f"seed_{seed}"
            try:
                metrics = _run_one(stage="promote", dataset=dataset, params=params, model_seed=seed,
                                   split_seed=seed, run_dir=run_dir, evaluate_test=False)
                values.append(float(metrics["best_val_metric"]))
            except Exception as error:
                _record_error("promote", dataset, {"rank": rank, "seed": seed}, error)
        mean = float(np.mean(values)) if values else float("nan")
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        candidates.append({"rank": rank, "trial": trial.number, "params": params,
                           "tune_best_val": float(trial.value), "values": values,
                           "mean": mean, "std": std,
                           "score": mean - 0.5 * std if math.isfinite(mean) else float("nan")})
    viable = [candidate for candidate in candidates if math.isfinite(candidate["score"])]
    if not viable:
        raise RuntimeError(f"No completed promotion configuration for {dataset}")
    selected = max(
        viable,
        key=lambda candidate: (candidate["score"], candidate["mean"], -candidate["std"], -candidate["rank"]),
    )
    output = {"dataset": dataset, "metric": "roc_auc" if dataset in ROC_AUC_DATASETS else "accuracy",
              "study_version": _study_version(),
              "candidates": candidates, "selected": selected}
    save_json(output, OUTPUT / "promote" / dataset / "summary.json")
    save_json(output, OUTPUT / "best_configs" / f"{dataset}.json")


def _update_summary(row: dict[str, Any]) -> None:
    path = OUTPUT / "summary.csv"
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if "dataset" in frame:
        frame = frame[frame["dataset"] != row["dataset"]]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame.sort_values("dataset").to_csv(path, index=False)


def _selected_params(dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    best_path = OUTPUT / "best_configs" / f"{dataset}.json"
    if not best_path.exists():
        raise RuntimeError(f"Promotion selection missing: {best_path}")
    selection = json.loads(best_path.read_text(encoding="utf-8"))
    if selection.get("study_version") != _study_version():
        raise RuntimeError(f"Promotion selection uses an older HPO protocol: {best_path}")
    selected = dict(selection["selected"])
    return dict(selected["params"]), selected


def final_dataset(dataset: str) -> None:
    params, selected = _selected_params(dataset)
    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = OUTPUT / "final_runs" / dataset / f"seed_{seed}"
        try:
            metrics = _run_one(stage="final", dataset=dataset, params=params, model_seed=seed,
                               split_seed=seed, run_dir=run_dir, evaluate_test=True)
            runs.append(metrics)
        except Exception as error:
            _record_error("final", dataset, {"seed": seed}, error)
    if not runs:
        raise RuntimeError(f"No completed final seeds for {dataset}")
    test = np.asarray([float(run["test_metric"]) for run in runs], dtype=float)
    val = np.asarray([float(run["best_val_metric"]) for run in runs], dtype=float)
    finite = np.isfinite(test)
    if not finite.any():
        raise RuntimeError(f"No finite final test metric for {dataset}")
    best_validation_index = int(np.nanargmax(val))
    best_test_index = int(np.nanargmax(test))
    row = {
        "dataset": dataset,
        "metric": "roc_auc" if dataset in ROC_AUC_DATASETS else "accuracy",
        "best_params": json.dumps(params, sort_keys=True),
        "tune_best_val": float(selected["tune_best_val"]),
        "promote_val_mean": float(selected["mean"]),
        "promote_val_std": float(selected["std"]),
        "test_mean": float(np.nanmean(test)),
        "test_std": float(np.nanstd(test, ddof=1)) if finite.sum() > 1 else 0.0,
        "test_min": float(np.nanmin(test)),
        "test_max": float(np.nanmax(test)),
        "best_seed": int(runs[best_validation_index]["model_seed"]),
        "best_seed_test": float(test[best_validation_index]),
        "oracle_test_seed": int(runs[best_test_index]["model_seed"]),
        "oracle_test_max": float(test[best_test_index]),
        "oracle_test_is_supplemental": True,
        "completed_seeds": json.dumps(sorted(int(run["model_seed"]) for run in runs)),
    }
    _update_summary(row)


def _best_available(dataset: str) -> tuple[dict[str, Any], int, int, Path, dict[str, Any]]:
    """Resolve the strongest validation-selected config/seed already on disk."""
    promoted = OUTPUT / "best_configs" / f"{dataset}.json"
    if promoted.exists():
        selection = json.loads(promoted.read_text(encoding="utf-8"))
        selected = dict(selection["selected"])
        values = [float(value) for value in selected.get("values", [])]
        seed = PROMOTE_SEEDS[int(np.nanargmax(values))] if values else 0
        source = OUTPUT / "promote" / dataset / f"rank_{int(selected['rank'])}" / f"seed_{seed}"
        return dict(selected["params"]), seed, seed, source, {
            "selection_source": "promote_mean_then_best_validation_seed",
            "validation_score": values[PROMOTE_SEEDS.index(seed)] if values else float("nan"),
            "trial": int(selected["trial"]),
        }
    trial = _top_trials(dataset)[0]
    params = dict(trial.user_attrs["resolved_params"])
    source = OUTPUT / "tuning" / dataset / f"trial_{trial.number:03d}"
    return params, TUNE_SEED, 0, source, {
        "selection_source": "best_completed_tune_trial",
        "validation_score": float(trial.value),
        "trial": int(trial.number),
    }


def best_single_dataset(dataset: str) -> None:
    """Continue the best existing validation run and evaluate its test split once."""
    params, model_seed, split_seed, source, selection = _best_available(dataset)
    _ensure_splits(dataset, _load_split_data(dataset, split_seed))
    run_dir = OUTPUT / "best_single" / dataset / f"seed_{model_seed}"
    if not (run_dir / "last.pt").exists():
        _prepare_seed_promotion(source, run_dir)
    metrics = _run_one(
        stage="best_single",
        dataset=dataset,
        params=params,
        model_seed=model_seed,
        split_seed=split_seed,
        run_dir=run_dir,
        evaluate_test=True,
    )
    payload = {
        "dataset": dataset,
        "metric": "roc_auc" if dataset in ROC_AUC_DATASETS else "accuracy",
        "model": "graphatlas",
        "params": params,
        "model_seed": model_seed,
        "split_seed": split_seed,
        "best_val_metric": float(metrics["best_val_metric"]),
        "test_metric": float(metrics["test_metric"]),
        "best_epoch": int(metrics["best_epoch"]),
        "runtime_seconds": float(metrics["runtime_seconds"]),
        **selection,
    }
    save_json(payload, OUTPUT / "best_single" / dataset / "summary.json")
    summary_path = OUTPUT / "best_single" / "summary.csv"
    frame = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if "dataset" in frame:
        frame = frame[frame["dataset"] != dataset]
    pd.concat([frame, pd.DataFrame([payload])], ignore_index=True).sort_values("dataset").to_csv(summary_path, index=False)


def _seed_result_row(metrics: dict[str, Any], metric: str) -> dict[str, Any]:
    return {
        "seed": int(metrics["model_seed"]),
        "split_seed": int(metrics["split_seed"]),
        "metric": metric,
        "best_val_metric": float(metrics["best_val_metric"]),
        "test_metric_at_best_val_checkpoint": float(metrics["test_metric"]),
        "best_epoch": int(metrics["best_epoch"]),
        "training_time": float(metrics["runtime_seconds"]),
        "membership_active_charts": float(metrics.get("mean_active_charts", float("nan"))),
        "routing_kl": float(metrics.get("routing_kl_from_membership", float("nan"))),
        "q_mean": float(metrics.get("transportability_mean", float("nan"))),
        "q_std": float(metrics.get("transportability_std", float("nan"))),
    }


def seed_scan_dataset(dataset: str, scan_seeds: int) -> None:
    params, _ = _selected_params(dataset)
    metric = "roc_auc" if dataset in ROC_AUC_DATASETS else "accuracy"
    rows: list[dict[str, Any]] = []
    # Fix split 0 so this experiment isolates initialization/dropout randomness.
    for model_seed in range(scan_seeds):
        run_dir = OUTPUT / "seed_ceiling" / "scan" / dataset / f"seed_{model_seed}"
        try:
            metrics = _run_one(
                stage="seed_scan",
                dataset=dataset,
                params=params,
                model_seed=model_seed,
                split_seed=0,
                run_dir=run_dir,
                evaluate_test=True,
            )
            rows.append(_seed_result_row(metrics, metric))
        except Exception as error:
            _record_error("seed_scan", dataset, {"seed": model_seed, "split_seed": 0}, error)
    if not rows:
        raise RuntimeError(f"No completed seed scan runs for {dataset}")
    rows.sort(key=lambda row: (-row["best_val_metric"], row["seed"]))
    output_dir = OUTPUT / "seed_ceiling" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "scan_results.csv", index=False)
    save_json(
        {
            "dataset": dataset,
            "study_version": _study_version(),
            "selection_policy": "validation_only",
            "fixed_split_seed": 0,
            "requested_scan_seeds": scan_seeds,
            "completed_scan_seeds": len(rows),
            "runs": rows,
        },
        output_dir / "scan_results.json",
    )


def _prepare_seed_promotion(scan_dir: Path, promote_dir: Path) -> None:
    promote_dir.mkdir(parents=True, exist_ok=True)
    destination = promote_dir / "last.pt"
    if destination.exists():
        return
    source = scan_dir / "last.pt"
    if not source.exists():
        raise RuntimeError(f"Seed scan checkpoint missing: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    # Promotion receives a fresh early-stopping window while preserving model,
    # optimizer, RNG, history, and the completed scan epoch.
    checkpoint["stale"] = 0
    _atomic_torch_save(checkpoint, destination)
    if (scan_dir / "best.pt").exists():
        shutil.copy2(scan_dir / "best.pt", promote_dir / "best.pt")


def seed_promote_dataset(dataset: str, promote_count: int) -> None:
    params, _ = _selected_params(dataset)
    scan_path = OUTPUT / "seed_ceiling" / dataset / "scan_results.json"
    if not scan_path.exists():
        raise RuntimeError(f"Seed scan results missing: {scan_path}")
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    if scan.get("study_version") != _study_version():
        raise RuntimeError(f"Seed scan uses an older HPO protocol: {scan_path}")
    candidates = sorted(
        scan["runs"],
        key=lambda row: (-float(row["best_val_metric"]), int(row["seed"])),
    )[:promote_count]
    metric = "roc_auc" if dataset in ROC_AUC_DATASETS else "accuracy"
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        model_seed = int(candidate["seed"])
        scan_dir = OUTPUT / "seed_ceiling" / "scan" / dataset / f"seed_{model_seed}"
        run_dir = OUTPUT / "seed_ceiling" / "promoted" / dataset / f"seed_{model_seed}"
        try:
            _prepare_seed_promotion(scan_dir, run_dir)
            metrics = _run_one(
                stage="seed_promote",
                dataset=dataset,
                params=params,
                model_seed=model_seed,
                split_seed=0,
                run_dir=run_dir,
                evaluate_test=True,
            )
            rows.append(_seed_result_row(metrics, metric))
        except Exception as error:
            _record_error("seed_promote", dataset, {"seed": model_seed, "split_seed": 0}, error)
    if not rows:
        raise RuntimeError(f"No completed promoted seeds for {dataset}")
    by_validation = max(rows, key=lambda row: (row["best_val_metric"], -row["seed"]))
    oracle = max(rows, key=lambda row: (row["test_metric_at_best_val_checkpoint"], -row["seed"]))
    output_dir = OUTPUT / "seed_ceiling" / dataset
    pd.DataFrame(rows).sort_values("seed").to_csv(output_dir / "promoted_results.csv", index=False)
    save_json(
        {
            "dataset": dataset,
            "metric": metric,
            "study_version": _study_version(),
            "claim": f"best-of-{scan['completed_scan_seeds']} empirical seed result on fixed split 0",
            "selection_policy": "top seeds selected only by validation metric",
            "fixed_split_seed": 0,
            "promoted_seed_count": len(rows),
            "validation_selected_seed": int(by_validation["seed"]),
            "validation_selected_test_metric": float(by_validation["test_metric_at_best_val_checkpoint"]),
            "oracle_test_seed": int(oracle["seed"]),
            "oracle_test_max": float(oracle["test_metric_at_best_val_checkpoint"]),
            "oracle_policy": "Internal analysis only; never use as a paper main result.",
            "runs": rows,
        },
        output_dir / "seed_ceiling_summary.json",
    )


def _run_stage(
    stage: str,
    dataset: str,
    n_trials: int,
    scan_seeds: int,
    seed_promote_count: int,
) -> None:
    if stage == "tune":
        tune_dataset(dataset, n_trials)
    elif stage == "promote":
        promote_dataset(dataset)
    elif stage == "final":
        final_dataset(dataset)
    elif stage == "seed_scan":
        seed_scan_dataset(dataset, scan_seeds)
    elif stage == "seed_promote":
        seed_promote_dataset(dataset, seed_promote_count)
    elif stage == "seed_ceiling":
        seed_scan_dataset(dataset, scan_seeds)
        seed_promote_dataset(dataset, seed_promote_count)
    elif stage == "best_single":
        best_single_dataset(dataset)
    elif stage == "all":
        tune_dataset(dataset, n_trials)
        promote_dataset(dataset)
        final_dataset(dataset)
    else:
        raise ValueError(f"Unknown selected_hpo stage: {stage}")


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"cumsum_cuda_kernel does not have a deterministic implementation.*",
        category=UserWarning,
    )
    parser = argparse.ArgumentParser(description="Optuna HPO protocol for the selected public GraphAtlas datasets.")
    parser.add_argument(
        "--stage",
        choices=("tune", "promote", "final", "seed_scan", "seed_promote", "seed_ceiling", "best_single", "all"),
        required=True,
    )
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--scan-seeds", type=int, default=DEFAULT_SCAN_SEEDS)
    parser.add_argument("--seed-promote-count", type=int, default=DEFAULT_SEED_PROMOTE_COUNT)
    args = parser.parse_args()
    if args.n_trials < 1:
        raise SystemExit("--n-trials must be positive")
    if args.scan_seeds < 1:
        raise SystemExit("--scan-seeds must be positive")
    if args.seed_promote_count < 1 or args.seed_promote_count > args.scan_seeds:
        raise SystemExit("--seed-promote-count must be between 1 and --scan-seeds")
    _require_cuda()
    for dataset in ((args.dataset,) if args.dataset else DATASETS):
        try:
            _run_stage(
                args.stage,
                dataset,
                args.n_trials,
                args.scan_seeds,
                args.seed_promote_count,
            )
        except Exception as error:
            _record_error(args.stage, dataset, {}, error)
            print(f"{dataset}: failed: {error}", flush=True)


if __name__ == "__main__":
    main()
