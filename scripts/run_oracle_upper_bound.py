#!/usr/bin/env python
"""Fast, resumable test-selected Oracle upper-bound search.

This runner is intentionally an oracle analysis: Optuna maximizes the official
test metric. Its outputs must not be reported as validation-selected results.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import traceback
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphatlas.config import ExperimentConfig  # noqa: E402
from graphatlas.data import GraphData, HeteroGraphData  # noqa: E402
from graphatlas.datasets import load_dataset  # noqa: E402
from graphatlas.trainer import Trainer  # noqa: E402
from graphatlas.utils import save_json, save_yaml  # noqa: E402


OUTPUT = ROOT / "outputs" / "oracle_upper_bound"
STUDY_VERSION = "oracle_fast_v4_dataset_specific_splits"
SPLIT_PROTOCOL = "dataset_specific"
DATASETS = (
    "actor", "questions", "dblp", "coauthor_cs", "coauthor_physics",
    "hetgb_texas", "hetgb_actor", "hetgb_amazon", "hetgb_cornell",
    "hetgb_wisconsin", "roman_empire", "tolokers", "h2gb_pdns",
)
EXCLUDED = ("h2gb_ieee_cis", "h2gb_mag_year")
SMALL = {
    "actor", "hetgb_texas", "hetgb_actor", "hetgb_amazon", "hetgb_cornell",
    "hetgb_wisconsin", "roman_empire", "tolokers",
}
MEDIUM = {"questions", "dblp", "coauthor_cs", "coauthor_physics"}
METRICS = {"questions": "roc_auc", "tolokers": "roc_auc", "h2gb_pdns": "f1"}
_DATA_CACHE: dict[tuple[str, int, str], GraphData | HeteroGraphData] = {}
_RAW_DATA_CACHE: dict[tuple[str, int], GraphData | HeteroGraphData] = {}


def budget(dataset: str) -> dict[str, int]:
    if dataset in SMALL:
        return {"trials": 40, "epochs": 120, "patience": 15, "eval_every": 2}
    if dataset in MEDIUM:
        return {"trials": 20, "epochs": 80, "patience": 10, "eval_every": 3}
    if dataset == "h2gb_pdns":
        return {"trials": 8, "epochs": 40, "patience": 6, "eval_every": 5}
    raise KeyError(dataset)


def _dataset_artifact(dataset: str) -> Path:
    if dataset == "h2gb_pdns":
        return ROOT / "data" / "h2gb" / dataset / "processed" / "graphatlas_h2gb.pt"
    return ROOT / "data" / dataset / "raw" / f"{dataset}.npz"


def _normalize_tensor(value: torch.Tensor, mode: str) -> torch.Tensor:
    value = value.to(torch.float32)
    if mode == "l2":
        return F.normalize(value, p=2, dim=-1, eps=1e-12)
    if mode == "layernorm":
        return F.layer_norm(value, (value.shape[-1],))
    return value


def _split_masks(
    dataset: str,
    labels: torch.Tensor,
    seed: int,
    *,
    train_fraction: float,
    val_fraction: float,
    stratified: bool,
    protocol: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if labels.ndim != 1:
        raise ValueError(f"{dataset} requires one-dimensional labels for split generation")
    test_fraction = 1.0 - train_fraction - val_fraction
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError(f"Invalid split fractions for {dataset}: {train_fraction}, {val_fraction}, {test_fraction}")
    split_path = OUTPUT / "splits" / protocol / dataset / f"split_{seed}.pt"
    if split_path.exists():
        saved = torch.load(split_path, map_location="cpu", weights_only=True)
        return saved["train"], saved["val"], saved["test"]
    values = labels.detach().cpu().numpy()
    indices = np.arange(values.shape[0])
    if stratified:
        train_val, test = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(indices, values)
        )
        train_relative, val_relative = next(
            StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_fraction / (train_fraction + val_fraction),
                random_state=seed + 10_000,
            ).split(train_val, values[train_val])
        )
        selected = (train_val[train_relative], train_val[val_relative], test)
    else:
        permutation = np.random.default_rng(seed).permutation(indices)
        train_count = int(round(values.shape[0] * train_fraction))
        val_count = int(round(values.shape[0] * val_fraction))
        selected = (
            permutation[:train_count],
            permutation[train_count:train_count + val_count],
            permutation[train_count + val_count:],
        )
    masks = []
    for chosen in selected:
        mask = torch.zeros(labels.shape[0], dtype=torch.bool)
        mask[torch.from_numpy(chosen).long()] = True
        masks.append(mask)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"protocol": protocol, "seed": seed, "train": masks[0], "val": masks[1], "test": masks[2]}, split_path)
    return masks[0], masks[1], masks[2]


def _dataset_split_protocol(dataset: str) -> str:
    return {
        "coauthor_cs": "random_60_20_20",
        "coauthor_physics": "random_60_20_20",
        "questions": "official_split_0",
        "tolokers": "official_split_0",
        "hetgb_amazon": "stratified_50_25_25",
        "hetgb_cornell": "stratified_48_32_20",
    }.get(dataset, "stratified_60_20_20")


def _masks_for_dataset(dataset: str, probe: GraphData | HeteroGraphData, seed: int):
    protocol = _dataset_split_protocol(dataset)
    if dataset in {"questions", "tolokers"}:
        return probe.train_mask, probe.val_mask, probe.test_mask, protocol
    if dataset in {"coauthor_cs", "coauthor_physics"}:
        return (*_split_masks(
            dataset, probe.y, seed, train_fraction=0.60, val_fraction=0.20,
            stratified=False, protocol=protocol,
        ), protocol)
    if dataset == "hetgb_amazon":
        return (*_split_masks(
            dataset, probe.y, seed, train_fraction=0.50, val_fraction=0.25,
            stratified=True, protocol=protocol,
        ), protocol)
    if dataset == "hetgb_cornell":
        return (*_split_masks(
            dataset, probe.y, seed, train_fraction=0.48, val_fraction=0.32,
            stratified=True, protocol=protocol,
        ), protocol)
    return (*_split_masks(
        dataset, probe.y, seed, train_fraction=0.60, val_fraction=0.20,
        stratified=True, protocol=protocol,
    ), protocol)


def _load_cached(dataset: str, seed: int, normalization: str) -> GraphData | HeteroGraphData:
    cache_seed = int(seed)
    cached = _DATA_CACHE.get((dataset, cache_seed, normalization))
    if cached is not None:
        return cached

    base = ExperimentConfig.from_yaml(ROOT / "configs" / "base.yaml")
    dataset_config = replace(
        base.dataset,
        name=dataset,
        root="data",
        split=0 if dataset in {"questions", "tolokers"} else int(seed),
        num_charts=4,
        metric=METRICS.get(dataset),
        feature_normalization=normalization,
        heterogeneous=dataset == "h2gb_pdns",
        target_node_type="domain_node" if dataset == "h2gb_pdns" else None,
    )
    probe = _RAW_DATA_CACHE.get((dataset, 0))
    if probe is None:
        probe = load_dataset(dataset_config, seed=0)
        _RAW_DATA_CACHE[(dataset, 0)] = probe
    if isinstance(probe, (GraphData, HeteroGraphData)):
        train_mask, val_mask, test_mask, split_protocol = _masks_for_dataset(dataset, probe, int(seed))
        probe = replace(
            probe,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            metadata={**(probe.metadata or {}), "split": 0 if dataset in {"questions", "tolokers"} else int(seed), "split_protocol": split_protocol},
        )
    key = (dataset, cache_seed, normalization)
    cached = _DATA_CACHE.get(key)
    if cached is not None:
        return cached
    if isinstance(probe, HeteroGraphData):
        probe = replace(
            probe,
            x_dict={
                node_type: None if value is None else _normalize_tensor(value, normalization)
                for node_type, value in probe.x_dict.items()
            },
            metadata={**probe.metadata, "feature_normalization": normalization},
        )
    else:
        probe = replace(
            probe,
            x=_normalize_tensor(probe.x, normalization),
            metadata={**(probe.metadata or {}), "feature_normalization": normalization},
            cached_signature=None,
            cached_signature_key=None,
        )
    probe.validate()
    _DATA_CACHE[key] = probe
    return probe


def _sample_params(trial: Any, dataset: str) -> dict[str, Any]:
    num_charts = int(trial.suggest_categorical("num_charts", [1, 2, 3, 4]))
    # Optuna requires one immutable distribution per parameter name. Sample
    # from the fixed [1, 2] space, then resolve the structural constraint.
    sampled_topk = int(trial.suggest_categorical("membership_topk", [1, 2]))
    return {
        "learning_rate": float(trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)),
        "weight_decay": float(trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)),
        "dropout": float(trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.4, 0.6])),
        "hidden_dim": int(trial.suggest_categorical(
            "hidden_dim", [64, 128] if dataset == "h2gb_pdns" else [64, 128, 256]
        )),
        "num_charts": num_charts,
        "membership_topk": min(sampled_topk, num_charts),
        "transportability_beta": float(trial.suggest_categorical(
            "transportability_beta", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
        )),
        "classification_loss": str(trial.suggest_categorical(
            "classification_loss", ["cross_entropy", "weighted_ce", "focal"]
        )),
        "feature_normalization": str(trial.suggest_categorical(
            "feature_normalization", ["none", "l2", "layernorm"]
        )),
        "seed": int(trial.suggest_int("seed", 0, 999)),
    }


def _config(dataset: str, params: dict[str, Any], *, diagnostics: bool = False) -> ExperimentConfig:
    base = ExperimentConfig.from_yaml(ROOT / "configs" / "base.yaml")
    b = budget(dataset)
    heterogeneous = dataset == "h2gb_pdns"
    charts = int(params["num_charts"])
    return replace(
        base,
        dataset=replace(
            base.dataset,
            name=dataset,
            root="data",
            split=int(params["seed"]),
            num_charts=charts,
            metric=METRICS.get(dataset),
            feature_normalization=str(params["feature_normalization"]),
            heterogeneous=heterogeneous,
            target_node_type="domain_node" if heterogeneous else None,
        ),
        model=replace(
            base.model,
            name="atn" if heterogeneous else "graphatlas_certified",
            label="atn" if heterogeneous else "graphatlas_c",
            heterogeneous=heterogeneous,
            hidden_dim=int(params["hidden_dim"]),
            dropout=float(params["dropout"]),
            observation_dim=24,
            chart_dim=4,
            vector_channels=6,
            num_charts=charts,
            num_layers=1,
            membership_topk=min(int(params["membership_topk"]), charts),
            transport_mode="certified",
            transportability_beta=float(params["transportability_beta"]),
            edge_chunk_size=100000,
            edge_chunk_threshold=2000000,
        ),
        loss=replace(
            base.loss,
            classification_loss=str(params["classification_loss"]),
            geometry=0.0,
        ),
        train=replace(
            base.train,
            seed=int(params["seed"]),
            device="cuda",
            epochs=b["epochs"],
            patience=b["patience"],
            eval_every=b["eval_every"],
            learning_rate=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
            deterministic=False,
            final_diagnostics=diagnostics,
            resume=True,
            mode="hetero_neighbor" if heterogeneous else "full_batch",
            batch_size=256 if heterogeneous else base.train.batch_size,
            neighbor_sizes=[20, 15],
            target_batching=heterogeneous,
            output_dir="outputs/oracle_upper_bound/runs",
        ),
    )


def _run_trial(dataset: str, trial: Any, params: dict[str, Any]) -> dict[str, Any]:
    import optuna

    config = _config(dataset, params, diagnostics=False)
    config.validate()
    run_dir = OUTPUT / "runs" / dataset / f"trial_{trial.number:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config.to_dict(), run_dir / "oracle_config.yaml")
    data = _load_cached(dataset, int(params["seed"]), str(params["feature_normalization"]))

    def oracle_callback(epoch: int, val: float, test: float, val_classes: int, test_classes: int) -> None:
        trial.report(val if math.isfinite(val) else -1e30, epoch)
        trial.set_user_attr("last_intermediate_test", test)
        if 10 <= epoch <= 15 and val_classes <= 1 and test_classes <= 1:
            raise optuna.TrialPruned(f"single-class validation and test predictions at epoch {epoch}")
        if epoch >= 10 and trial.should_prune():
            raise optuna.TrialPruned(f"pruned at epoch {epoch}")

    _, metrics = Trainer(config).fit(
        data,
        run_dir,
        evaluate_test=True,
        include_intervention=False,
        on_oracle_evaluation=oracle_callback,
    )
    metrics.update({"oracle_params": params, "oracle_trial": trial.number, "oracle_selection": "test_metric"})
    save_json(metrics, run_dir / "metrics.json")
    return metrics


def _study(dataset: str, resume: bool):
    import optuna

    path = OUTPUT / "studies" / SPLIT_PROTOCOL / f"{dataset}.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not resume:
        raise FileExistsError(f"Study exists: {path}. Pass --resume to continue it.")
    b = budget(dataset)
    try:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=10,
            max_resource=b["epochs"],
            reduction_factor=3,
        )
    except (TypeError, ValueError):
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10)
    return optuna.create_study(
        study_name=f"{STUDY_VERSION}_{dataset}",
        storage=f"sqlite:///{path.resolve().as_posix()}",
        load_if_exists=resume,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2026, multivariate=True, group=True),
        pruner=pruner,
    )


def _trial_payload(trial: Any, policy: str) -> dict[str, Any]:
    return {
        "selection_policy": policy,
        "trial": int(trial.number),
        "objective_test_metric": float(trial.value),
        "validation_metric_at_validation_selected_checkpoint": float(trial.user_attrs["validation_metric"]),
        "test_metric_at_validation_selected_checkpoint": float(trial.user_attrs["test_metric"]),
        "metric_name": trial.user_attrs["metric_name"],
        "split_protocol": trial.user_attrs.get("split_protocol", SPLIT_PROTOCOL),
        "params": trial.user_attrs["resolved_params"],
        "run_dir": trial.user_attrs["run_dir"],
        "warning": "Oracle test-selected result; do not report as a validation-selected benchmark.",
    }


def _write_outputs(dataset: str, study: Any) -> Any | None:
    import optuna

    output = OUTPUT / dataset
    output.mkdir(parents=True, exist_ok=True)
    frame = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    frame.to_csv(output / "trials.csv", index=False)
    complete = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None and math.isfinite(float(trial.value))
        and "validation_metric" in trial.user_attrs
    ]
    if not complete:
        return None
    best_test = max(complete, key=lambda item: float(item.value))
    best_validation = max(complete, key=lambda item: float(item.user_attrs["validation_metric"]))
    save_json(_trial_payload(best_test, "oracle_test_selected"), output / "best_test_selected.json")
    save_json(_trial_payload(best_validation, "validation_selected"), output / "best_validation_selected.json")

    rows = []
    summary_path = OUTPUT / "oracle_summary.csv"
    if summary_path.exists():
        rows = pd.read_csv(summary_path).to_dict("records")
        rows = [row for row in rows if row.get("dataset") != dataset]
    rows.append({
        "dataset": dataset,
        "metric": best_test.user_attrs["metric_name"],
        "oracle_test_metric": float(best_test.value),
        "oracle_trial": int(best_test.number),
        "oracle_seed": int(best_test.user_attrs["resolved_params"]["seed"]),
        "split_protocol": _dataset_split_protocol(dataset),
        "validation_selected_val_metric": float(best_validation.user_attrs["validation_metric"]),
        "validation_selected_test_metric": float(best_validation.user_attrs["test_metric"]),
        "validation_selected_trial": int(best_validation.number),
        "completed_trials": len(complete),
    })
    pd.DataFrame(rows).sort_values("dataset").to_csv(summary_path, index=False)
    return best_test


def _compute_best_diagnostics(dataset: str, trial: Any) -> None:
    """Reuse the winning checkpoint and compute expensive geometry diagnostics once."""
    output = OUTPUT / dataset / "best_trial_diagnostics.json"
    if output.exists() or dataset == "h2gb_pdns":
        return
    params = dict(trial.user_attrs["resolved_params"])
    source = OUTPUT / "runs" / dataset / f"trial_{trial.number:04d}"
    diagnostic_run = OUTPUT / "runs" / dataset / f"trial_{trial.number:04d}_diagnostics"
    diagnostic_run.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "last.pt", "history.csv"):
        if (source / name).exists() and not (diagnostic_run / name).exists():
            shutil.copy2(source / name, diagnostic_run / name)
    config = _config(dataset, params, diagnostics=True)
    last_path = diagnostic_run / "last.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        config = replace(config, train=replace(config.train, epochs=max(1, int(checkpoint["epoch"]))))
    data = _load_cached(dataset, int(params["seed"]), str(params["feature_normalization"]))
    _, metrics = Trainer(config).fit(
        data, diagnostic_run, evaluate_test=True, include_intervention=False
    )
    save_json(metrics, output)


def run_dataset(dataset: str, resume: bool) -> None:
    import optuna

    study = _study(dataset, resume)
    target = budget(dataset)["trials"]
    settled = sum(trial.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
                  for trial in study.trials)

    def objective(trial: Any) -> float:
        params = _sample_params(trial, dataset)
        try:
            metrics = _run_trial(dataset, trial, params)
        except optuna.TrialPruned:
            raise
        except Exception as error:
            errors = OUTPUT / dataset / "errors.jsonl"
            errors.parent.mkdir(parents=True, exist_ok=True)
            with errors.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "trial": trial.number, "params": params, "error": repr(error),
                    "traceback": traceback.format_exc(),
                }, ensure_ascii=False) + "\n")
            raise
        test = float(metrics["test_metric"])
        val = float(metrics["best_val_metric"])
        if not math.isfinite(test):
            raise optuna.TrialPruned("non-finite official test metric")
        trial.set_user_attr("resolved_params", params)
        trial.set_user_attr("validation_metric", val)
        trial.set_user_attr("test_metric", test)
        trial.set_user_attr("metric_name", str(metrics["metric_name"]))
        trial.set_user_attr("dataset", dataset)
        trial.set_user_attr("split_protocol", _dataset_split_protocol(dataset))
        trial.set_user_attr("run_dir", str((OUTPUT / "runs" / dataset / f"trial_{trial.number:04d}").relative_to(ROOT)))
        return test

    remaining = max(0, target - settled)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, catch=(Exception,), gc_after_trial=True)
    best_test = _write_outputs(dataset, study)
    if best_test is not None:
        _compute_best_diagnostics(dataset, best_test)


def dry_run(datasets: list[str]) -> None:
    rows = []
    for dataset in datasets:
        artifact = _dataset_artifact(dataset)
        b = budget(dataset)
        rows.append({
            "dataset": dataset,
            **b,
            "metric": METRICS.get(dataset, "accuracy"),
            "artifact": str(artifact.relative_to(ROOT)),
            "available": artifact.exists(),
            "study": str((OUTPUT / "studies" / SPLIT_PROTOCOL / f"{dataset}.db").relative_to(ROOT)),
        })
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    print("Split protocols: dataset-specific; Questions/Tolokers use official split 0")
    print(f"Excluded: {', '.join(EXCLUDED)}")
    if not frame["available"].all():
        missing = frame.loc[~frame["available"], "dataset"].tolist()
        print(f"WARNING: missing local datasets: {', '.join(missing)}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile", choices=["fast"], default="fast")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    datasets = list(DATASETS) if args.all else [args.dataset]
    if args.dry_run:
        dry_run(datasets)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Oracle upper-bound search requires one CUDA GPU")
    # Oracle trials intentionally trade bitwise determinism for throughput.
    torch.use_deterministic_algorithms(False)
    torch.set_deterministic_debug_mode("default")
    warnings.filterwarnings(
        "ignore",
        message=r"cumsum_cuda_kernel does not have a deterministic implementation.*",
    )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for dataset in datasets:
        run_dataset(dataset, args.resume)


if __name__ == "__main__":
    main()
