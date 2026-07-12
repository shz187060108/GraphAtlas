from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
import traceback
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm
from threadpoolctl import threadpool_limits

from graphatlas.config import ExperimentConfig
from graphatlas.datasets.names import is_atlas_het_name
from graphatlas.utils import deep_update, load_yaml, save_json, save_yaml, stable_hash




def _source_fingerprint(project_root: Path) -> str:
    """Hash result-affecting source files so stale checkpoints are never reused."""
    digest = hashlib.sha256()
    roots = [project_root / "src" / "graphatlas"]
    extra_files = [project_root / "pyproject.toml", project_root / "requirements.txt"]
    files: list[Path] = []
    for root in roots:
        files.extend(path for path in root.rglob("*.py") if path.is_file())
    files.extend(path for path in extra_files if path.exists())
    for path in sorted(files, key=lambda item: str(item.relative_to(project_root))):
        relative = str(path.relative_to(project_root)).encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()[:16]

def _safe_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def _validate_run_artifacts(run_dir: Path) -> None:
    required = ["config.yaml", "environment.json", "best.pt", "metrics.json", "history.csv", "predictions.pt"]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Run {run_dir} is missing artifacts: {missing}")


def _job_config(
    base: dict[str, Any],
    preset: dict[str, Any],
    dataset_override: dict[str, Any],
    model_override: dict[str, Any],
    seed: int,
) -> ExperimentConfig:
    config_dict = deep_update(base, {"train": preset.get("train", {}), "loss": preset.get("loss", {})})
    config_dict = deep_update(
        config_dict,
        {"dataset": dataset_override, "model": model_override, "train": {"seed": seed}},
    )
    config_dict["model"]["num_charts"] = config_dict["dataset"].get(
        "num_charts", config_dict["model"]["num_charts"]
    )
    dataset_name = config_dict["dataset"]["name"].lower().replace("-", "_")
    # The benchmark supplies ten fixed splits. The paper preset pairs split s
    # with initialization seed s, matching the standard ten-run protocol.
    if not is_atlas_het_name(dataset_name) and "split" not in dataset_override:
        config_dict["dataset"]["split"] = seed
    model_name = config_dict["model"]["name"]
    if model_name == "graphatlas_no_metric":
        config_dict["loss"]["metric"] = 0.0
    if model_name == "graphatlas_no_cocycle":
        config_dict["loss"]["cocycle"] = 0.0
        config_dict["loss"]["inverse_cycle"] = 0.0
        config_dict["loss"]["path_consistency"] = 0.0
    if model_name == "graphatlas_no_rank":
        config_dict["loss"]["chart_rank"] = 0.0
    if not is_atlas_het_name(dataset_name):
        config_dict["loss"]["geometry"] = 0.0
    return ExperimentConfig.from_dict(config_dict)


def _run_identity(config: ExperimentConfig, source_hash: str) -> tuple[str, str]:
    config_hash = stable_hash({"config": config.fingerprint_dict(), "source_hash": source_hash})
    run_id = (
        f"{_safe_name(config.dataset.name)}__{_safe_name(config.dataset.task)}"
        f"__{_safe_name(config.model.name)}__split{config.dataset.split}"
        f"__seed{config.train.seed}__{config_hash}"
    )
    return run_id, config_hash


def _run_directory(project_root: Path, config: ExperimentConfig, config_hash: str) -> Path:
    return (
        project_root
        / config.train.output_dir
        / _safe_name(config.dataset.name)
        / _safe_name(config.dataset.task)
        / _safe_name(config.model.name)
        / f"split_{config.dataset.split}"
        / f"seed_{config.train.seed}"
        / f"cfg_{config_hash}"
    )



def _merge_master_results(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()
    combined = pd.concat([existing, frame], ignore_index=True, sort=False)
    if "run_id" in combined:
        combined = combined.drop_duplicates(subset=["run_id"], keep="last")
    sort_columns = [column for column in ["dataset", "task", "model", "split", "seed"] if column in combined]
    if sort_columns:
        combined = combined.sort_values(sort_columns).reset_index(drop=True)
    combined.to_csv(path, index=False)

def _run_job_subprocess(
    project_root: Path,
    config: ExperimentConfig,
    run_dir: Path,
    run_id: str,
    config_hash: str,
    source_hash: str,
) -> dict[str, Any]:
    """Execute one experiment in a fresh process.

    Process isolation releases PyTorch and native allocator state after every
    run. This prevents long matrices from accumulating CPU or CUDA memory.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    job_config = run_dir / "job_config.yaml"
    save_yaml(config.to_dict(), job_config)
    command = [
        sys.executable,
        "-u",
        str(project_root / "scripts" / "train.py"),
        "--config",
        str(job_config),
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--config-hash",
        config_hash,
        "--source-hash",
        source_hash,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    thread_count = str(config.train.num_threads)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        environment[variable] = thread_count
    completed = subprocess.run(command, cwd=project_root, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Worker process failed with exit code {completed.returncode} for {run_id}. "
            f"Inspect {run_dir} and the terminal output."
        )
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Worker completed without metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = {"run_id": run_id, "config_hash": config_hash, "source_hash": source_hash}
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise RuntimeError(f"Worker metrics {key} mismatch in {metrics_path}")
    return metrics


def run_preset(
    preset_path: str | Path,
    force: bool = False,
    limit: int | None = None,
    continue_on_error: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    preset_path = Path(preset_path).resolve()
    preset = load_yaml(preset_path)
    project_root = preset_path.parents[2]
    source_hash = _source_fingerprint(project_root)
    base_path = project_root / preset["base_config"]
    base = load_yaml(base_path)
    active_datasets = [entry for entry in preset["datasets"] if entry.get("enabled", True)]
    active_models = [entry for entry in preset["models"] if entry.get("enabled", True)]
    jobs = list(itertools.product(active_datasets, active_models, preset["seeds"]))
    if limit is not None:
        jobs = jobs[:limit]
    outputs = project_root / "outputs"
    results_dir = outputs / "results"
    status_dir = outputs / "status"
    manifests_dir = outputs / "manifests"
    results_path = results_dir / "all.csv"
    preset_results_path = results_dir / f"{preset_path.stem}.csv"
    status_path = status_dir / f"experiment_{preset_path.stem}.json"
    failures_path = status_dir / f"failures_{preset_path.stem}.json"
    manifest_path = manifests_dir / f"{preset_path.stem}.csv"
    collected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start_time = time.time()

    manifest_rows: list[dict[str, Any]] = []
    prepared: list[tuple[dict[str, Any], dict[str, Any], int, ExperimentConfig, str, str, Path]] = []
    for dataset_override, model_override, seed in jobs:
        config = _job_config(base, preset, dataset_override, model_override, seed)
        config.validate()
        run_id, config_hash = _run_identity(config, source_hash)
        run_dir = _run_directory(project_root, config, config_hash)
        prepared.append((dataset_override, model_override, seed, config, run_id, config_hash, run_dir))
        manifest_rows.append(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "source_hash": source_hash,
                "dataset": config.dataset.name,
                "task": config.dataset.task,
                "model": config.model.label or config.model.name,
                "model_family": config.model.name,
                "seed": seed,
                "split": config.dataset.split,
                "run_dir": str(run_dir.relative_to(project_root)),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    if dry_run:
        return manifest

    progress = tqdm(
        prepared,
        desc="Complete experiment matrix",
        unit="run",
        disable=os.environ.get("GRAPHATLAS_PROGRESS", "1") == "0",
    )
    for job_index, (_, _, seed, config, run_id, config_hash, run_dir) in enumerate(progress, start=1):
        progress.set_postfix(dataset=config.dataset.name, model=config.model.name, seed=seed)
        metrics_path = run_dir / "metrics.json"
        current = {
            "index": job_index,
            "run_id": run_id,
            "dataset": config.dataset.name,
            "task": config.dataset.task,
            "model": config.model.name,
            "seed": seed,
            "split": config.dataset.split,
        }
        try:
            if force and run_dir.exists():
                shutil.rmtree(run_dir)
            if metrics_path.exists() and not force:
                try:
                    _validate_run_artifacts(run_dir)
                except RuntimeError:
                    # Older or interrupted runs may have metrics but lack a
                    # required reproducibility artifact. Finalize them again.
                    pass
                else:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if metrics.get("config_hash") != config_hash:
                        raise RuntimeError(f"Configuration hash mismatch in {metrics_path}")
                    collected.append(metrics)
                    continue
            if os.environ.get("GRAPHATLAS_IN_PROCESS", "0") == "1":
                from graphatlas.datasets import load_dataset
                from graphatlas.trainer import Trainer
                from graphatlas.utils import configure_torch_threads

                configure_torch_threads(config.train.num_threads)
                with threadpool_limits(limits=config.train.num_threads):
                    data = load_dataset(config.dataset, seed)
                    trainer = Trainer(config)
                    _, metrics = trainer.fit(data, run_dir)
                metrics.update({"run_id": run_id, "config_hash": config_hash, "source_hash": source_hash})
                save_json(metrics, metrics_path)
            else:
                metrics = _run_job_subprocess(
                    project_root, config, run_dir, run_id, config_hash, source_hash
                )
            _validate_run_artifacts(run_dir)
            collected.append(metrics)
        except Exception as error:  # noqa: BLE001 - batch runner records full context
            failure = {
                **current,
                "config_hash": config_hash,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            save_json({"failures": failures}, failures_path)
            if not continue_on_error:
                raise
        finally:
            frame = pd.DataFrame(collected)
            if not frame.empty:
                frame.to_csv(preset_results_path, index=False)
                _merge_master_results(results_path, frame)
            status_payload = {
                "preset": preset_path.stem,
                "expected_runs": len(prepared),
                "completed_runs": len(collected),
                "failed_runs": len(failures),
                "remaining_runs": len(prepared) - len(collected) - len(failures),
                "progress_fraction": (len(collected) + len(failures)) / max(len(prepared), 1),
                "elapsed_seconds": time.time() - start_time,
                "current": current,
                "complete": len(collected) + len(failures) == len(prepared) and not failures,
            }
            save_json(status_payload, status_path)

    frame = pd.DataFrame(collected)
    if not frame.empty:
        sort_columns = [column for column in ["dataset", "task", "model", "split", "seed"] if column in frame]
        frame = frame.sort_values(sort_columns).reset_index(drop=True)
        frame.to_csv(preset_results_path, index=False)
        _merge_master_results(results_path, frame)
    if failures:
        save_json({"failures": failures}, failures_path)
        if not continue_on_error:
            raise RuntimeError(f"{len(failures)} experiment runs failed; see {failures_path}")
    elif failures_path.exists():
        failures_path.unlink()
    return frame
