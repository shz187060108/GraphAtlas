#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _acquire_experiment_lock():
    lock_path = Path(os.environ.get("GRAPHATLAS_LOCK_PATH", "/tmp/graphatlas_experiments.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            handle.write("0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        print(f"Another GraphAtlas experiment matrix is already running (lock: {lock_path}).", file=sys.stderr)
        handle.close()
        raise SystemExit(75)
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _load_plan(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _valid_completed(root: Path, record: dict[str, object]) -> bool:
    run_dir = root / str(record["run_dir"])
    required = ["config.yaml", "environment.json", "best.pt", "metrics.json", "history.csv", "predictions.pt"]
    if any(not (run_dir / name).exists() for name in required):
        return False
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(metrics.get(key) == record[key] for key in ("run_id", "config_hash", "source_hash"))


def _write_status(root: Path, preset: str, total: int, completed: int, failed: list[dict[str, object]], current):
    payload = {
        "preset": preset,
        "expected_runs": total,
        "completed_runs": completed,
        "failed_runs": len(failed),
        "remaining_runs": total - completed - len(failed),
        "progress_fraction": (completed + len(failed)) / max(total, 1),
        "current": current,
        "complete": completed == total and not failed,
    }
    path = root / "outputs" / ".work" / "status" / f"experiment_{preset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    lock_handle = _acquire_experiment_lock()
    parser = argparse.ArgumentParser(description="Run a resumable GraphAtlas experiment matrix.")
    parser.add_argument("--preset", default="full", help="Preset name from configs/presets or a YAML path.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    preset_path = Path(args.preset)
    if not preset_path.suffix:
        preset_path = root / "configs" / "presets" / f"{args.preset}.yaml"
    elif not preset_path.is_absolute():
        preset_path = root / preset_path
    preset_name = preset_path.stem
    plan_path = root / "outputs" / "manifests" / f"{preset_name}.jsonl"
    materialize = [
        sys.executable, "-u", str(root / "scripts" / "materialize_jobs.py"),
        "--preset", str(preset_path), "--plan", str(plan_path),
    ]
    if args.limit is not None:
        materialize.extend(["--limit", str(args.limit)])
    if args.force:
        materialize.append("--force")
    subprocess.run(materialize, cwd=root, env=os.environ.copy(), check=True)
    plan = _load_plan(plan_path)
    if args.dry_run:
        print(f"Prepared {len(plan)} runs in {plan_path}")
        lock_handle.close()
        return

    failures: list[dict[str, object]] = []
    completed = sum(_valid_completed(root, record) for record in plan)
    _write_status(root, preset_name, len(plan), completed, failures, None)
    for index, record in enumerate(plan, start=1):
        if _valid_completed(root, record):
            continue
        current = {key: record[key] for key in ("run_id", "dataset", "task", "model", "seed", "split")}
        current["index"] = index
        _write_status(root, preset_name, len(plan), completed, failures, current)
        percent = 100.0 * (completed + len(failures)) / max(len(plan), 1)
        print(
            f"[{completed + len(failures):>3}/{len(plan):<3} {percent:6.2f}%] "
            f"{record['dataset']}/{record['model']}/seed={record['seed']}",
            flush=True,
        )
        run_dir = root / str(record["run_dir"])
        command = [
            sys.executable, "-u", str(root / "scripts" / "train.py"),
            "--config", str(root / str(record["job_config"])),
            "--run-dir", str(run_dir),
            "--run-id", str(record["run_id"]),
            "--config-hash", str(record["config_hash"]),
            "--source-hash", str(record["source_hash"]),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        thread_count = str(record.get("num_threads", 1))
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            environment[variable] = thread_count
        started = time.time()
        result = subprocess.run(command, cwd=root, env=environment, check=False)
        if result.returncode == 0 and _valid_completed(root, record):
            completed += 1
            print(f"      completed in {time.time() - started:.2f}s", flush=True)
        else:
            failure = {**current, "returncode": result.returncode}
            failures.append(failure)
            print(f"      FAILED with exit code {result.returncode}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                _write_status(root, preset_name, len(plan), completed, failures, current)
                raise SystemExit(result.returncode or 1)
        _write_status(root, preset_name, len(plan), completed, failures, current)

    collect = subprocess.run(
        [sys.executable, "-u", str(root / "scripts" / "collect_results.py"), "--plan", str(plan_path), "--preset-name", preset_name],
        cwd=root,
        env=os.environ.copy(),
        check=False,
    )
    lock_handle.close()
    if collect.returncode != 0:
        raise SystemExit(collect.returncode)
    print(f"[{len(plan):>3}/{len(plan):<3} 100.00%] complete", flush=True)
    if os.environ.get("GRAPHATLAS_DEFER_SYNC", "0") != "1":
        subprocess.run(
            [sys.executable, "-u", str(root / "scripts" / "sync_github.py"),
             "--message", f"Complete {preset_name} experiment"],
            cwd=root,
            env=os.environ.copy(),
            check=True,
        )


if __name__ == "__main__":
    main()
