#!/usr/bin/env python
"""Remove disposable GraphAtlas artifacts and restore the concise output layout."""
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import shutil
from pathlib import Path

import pandas as pd

from graphatlas.experiments import concise_results_frame


ROOT = Path(PROJECT_ROOT)
OUTPUTS = ROOT / "outputs"


def _read(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def _clear_directory(path: Path) -> None:
    """Clear children while leaving an IDE-locked artifact in place for retry."""
    if not path.exists():
        return
    for child in path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except PermissionError:
            print(f"locked; retained for next cleanup: {child.relative_to(OUTPUTS)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove temporary GraphAtlas output artifacts.")
    parser.add_argument("--apply", action="store_true", help="Perform the cleanup; otherwise only print the plan.")
    args = parser.parse_args()

    keep_result = OUTPUTS / "results" / "selected_seed0.csv"
    keep_manifests = {
        "public_graphatlas_seed0.jsonl": _read(OUTPUTS / "manifests" / "selected_seed0.jsonl"),
        "public_graphatlas_seed0.csv": _read(OUTPUTS / "manifests" / "selected_seed0.csv"),
        "ogb_scalable.jsonl": _read(OUTPUTS / "manifests" / "ogb_scalable.jsonl"),
        "ogb_scalable.csv": _read(OUTPUTS / "manifests" / "ogb_scalable.csv"),
    }
    result_frame = pd.read_csv(keep_result) if keep_result.exists() else pd.DataFrame()
    remove = (".cache", ".mplconfig", "cache", "rapid_sota", "runs", "reports", "status", ".work")
    print("remove:", ", ".join(remove))
    print("keep: public_graphatlas_seed0.csv and the public/OGB manifests")
    if not args.apply:
        return

    for name in remove:
        path = OUTPUTS / name
        if path.exists():
            shutil.rmtree(path)
    for name in ("results", "manifests"):
        _clear_directory(OUTPUTS / name)

    results = OUTPUTS / "results"
    manifests = OUTPUTS / "manifests"
    reports = OUTPUTS / "reports"
    work = OUTPUTS / ".work"
    for path in (results, manifests, reports, work):
        path.mkdir(parents=True, exist_ok=True)

    if not result_frame.empty:
        concise = concise_results_frame(result_frame)
        concise.to_csv(results / "public_graphatlas_seed0.csv", index=False)
        concise.to_csv(results / "latest.csv", index=False)
    for name, payload in keep_manifests.items():
        if payload is not None:
            (manifests / name).write_bytes(payload)

    (OUTPUTS / "README.md").write_text(
        "# GraphAtlas outputs\n\n"
        "- `results/`: small tables intended for reading and comparison. `latest.csv` is the latest completed matrix.\n"
        "- `reports/`: figures and statistical summaries for a completed matrix.\n"
        "- `manifests/`: only the plans that are still relevant.\n"
        "- `.work/`: hidden resumable checkpoints, raw predictions, and live status; safe to delete with this script when no run is active.\n",
        encoding="utf-8",
    )
    (results / "README.md").write_text(
        "Each row is one dataset/model/seed result. Full diagnostic tensors and checkpoints are intentionally kept out of this directory.\n",
        encoding="utf-8",
    )
    print("cleanup complete")


if __name__ == "__main__":
    main()
