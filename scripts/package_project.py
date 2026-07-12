#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path


EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
}
EXCLUDED_TOP_LEVEL = {"data", "dist"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".part"}


def _include(path: Path, root: Path, include_runs: bool) -> bool:
    relative = path.relative_to(root)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.parts and relative.parts[0] in EXCLUDED_TOP_LEVEL:
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    if relative.as_posix() == "PACKAGE_MANIFEST.json":
        return False
    if relative.parts and relative.parts[0] == "outputs":
        if len(relative.parts) >= 2 and relative.parts[1] == "runs":
            return include_runs
        if len(relative.parts) >= 2 and relative.parts[1] in {"results", "status", "manifests"}:
            return True
        return len(relative.parts) >= 3 and relative.parts[1] == "reports" and relative.parts[2] in {
            "smoke", "mechanism", "mechanism_final", "mechanism_one_layer"
        }
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a portable GraphAtlas source package.")
    parser.add_argument("--output", default="dist/graphatlas_project.zip")
    parser.add_argument("--include-runs", action="store_true", help="Include checkpoints under outputs/runs.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != output.resolve() and _include(path, root, args.include_runs)
    ]
    files.sort(key=lambda path: str(path.relative_to(root)))
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": "GraphAtlas",
        "file_count": len(files),
        "includes_training_runs": bool(args.include_runs),
        "entry_points": [
            "setup.sh",
            "run_smoke.sh",
            "run_mechanism.sh",
            "run_screening.sh",
            "run_main_tables.sh",
            "run_all_ablations.sh",
            "run_large_scale.sh",
            "run_paper.sh",
            "run_link.sh",
            "run_everything.sh",
            "run_full_benchmark.ps1",
            "run_full_benchmark.bat",
        ],
    }

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = Path("graphatlas_project") / path.relative_to(root)
            info = zipfile.ZipInfo.from_file(path, arcname=str(relative))
            if os.access(path, os.X_OK):
                info.external_attr = (stat.S_IFREG | 0o755) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        archive.writestr(
            "graphatlas_project/PACKAGE_MANIFEST.json",
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        )

    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{_sha256(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps({**metadata, "archive": str(output), "sha256_file": str(checksum_path)}, indent=2))


if __name__ == "__main__":
    main()
