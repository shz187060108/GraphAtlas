#!/usr/bin/env python
"""Offline-safe H2GB archive extraction and native cache conversion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"pdns": "h2gb_pdns", "mag-year": "h2gb_mag_year", "ieee-cis": "h2gb_ieee_cis"}
ARCHIVES = {"pdns": "PDNS.zip", "ieee-cis": "IEEE-CIS.zip"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member rejected: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(target)
    return extracted


def validate_payload(path: Path, dataset: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema_version", "dataset_name", "benchmark_family", "source_repository", "source_revision",
        "node_types", "edge_types", "task_entity", "x_dict", "num_nodes_dict", "edge_index_dict",
        "target_y", "train_mask", "val_mask", "test_mask", "primary_metric", "metadata",
    }
    missing = required - set(payload)
    if missing or payload.get("dataset_name") != dataset:
        raise RuntimeError(f"Invalid bridge output for {dataset}; missing={sorted(missing)}")
    return payload


def convert(name: str, source_root: Path, output_root: Path, h2gb_python: Path, offline: bool) -> None:
    dataset = NAMES[name]
    processed = output_root / dataset / "processed"
    cache = processed / "graphatlas_h2gb.pt"
    manifest = processed / "manifest.json"
    if cache.exists() and manifest.exists():
        validate_payload(cache, dataset)
        print(f"{dataset}: verified existing cache")
        return
    if not h2gb_python.exists():
        raise FileNotFoundError(f"H2GB interpreter does not exist: {h2gb_python}")
    raw_root = source_root / ("ogb" if name == "mag-year" else ("PDNS" if name == "pdns" else "IEEE-CIS"))
    source_files: list[Path] = []
    if name in ARCHIVES:
        archive = source_root / "archives" / ARCHIVES[name]
        if not archive.exists():
            if offline:
                raise FileNotFoundError(f"Offline H2GB archive missing: {archive}")
            raise RuntimeError("Online H2GB download is intentionally unsupported; provide the official archive")
        raw_dir = raw_root / "raw"
        if not raw_dir.exists() or not any(raw_dir.iterdir()):
            with tempfile.TemporaryDirectory(dir=source_root) as temp:
                stage = Path(temp) / "extract"
                source_files = safe_extract(archive, stage)
                extracted_raw = stage / "raw"
                if not extracted_raw.is_dir():
                    raise RuntimeError(f"Official archive {archive.name} does not contain a raw/ directory")
                raw_dir.parent.mkdir(parents=True, exist_ok=True)
                os.replace(extracted_raw, raw_dir)
        source_files = [archive]
    elif not raw_root.exists():
        raise FileNotFoundError(f"Offline ogbn-mag cache missing: {raw_root}")
    else:
        source_files = [path for path in raw_root.rglob("*") if path.is_file()]
    processed.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=processed) as temp:
        temporary = Path(temp) / "graphatlas_h2gb.pt"
        command = [
            str(h2gb_python), "-u", str(ROOT / "scripts" / "h2gb_bridge.py"),
            "--dataset", dataset, "--root", str(raw_root), "--output", str(temporary),
        ]
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            raise RuntimeError(f"H2GB bridge failed for {dataset} with exit code {result.returncode}")
        payload = validate_payload(temporary, dataset)
        hashes = {str(path.relative_to(source_root)): sha256(path) for path in source_files if path.is_file()}
        manifest_payload = {
            "schema_version": 1,
            "dataset_name": dataset,
            "benchmark_family": "H2GB",
            "source_repository": "junhongmit/H2GB",
            "source_revision": payload["source_revision"],
            "source_files": sorted(hashes),
            "source_sha256": hashes,
            "cache_sha256": sha256(temporary),
            "node_types": payload["node_types"],
            "edge_types": [list(value) for value in payload["edge_types"]],
            "task_entity": payload["task_entity"],
            "primary_metric": payload["primary_metric"],
        }
        manifest_temp = Path(temp) / "manifest.json"
        manifest_temp.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, cache)
        os.replace(manifest_temp, manifest)
    print(f"{dataset}: converted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(NAMES), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--h2gb-python", type=Path, default=os.environ.get("H2GB_PYTHON"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.h2gb_python is None:
        raise SystemExit("H2GB_PYTHON is not set; pass --h2gb-python explicitly")
    failures = []
    for name in args.datasets:
        try:
            convert(name, args.source_root, args.output_root, args.h2gb_python, args.offline)
        except Exception as error:
            failures.append(f"{name}: {error}")
            print(f"ERROR {name}: {error}")
    if failures:
        raise SystemExit("H2GB conversion failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
