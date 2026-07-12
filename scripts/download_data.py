#!/usr/bin/env python
from __future__ import annotations

try:
    from _bootstrap import PROJECT_ROOT  # noqa: F401
except ModuleNotFoundError:  # imported as scripts.download_data in tests
    from scripts._bootstrap import PROJECT_ROOT  # type: ignore[no-redef]  # noqa: F401

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from graphatlas.datasets.real import SUPPORTED_NAMES, download_dataset, validate_dataset_file
from graphatlas.utils import load_yaml


MAIN_DATASETS = [
    "roman_empire",
    "amazon_ratings",
    "minesweeper",
    "tolokers",
    "questions",
    "actor",
    "chameleon_filtered",
    "squirrel_filtered",
]


def _normalize(name: str) -> str:
    return name.lower().replace("-", "_").strip()


def _preset_path(value: str, root: Path) -> Path:
    path = Path(value)
    if not path.suffix:
        path = root / "configs" / "presets" / f"{value}.yaml"
    elif not path.is_absolute():
        path = root / path
    return path.resolve()


def _datasets_from_preset(value: str, root: Path) -> list[str]:
    preset = load_yaml(_preset_path(value, root))
    names = [_normalize(entry["name"]) for entry in preset.get("datasets", []) if entry.get("enabled", True)]
    return [name for name in names if name != "atlas_het"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate GraphAtlas real-data benchmarks.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--main", action="store_true")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--preset", action="append", default=[], help="Download real datasets used by a preset.")
    parser.add_argument("--root", default="data")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]

    requested: list[str] = []
    if args.all:
        requested.extend(sorted(SUPPORTED_NAMES))
    if args.main:
        requested.extend(MAIN_DATASETS)
    requested.extend(args.dataset)
    for preset in args.preset:
        requested.extend(_datasets_from_preset(preset, project_root))
    names = list(dict.fromkeys(_normalize(name) for name in requested))
    if not names:
        parser.error("Select --all, --main, --preset, or at least one --dataset.")

    unsupported = sorted(set(names) - SUPPORTED_NAMES)
    if unsupported:
        parser.error(f"Unsupported datasets: {unsupported}")

    root = Path(args.root)
    for name in tqdm(names, desc="Datasets", unit="dataset"):
        path = root / name / "raw" / f"{name}.npz"
        if args.verify_only:
            info = validate_dataset_file(path)
        else:
            download_dataset(name, root)
            info = validate_dataset_file(path)
        tqdm.write(f"{name}: {info}")


if __name__ == "__main__":
    main()
