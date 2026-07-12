#!/usr/bin/env python
from __future__ import annotations

try:
    from _bootstrap import PROJECT_ROOT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT_ROOT

import argparse
from pathlib import Path

from graphatlas.utils import load_yaml, save_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Enable only datasets already materialized as standardized NPZ files.")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    source = Path(args.preset)
    if not source.suffix:
        source = PROJECT_ROOT / "configs" / "presets" / f"{source}.yaml"
    elif not source.is_absolute():
        source = PROJECT_ROOT / source
    payload = load_yaml(source)
    active = 0
    for entry in payload.get("datasets", []):
        name = entry["name"].lower().replace("-", "_")
        if name == "atlas_het":
            entry["enabled"] = True
            active += 1
            continue
        exists = (PROJECT_ROOT / args.data_root / name / "raw" / f"{name}.npz").exists()
        entry["enabled"] = bool(exists)
        active += int(exists)
    output = Path(args.output) if args.output else source.with_name(source.stem + "_available.yaml")
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    save_yaml(payload, output)
    print(f"wrote {output} with {active} active dataset entries")

if __name__ == "__main__":
    main()
