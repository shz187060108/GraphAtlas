#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Show GraphAtlas experiment progress from status files.")
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()
    outputs = Path(args.outputs) / "status"
    files = sorted(outputs.glob("experiment_*.json"))
    if not files:
        print("No experiment status files found.")
        return
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        total = int(payload.get("expected_runs", 0))
        completed = int(payload.get("completed_runs", 0))
        failed = int(payload.get("failed_runs", 0))
        fraction = float(payload.get("progress_fraction", 0.0))
        width = 30
        filled = min(width, max(0, round(width * fraction)))
        bar = "#" * filled + "." * (width - filled)
        print(f"{path.stem:32s} [{bar}] {completed + failed}/{total}  failed={failed}")


if __name__ == "__main__":
    main()
