#!/usr/bin/env python
"""Record availability of optional external baseline repositories; never clones."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-external", action="store_true")
    parser.add_argument("--output", default="outputs/reports/external_baselines_availability.json")
    args = parser.parse_args()
    roots = {"neural_sheaf_diffusion": os.getenv("NSD_ROOT"), "bunn": os.getenv("BUNN_ROOT")}
    status = {name: {"root": root, "available": bool(root and Path(root).is_dir())} for name, root in roots.items()}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    unavailable = [name for name, value in status.items() if not value["available"]]
    if args.require_external and unavailable:
        raise SystemExit("Required external baselines unavailable: " + ", ".join(unavailable))


if __name__ == "__main__":
    main()
