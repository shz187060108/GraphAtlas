#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphatlas.datasets.h2gb import load_h2gb_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/h2gb"))
    args = parser.parse_args()
    failures = []
    for name in ("h2gb_pdns", "h2gb_mag_year", "h2gb_ieee_cis"):
        try:
            data = load_h2gb_dataset(name, args.root.parent)
            manifest = json.loads((args.root / name / "processed" / "manifest.json").read_text(encoding="utf-8"))
            print(f"{name}: ok nodes={sum(data.num_nodes_dict.values())} relations={len(data.edge_types)} revision={manifest['source_revision']}")
        except Exception as error:
            failures.append(f"{name}: {error}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
