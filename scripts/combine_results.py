#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine GraphAtlas result tables without duplicate runs.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frames = []
    for value in args.input:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "run_id" in combined:
        combined = combined.drop_duplicates("run_id", keep="last")
    sort_columns = [column for column in ["dataset", "task", "model", "split", "seed"] if column in combined]
    if sort_columns:
        combined = combined.sort_values(sort_columns).reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print(f"Wrote {len(combined)} runs to {output}")


if __name__ == "__main__":
    main()
