#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import PROJECT_ROOT
import argparse, subprocess, sys
from pathlib import Path
from graphatlas.utils import load_yaml, save_yaml


def main() -> None:
    parser=argparse.ArgumentParser(description="Enable locally materialized datasets in an optional preset and run it if non-empty.")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    args=parser.parse_args(); root=Path(PROJECT_ROOT)
    source=Path(args.preset)
    if not source.suffix: source=root/"configs"/"presets"/f"{source}.yaml"
    elif not source.is_absolute(): source=root/source
    payload=load_yaml(source); active=0
    for entry in payload.get("datasets",[]):
        name=entry["name"].lower().replace("-","_")
        if name=="atlas_het":
            entry["enabled"]=True; active+=1; continue
        available=(root/args.data_root/name/"raw"/f"{name}.npz").exists()
        entry["enabled"]=available; active+=int(available)
    generated=root/"outputs"/"active_presets"/f"{source.stem}_available.yaml"
    generated.parent.mkdir(parents=True,exist_ok=True); save_yaml(payload,generated)
    if active==0:
        print(f"SKIPPED {source.stem}: no standardized local datasets were found")
        return
    cmd=[sys.executable,"scripts/run_pipeline.py","--preset",str(generated)]
    if args.limit is not None: cmd += ["--limit",str(args.limit)]
    if args.continue_on_error: cmd.append("--continue-on-error")
    raise SystemExit(subprocess.run(cmd,cwd=root,check=False).returncode)
if __name__=="__main__": main()
