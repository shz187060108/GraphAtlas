#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import PROJECT_ROOT
import argparse, json
from pathlib import Path
from graphatlas.published_results import build_comparison_bundle

def main():
    p=argparse.ArgumentParser(description="Build strictly provenance-aware GraphAtlas versus published-result tables.")
    p.add_argument("--results", required=True)
    p.add_argument("--published", action="append", default=["published_results/data"])
    p.add_argument("--protocols", default="configs/comparison_protocols.yaml")
    p.add_argument("--output-dir", required=True)
    a=p.parse_args(); root=Path(PROJECT_ROOT)
    def R(x):
        q=Path(x); return q if q.is_absolute() else root/q
    outputs=build_comparison_bundle(R(a.results), [R(x) for x in a.published], R(a.protocols), R(a.output_dir))
    print(json.dumps({k:str(v) for k,v in outputs.items()}, indent=2))
if __name__=="__main__": main()
