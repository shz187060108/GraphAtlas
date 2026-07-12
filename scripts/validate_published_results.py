#!/usr/bin/env python
from __future__ import annotations
from _bootstrap import PROJECT_ROOT
import argparse, json
from pathlib import Path
from graphatlas.published_results import load_published_results

def main():
    p=argparse.ArgumentParser(description="Validate curated published baseline results and provenance metadata.")
    p.add_argument("--input", action="append", default=["published_results/data"])
    p.add_argument("--output", default="outputs/results/published_results_validated.csv")
    a=p.parse_args(); root=Path(PROJECT_ROOT)
    paths=[root/x if not Path(x).is_absolute() else Path(x) for x in a.input]
    frame=load_published_results(paths)
    out=root/a.output if not Path(a.output).is_absolute() else Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(out,index=False)
    print(json.dumps({"rows":len(frame),"sources":frame.source_id.nunique(),"datasets":frame.dataset.nunique(),"output":str(out)},indent=2))
if __name__=="__main__": main()
