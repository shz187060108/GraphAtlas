#!/usr/bin/env python
"""Export compact, deterministic local cases from existing predictions."""
from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graphatlas.case_studies import CASE_NAMES, case_rows, export_case_manifest, select_case_nodes, two_hop_ego
from graphatlas.config import ExperimentConfig
from graphatlas.datasets import load_dataset


def _load_prediction(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/case_studies")
    parser.add_argument("--baseline-predictions")
    parser.add_argument("--cases-per-category", "--max-cases", dest="cases_per_category", type=int, default=8)
    parser.add_argument("--max-ego-nodes", type=int, default=300)
    args = parser.parse_args()
    run_dir = Path(args.run_dir); root = Path(args.output_dir); (root / "tensors").mkdir(parents=True, exist_ok=True); (root / "metadata").mkdir(parents=True, exist_ok=True); (root / "supplementary").mkdir(parents=True, exist_ok=True)
    try:
        config = ExperimentConfig.from_yaml(run_dir / "config.yaml"); config.validate()
        data = load_dataset(config.dataset, config.train.seed)
        payload = _load_prediction(run_dir / "predictions.pt")
        probabilities = payload.get("probabilities")
        if probabilities is None and payload.get("logits") is not None:
            logits = payload["logits"]; probabilities = torch.sigmoid(logits) if getattr(data, "is_multilabel", False) else torch.softmax(logits, -1)
        if probabilities is None: raise ValueError("predictions.pt lacks probabilities")
        labels = payload.get("labels", data.y); masks = payload.get("masks", {"test": data.test_mask})
        test_mask = masks.get("test", masks.get("test_mask", data.test_mask)) if isinstance(masks, dict) else data.test_mask
        artifact_keys = (
            "embedding", "membership", "coordinates", "node_cross_chart_mass",
            "node_q_spread", "node_transport_risk",
        )
        diagnostics = {key: payload[key] for key in artifact_keys if key in payload}
        baseline = _load_prediction(Path(args.baseline_predictions)).get("probabilities") if args.baseline_predictions else None
        selected = select_case_nodes(probabilities, labels, test_mask, diagnostics, baseline, args.cases_per_category)
        rows = case_rows(selected, probabilities, labels, diagnostics); rows["dataset"] = config.dataset.name; rows["run_id"] = run_dir.name
        case_ids = {(category, int(node)): f"{config.dataset.name}__{run_dir.name}__{category.replace('-', '_')}__n{int(node)}"
                    for category, nodes in selected.items() for node in nodes}
        rows["case_id"] = [case_ids[(row.case_category, int(row.node_id))] for row in rows.itertuples()]
        index_path = root / "case_index.csv"; rows.to_csv(index_path, index=False)
        records = []
        tensor_dir = root / "tensors"; metadata_dir = root / "metadata"
        for category, nodes in selected.items():
            for node in nodes:
                global_nodes, local_edges, truncated = two_hop_ego(data.edge_index, int(node), args.max_ego_nodes)
                case_id = f"{config.dataset.name}__{run_dir.name}__{category.replace('-', '_')}__n{int(node)}"
                arrays = {
                    "node_ids": global_nodes,
                    "edge_index": local_edges,
                    "probabilities": np.asarray(probabilities)[global_nodes],
                    "labels": np.asarray(labels)[global_nodes],
                }
                for key, value in diagnostics.items():
                    array = np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value)
                    if array.ndim > 0 and len(array) == len(probabilities):
                        arrays[key] = array[global_nodes]
                np.savez_compressed(tensor_dir / f"{case_id}.npz", **arrays)
                metadata = {"case_id": case_id, "category": category, "center_node": int(node), "dataset": config.dataset.name, "run_id": run_dir.name, "truncated": bool(truncated)}
                (metadata_dir / f"{case_id}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8"); records.append(metadata)
        export_case_manifest(records, root); rows.to_csv(root / "supplementary" / "gallery_index.csv", index=False); pd.DataFrame(columns=["run_dir", "reason"]).to_csv(root / "skipped_cases.csv", index=False)
        print(json.dumps({"cases": len(records), "index": str(index_path)}, indent=2))
    except Exception as exc:
        pd.DataFrame([{"run_dir": str(run_dir), "reason": f"{type(exc).__name__}: {exc}"}]).to_csv(root / "skipped_cases.csv", index=False)
        export_case_manifest([], root)
        print(json.dumps({"cases": 0, "skipped": str(root / "skipped_cases.csv"), "reason": str(exc)}, indent=2))


if __name__ == "__main__":
    main()
