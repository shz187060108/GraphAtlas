#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from graphatlas.config import ExperimentConfig
from graphatlas.datasets.synthetic import generate_atlas_het
from graphatlas.nn.model import build_model
from graphatlas.utils import configure_torch_threads


MODEL_NAMES = ("graphatlas", "ambient_vector_gnn", "signature_gnn", "geometry_moe")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit matched-baseline model capacity and CPU forward cost.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default="outputs/reports/model_capacity")
    args = parser.parse_args()

    root = Path(PROJECT_ROOT)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = ExperimentConfig.from_yaml(config_path)
    dataset_config = replace(
        config.dataset,
        name="atlas_het",
        num_nodes=96,
        knn=min(config.dataset.knn, 5),
        relation_edges_per_node=1,
    )
    data = generate_atlas_het(dataset_config, seed=config.train.seed).to("cpu")
    configure_torch_threads(1)

    rows: list[dict[str, object]] = []
    for name in MODEL_NAMES:
        model_config = replace(config.model, name=name, num_charts=dataset_config.num_charts)
        model = build_model(data.num_features, data.num_classes, model_config, data.num_nodes).cpu().eval()
        parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        with torch.no_grad():
            model(data, compact=True)
            started = time.perf_counter()
            output = model(data, compact=True)
            elapsed_ms = 1000.0 * (time.perf_counter() - started)
        rows.append(
            {
                "model": name,
                "parameters": parameters,
                "forward_cpu_ms": elapsed_ms,
                "logits_shape": list(output["logits"].shape),
                "embedding_shape": list(output["embedding"].shape),
            }
        )

    graphatlas_parameters = int(rows[0]["parameters"])
    for row in rows:
        row["relative_to_graphatlas"] = int(row["parameters"]) / graphatlas_parameters

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "model_capacity.csv", index=False)
    (output_dir / "model_capacity.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    ambient_ratio = float(frame.loc[frame["model"] == "ambient_vector_gnn", "relative_to_graphatlas"].iloc[0])
    if not 0.85 <= ambient_ratio <= 1.15:
        raise RuntimeError(f"Ambient Vector GNN capacity ratio {ambient_ratio:.3f} is outside [0.85, 1.15]")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
