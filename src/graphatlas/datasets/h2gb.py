from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from graphatlas.data import HeteroGraphData


ALIASES = {
    "h2gb_pdns": "h2gb_pdns",
    "pdns": "h2gb_pdns",
    "h2gb_mag_year": "h2gb_mag_year",
    "mag_year": "h2gb_mag_year",
    "mag-year": "h2gb_mag_year",
    "h2gb_ieee_cis": "h2gb_ieee_cis",
    "ieee_cis": "h2gb_ieee_cis",
    "ieee-cis": "h2gb_ieee_cis",
}


def is_h2gb_name(name: str) -> bool:
    return name.lower().replace("-", "_") in {"h2gb_pdns", "h2gb_mag_year", "h2gb_ieee_cis"}


def _edge_type(value: Any) -> tuple[str, str, str]:
    if isinstance(value, tuple) and len(value) == 3:
        return tuple(str(part) for part in value)  # type: ignore[return-value]
    if isinstance(value, list) and len(value) == 3:
        return tuple(str(part) for part in value)  # type: ignore[return-value]
    raise ValueError(f"Invalid H2GB relation key: {value!r}")


def load_h2gb_dataset(name: str, root: str | Path = "data") -> HeteroGraphData:
    normalized = name.lower().replace("-", "_")
    canonical = ALIASES.get(normalized, normalized)
    processed = Path(root) / "h2gb" / canonical / "processed"
    cache_path = processed / "graphatlas_h2gb.pt"
    manifest_path = processed / "manifest.json"
    if not cache_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Native H2GB cache for {canonical} is missing. Run scripts/download_h2gb.py explicitly."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != 1 or int(manifest.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported H2GB cache schema for {canonical}")
    if payload.get("dataset_name") != canonical or payload.get("benchmark_family") != "H2GB":
        raise ValueError(f"H2GB cache identity mismatch for {canonical}")
    num_nodes_dict = {str(key): int(value) for key, value in payload["num_nodes_dict"].items()}
    global_ids = payload.get("global_node_id_dict") or {
        node_type: torch.arange(count, dtype=torch.long)
        for node_type, count in num_nodes_dict.items()
    }
    data = HeteroGraphData(
        x_dict={str(key): value for key, value in payload["x_dict"].items()},
        num_nodes_dict=num_nodes_dict,
        edge_index_dict={_edge_type(key): value.long() for key, value in payload["edge_index_dict"].items()},
        task_entity=str(payload["task_entity"]),
        y=payload["target_y"],
        train_mask=payload["train_mask"].bool(),
        val_mask=payload["val_mask"].bool(),
        test_mask=payload["test_mask"].bool(),
        metadata={**dict(payload.get("metadata", {})), "primary_metric": payload["primary_metric"], "manifest": manifest},
        global_node_id_dict={str(key): value.long() for key, value in global_ids.items()},
    )
    data.validate()
    return data
