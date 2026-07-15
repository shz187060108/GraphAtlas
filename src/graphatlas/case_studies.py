"""Deterministic node-level case selection and compact ego export helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:  # plotting/data-only tests need not import the CUDA build of torch
    import torch
except Exception:  # pragma: no cover - environment-dependent fallback
    torch = None


CASE_NAMES = ("high-risk", "low-risk", "rescued", "failed", "high-opportunity", "confident-error")


def _as_numpy(value: Any, n: int | None = None) -> np.ndarray:
    if value is None:
        return np.zeros(0 if n is None else n, dtype=float)
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _wrong(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if labels.ndim == 2:
        return np.any((probabilities >= 0.5) != labels, axis=1)
    return probabilities.argmax(axis=-1) != labels.astype(int)


def _top(indices: np.ndarray, values: np.ndarray, count: int, descending: bool) -> np.ndarray:
    if not len(indices):
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((indices, -values if descending else values))
    return indices[order[:count]].astype(np.int64)


def select_case_nodes(probabilities: Any, labels: Any, test_mask: Any, diagnostics: dict[str, Any] | None = None,
                      baseline_probabilities: Any | None = None, cases_per_category: int = 8) -> dict[str, np.ndarray]:
    """Select identical, deterministic categories from node arrays."""
    probs = _as_numpy(probabilities); y = _as_numpy(labels); mask = _as_numpy(test_mask).astype(bool)
    n = len(probs); ids = np.flatnonzero(mask)
    diag = diagnostics or {}
    risk = _as_numpy(diag.get("node_transport_risk"), n); opportunity = _as_numpy(diag.get("node_cross_chart_mass"), n) * _as_numpy(diag.get("node_q_spread"), n)
    if len(risk) != n: risk = np.zeros(n, dtype=float)
    if len(opportunity) != n: opportunity = np.zeros(n, dtype=float)
    wrong = _wrong(probs, y)
    result = {
        "high-risk": _top(ids, risk[ids], cases_per_category, True),
        "low-risk": _top(ids, risk[ids], cases_per_category, False),
        "high-opportunity": _top(ids, opportunity[ids], cases_per_category, True),
        "confident-error": _top(ids[wrong[ids]], np.max(probs[ids[wrong[ids]]], axis=-1) if len(probs) and probs.ndim > 1 else np.zeros(int(wrong[ids].sum())), cases_per_category, True),
        "rescued": np.zeros(0, dtype=np.int64),
        "failed": np.zeros(0, dtype=np.int64),
    }
    if baseline_probabilities is not None:
        base_wrong = _wrong(_as_numpy(baseline_probabilities), y)
        result["rescued"] = _top(ids[base_wrong[ids] & ~wrong[ids]], risk[ids[base_wrong[ids] & ~wrong[ids]]], cases_per_category, True)
        result["failed"] = _top(ids[~base_wrong[ids] & wrong[ids]], risk[ids[~base_wrong[ids] & wrong[ids]]], cases_per_category, True)
    return result


def two_hop_ego(edge_index: Any, center: int, max_nodes: int = 300) -> tuple[np.ndarray, np.ndarray, bool]:
    edges = _as_numpy(edge_index).astype(np.int64)
    if edges.size == 0:
        return np.asarray([center], dtype=np.int64), np.zeros((2, 0), dtype=np.int64), False
    adjacency: dict[int, set[int]] = {}
    for src, dst in edges.T:
        adjacency.setdefault(int(src), set()).add(int(dst)); adjacency.setdefault(int(dst), set()).add(int(src))
    selected = {int(center)}; frontier = {int(center)}
    for _ in range(2):
        frontier = {neighbor for node in frontier for neighbor in adjacency.get(node, set())} - selected
        selected.update(frontier)
    nodes = np.asarray(sorted(selected), dtype=np.int64); truncated = len(nodes) > max_nodes
    if truncated: nodes = nodes[:max_nodes]
    lookup = {int(node): index for index, node in enumerate(nodes)}
    keep = [(lookup[int(src)], lookup[int(dst)]) for src, dst in edges.T if int(src) in lookup and int(dst) in lookup]
    local_edges = np.asarray(keep, dtype=np.int64).T if keep else np.zeros((2, 0), dtype=np.int64)
    return nodes, local_edges, truncated


def case_rows(selected: dict[str, np.ndarray], probabilities: Any, labels: Any, diagnostics: dict[str, Any] | None = None) -> pd.DataFrame:
    probs = _as_numpy(probabilities); labels_np = _as_numpy(labels); diag = diagnostics or {}
    rows = []
    for category, nodes in selected.items():
        for rank, node in enumerate(nodes):
            diagnostic_values = {}
            for name, value in diag.items():
                array = _as_numpy(value)
                if array.ndim > 0 and len(array) > int(node):
                    diagnostic_values[name] = float(array[node])
            rows.append({"case_category": category, "node_id": int(node), "rank": rank, "label": json.dumps(labels_np[node].tolist() if labels_np.ndim > 1 else int(labels_np[node])), "probability": json.dumps(probs[node].tolist()), **diagnostic_values})
    return pd.DataFrame(rows)


def export_case_manifest(records: list[dict[str, Any]], output_dir: str | Path) -> Path:
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    path = root / "case_manifest.json"; path.write_text(json.dumps({"cases": records}, indent=2, ensure_ascii=False), encoding="utf-8"); return path
